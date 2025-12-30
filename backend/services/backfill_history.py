# backend/services/backfill_history.py
"""
backfill_history.py — v3.6
(Rolling-Native, Normalized Batch StockAnalysis Bundle, HF History Bootstrap + yfinance Fallback)

Purpose
- Refreshes and repairs ticker data directly inside Rolling cache (rolling_body.json.gz).
- BATCH fetches metrics from StockAnalysis (parallel /s/d/<metric> requests).
- Uses /s/i only for basic metadata (symbol, name, price, volume, marketCap, peRatio, industry).
- Uses /s/d/<metric> for everything else (incl. open/high/low/close, rsi, growth, etc.).
- Normalizes all fetched field names before saving (camelCase → snake_case, rsi → rsi_14).
- Writes directly into rolling_body.json.gz using backend.core.data_pipeline helpers.

History bootstrap (when history is too short)
- Preferred: Hugging Face dataset "bwzheng2010/yahoo-finance-data" queried via DuckDB over remote Parquet.
  NEW in v3.6: **batched HF bootstrap** — pulls many symbols per DuckDB query to amortize remote Parquet overhead.
- Fallback: yfinance with a hard global rate limiter (default ~20 calls/min) + retry backoff.

Safety / invariants
- History is append-only with per-date dedupe (never wipes existing history).
- Rolling cache is never pruned by this service. (We may LIMIT what we fetch from HF/YF, but we never delete data already stored.)
- StockAnalysis is still used to write a "today" bar and fresh snapshot fields.
- Recent-day patching via yfinance is available but OFF by default (see env vars below).

Requirements (only if you enable HF bootstrap)
- duckdb >= 0.10 (recommended): pip install duckdb
"""

from __future__ import annotations

import gzip
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
import yfinance as yf

from backend.core.config import PATHS
from backend.core.data_pipeline import _read_rolling, log, save_rolling
from backend.services.metrics_fetcher import build_latest_metrics
from utils.progress_bar import progress_bar

UNIVERSE_FILE = PATHS["universe"] / "master_universe.json"

# -------------------------------------------------------------------
# Verbosity controls (tune as you like)
# -------------------------------------------------------------------
VERBOSE_BOOTSTRAP = False          # per-ticker “Bootstrapped history for XYZ…”
VERBOSE_BOOTSTRAP_ERRORS = False   # per-ticker bootstrap failure messages

# -------------------------------------------------------------------
# HF / yfinance bootstrap controls (env overridable)
# -------------------------------------------------------------------
# Prefer HF dataset bootstrap when available.
USE_HF_BOOTSTRAP = os.getenv("AION_USE_HF_BOOTSTRAP", "1").strip() not in {"0", "false", "False"}

# DuckDB Parquet source for HF dataset (can be overridden to a local file path).
HF_STOCK_PRICES_PARQUET = os.getenv(
    "AION_HF_STOCK_PRICES_PARQUET",
    "https://huggingface.co/datasets/bwzheng2010/yahoo-finance-data/resolve/main/data/stock_prices.parquet",
).strip()

# Limit concurrent HF queries (to avoid bandwidth spikes). 0/1 = effectively serial.
HF_MAX_CONCURRENCY = int(os.getenv("AION_HF_MAX_CONCURRENCY", "2"))
_HF_SEMA = threading.Semaphore(max(1, HF_MAX_CONCURRENCY))

# NEW: batched HF bootstrap (amortize remote Parquet overhead).
HF_BATCH_SIZE = int(os.getenv("AION_HF_BATCH_SIZE", "75"))  # symbols per DuckDB query
HF_PREFETCH_ENABLED = os.getenv("AION_HF_PREFETCH_ENABLED", "1").strip() not in {"0", "false", "False"}

# How far back we scan in HF dataset (bounds remote scan work).
HF_LOOKBACK_YEARS = int(os.getenv("AION_HF_LOOKBACK_YEARS", "6"))

# yfinance global rate limiter:
#  - default: ~1 call / 3.2s ≈ 18.75 calls/min
YF_MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("AION_YF_MIN_SECONDS_BETWEEN_CALLS", "3.2"))
YF_MAX_RETRIES = int(os.getenv("AION_YF_MAX_RETRIES", "4"))
YF_BACKOFF_BASE_SECONDS = float(os.getenv("AION_YF_BACKOFF_BASE_SECONDS", "6.0"))

# Optional: patch the most recent N calendar days with yfinance when HF data is stale.
# WARNING: doing this for thousands of symbols will take hours at ~20 calls/min.
YF_PATCH_RECENT_DAYS = int(os.getenv("AION_YF_PATCH_RECENT_DAYS", "0"))  # 0 = off
YF_PATCH_MAX_SYMBOLS = int(os.getenv("AION_YF_PATCH_MAX_SYMBOLS", "0"))  # 0 = no patching

# -------------------------------------------------------------------
# StockAnalysis endpoints
# -------------------------------------------------------------------
SA_BASE = "https://stockanalysis.com/api/screener"

SA_INDEX_FIELDS = [
    "symbol", "name", "price", "change", "volume",
    "marketCap", "peRatio", "industry",
]

SA_METRICS = [
    "rsi", "ma50", "ma200",
    "pbRatio", "psRatio", "pegRatio",
    "beta",
    "fcfYield", "earningsYield", "dividendYield",
    "revenueGrowth", "epsGrowth",
    "profitMargin", "operatingMargin", "grossMargin",
    "debtEquity", "debtEbitda",
    "sector", "float", "sharesOut",
    "ch1w", "ch1m", "ch3m", "ch6m", "ch1y", "chYTD",
    "open", "high", "low", "close",
]

# How many days of history to FETCH from HF/YF when bootstrapping.
# NOTE: This does NOT prune rolling. It only limits bootstrap fetch volume.
MAX_HISTORY_DAYS = int(os.getenv("AION_BOOTSTRAP_MAX_DAYS", "750"))

# Directory for audit bundle
METRICS_BUNDLE_DIR = Path("data") / "metrics_cache" / "bundle"
METRICS_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

# Thread-safety: rolling is a shared dict mutated by multiple workers.
_ROLLING_LOCK = threading.Lock()


# -------------------------------------------------------------------
# Rolling read retry (prevents transient empty reads during atomic replace)
# -------------------------------------------------------------------
def _read_rolling_with_retry(attempts: int = 5, sleep_secs: float = 0.5) -> Dict[str, Any]:
    last: Dict[str, Any] = {}
    for i in range(max(1, int(attempts))):
        last = _read_rolling() or {}
        if last:
            return last
        try:
            rp = Path(PATHS.get("rolling_body") or PATHS.get("rolling") or "")
            if rp and rp.exists() and rp.stat().st_size > 0:
                log(f"[backfill_history] ⏳ rolling_body present but empty-read (attempt {i+1}/{attempts}); retrying…")
            else:
                return {}
        except Exception:
            return {}
        time.sleep(float(sleep_secs))
    return last or {}


# -------------------------------------------------------------------
# Universe
# -------------------------------------------------------------------
def load_universe() -> list[str]:
    if not UNIVERSE_FILE.exists():
        log(f"⚠️ Universe file not found at {UNIVERSE_FILE}")
        return []
    try:
        with open(UNIVERSE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "symbols" in data:
            return data["symbols"]
        if isinstance(data, list):
            return data
    except Exception as e:
        log(f"⚠️ Failed to load universe: {e}")
    return []


# -------------------------------------------------------------------
# HTTP helpers (StockAnalysis)
# -------------------------------------------------------------------
def _sa_post_json(path: str, payload: dict | None = None, timeout: int = 20) -> dict | None:
    url = f"{SA_BASE}/{path.strip('/')}"
    try:
        if payload is not None:
            r = requests.post(url, json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log(f"⚠️ SA request failed for {url}: {e}")
    return None


_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}


def _fetch_from_stockanalysis(sym: str) -> Dict[str, Any] | None:
    global _INDEX_CACHE
    sym = sym.upper()

    if not _INDEX_CACHE:
        payload = {
            "fields": SA_INDEX_FIELDS,
            "filter": {"exchange": "all"},
            "order": ["marketCap", "desc"],
            "offset": 0,
            "limit": 10000,
        }
        js = _sa_post_json("s/i", payload)
        rows = (js or {}).get("data", {}).get("data", [])
        for row in rows:
            rsym = (row.get("symbol") or row.get("s") or "").upper()
            if not rsym:
                continue
            _INDEX_CACHE[rsym] = {
                "symbol": rsym,
                "name": row.get("name") or row.get("n"),
                "price": row.get("price"),
                "change": row.get("change"),
                "volume": row.get("volume"),
                "marketCap": row.get("marketCap"),
                "pe_ratio": row.get("peRatio"),
                "industry": row.get("industry"),
            }

    return _INDEX_CACHE.get(sym)


def _fetch_sa_index_batch() -> Dict[str, Dict[str, Any]]:
    payload = {
        "fields": SA_INDEX_FIELDS,
        "filter": {"exchange": "all"},
        "order": ["marketCap", "desc"],
        "offset": 0,
        "limit": 10000,
    }
    js = _sa_post_json("s/i", payload)
    out: Dict[str, Dict[str, Any]] = {}
    try:
        rows = (js or {}).get("data", {}).get("data", [])
        for row in rows:
            sym = (row.get("symbol") or row.get("s") or "").upper()
            if not sym:
                continue
            out[sym] = {
                "symbol": sym,
                "name": row.get("name") or row.get("n"),
                "price": row.get("price"),
                "change": row.get("change"),
                "volume": row.get("volume"),
                "marketCap": row.get("marketCap"),
                "pe_ratio": row.get("peRatio"),
                "industry": row.get("industry"),
            }
    except Exception as e:
        log(f"⚠️ Failed to parse /s/i: {e}")
    return out


def _fetch_sa_metric(metric: str, timeout: int = 20) -> Dict[str, Any]:
    js = _sa_post_json(f"s/d/{metric}", timeout=timeout)
    out: Dict[str, Any] = {}
    try:
        rows = (js or {}).get("data", {}).get("data", [])
        for r in rows:
            if isinstance(r, list) and len(r) >= 2:
                out[str(r[0]).upper()] = r[1]
            elif isinstance(r, dict):
                sym = r.get("symbol") or r.get("s")
                val = r.get(metric)
                if sym:
                    out[str(sym).upper()] = val
    except Exception:
        pass
    return out


def _fetch_sa_metrics_bulk(metrics: Iterable[str], max_workers: int = 8) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    metrics = list(metrics)

    def _job(m: str):
        return m, _fetch_sa_metric(m)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_job, m) for m in metrics]
        for fut in as_completed(futs):
            m, tbl = fut.result()
            result[m] = tbl or {}
    return result


# -------------------------------------------------------------------
# Normalization helper
# -------------------------------------------------------------------
def _normalize_node_keys(node: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(node, dict):
        return node
    replacements = {
        "peRatio": "pe_ratio",
        "pbRatio": "pb_ratio",
        "psRatio": "ps_ratio",
        "pegRatio": "peg_ratio",
        "debtEquity": "debt_equity",
        "debtEbitda": "debt_ebitda",
        "revenueGrowth": "revenue_growth",
        "epsGrowth": "eps_growth",
        "profitMargin": "profit_margin",
        "operatingMargin": "operating_margin",
        "grossMargin": "gross_margin",
        "dividendYield": "dividend_yield",
        "fcfYield": "fcf_yield",
        "earningsYield": "earnings_yield",
        "rsi": "rsi_14",
        "sharesOut": "shares_outstanding",
    }
    for old, new in replacements.items():
        if old in node:
            node[new] = node.pop(old)
    return node


# -------------------------------------------------------------------
# Merge + bundle save
# -------------------------------------------------------------------
def _merge_index_and_metrics(
    index_map: Dict[str, Dict[str, Any]],
    metrics_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out = dict(index_map)
    for metric, tbl in (metrics_map or {}).items():
        for sym, val in (tbl or {}).items():
            if sym not in out:
                out[sym] = {"symbol": sym}
            out[sym][metric] = val
    return out


def _normalize_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    changed = 0
    for sym, node in bundle.items():
        before = set(node.keys())
        bundle[sym] = _normalize_node_keys(node)
        after = set(bundle[sym].keys())
        diff = len(after - before)
        if diff:
            changed += diff
    log(
        f"🔧 Normalization summary — {len(bundle)} tickers, "
        f"~{changed} fields normalized (rsi→rsi_14, sharesOut→shares_outstanding, etc.)."
    )
    return bundle


def _save_sa_bundle_snapshot(bundle: Dict[str, Any]) -> str | None:
    try:
        ts = datetime.utcnow().strftime("%Y-%m-%d")
        path = METRICS_BUNDLE_DIR / f"sa_bundle_{ts}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({"date": ts, "data": bundle}, f)
        log(f"✅ Saved StockAnalysis bundle → {path}")
        return str(path)
    except Exception as e:
        log(f"⚠️ Failed to save SA bundle: {e}")
        return None


def fetch_sa_bundle_parallel(max_workers: int = 8) -> Dict[str, Dict[str, Any]]:
    base = _fetch_sa_index_batch()
    if not base:
        log("⚠️ /s/i returned no rows.")
        return {}

    metrics_map = _fetch_sa_metrics_bulk(SA_METRICS, max_workers=max_workers)
    bundle = _merge_index_and_metrics(base, metrics_map)
    bundle = _normalize_bundle(bundle)
    _save_sa_bundle_snapshot(bundle)
    return bundle


# -------------------------------------------------------------------
# History bootstrap helpers (HF preferred, yfinance fallback)
# -------------------------------------------------------------------
_YF_RATE_LOCK = threading.Lock()
_YF_LAST_CALL_TS = 0.0


def _is_rate_limit_error(e: Exception) -> bool:
    name = e.__class__.__name__.lower()
    msg = str(e).lower()
    return ("ratelimit" in name) or ("rate limit" in msg) or ("too many requests" in msg)


def _yf_rate_limit_wait() -> None:
    global _YF_LAST_CALL_TS
    with _YF_RATE_LOCK:
        now = time.time()
        delta = now - _YF_LAST_CALL_TS
        if delta < YF_MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(max(0.0, YF_MIN_SECONDS_BETWEEN_CALLS - delta))
        _YF_LAST_CALL_TS = time.time()


def _bootstrap_history_yf(symbol: str, max_days: int = MAX_HISTORY_DAYS, period: str = "3y") -> List[Dict[str, Any]]:
    """
    Rate-limited yfinance download (fallback path).
    Returns list of {date, open, high, low, close, volume}.
    """
    symbol = symbol.upper()
    for attempt in range(max(1, YF_MAX_RETRIES)):
        try:
            _yf_rate_limit_wait()
            df = yf.download(
                tickers=symbol,
                interval="1d",
                period=period,
                auto_adjust=False,
                progress=False,
                threads=False,  # critical: avoid bursty parallel HTTP
            )
            if df is None or df.empty:
                return []

            cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[cols].dropna(how="all")

            bars: List[Dict[str, Any]] = []
            for idx, open_, high, low, close, volume in df.itertuples():
                bars.append(
                    {
                        "date": idx.strftime("%Y-%m-%d"),
                        "open": float(open_ or 0.0),
                        "high": float(high or 0.0),
                        "low": float(low or 0.0),
                        "close": float(close or 0.0),
                        "volume": float(volume or 0.0),
                    }
                )

            # This is a FETCH LIMIT, not a rolling prune.
            return bars[-max_days:] if bars else []

        except Exception as e:
            if _is_rate_limit_error(e):
                sleep_s = min(120.0, YF_BACKOFF_BASE_SECONDS * (2 ** attempt))
                if VERBOSE_BOOTSTRAP_ERRORS:
                    log(f"⚠️ YF rate limited for {symbol} (attempt {attempt+1}/{YF_MAX_RETRIES}) — sleeping {sleep_s:.0f}s")
                time.sleep(sleep_s)
                continue

            if VERBOSE_BOOTSTRAP_ERRORS:
                log(f"⚠️ YF bootstrap failed for {symbol}: {e}")
            return []

    return []


# ---------------- HF / DuckDB ----------------

_HF_DUCKDB_ERR = None

# Thread-local DuckDB connections (safer with ThreadPoolExecutor)
_HF_LOCAL = threading.local()


def _hf_duckdb_conn():
    """
    Create/get a per-thread DuckDB connection.
    Uses DuckDB httpfs extension when possible.
    """
    global _HF_DUCKDB_ERR
    con = getattr(_HF_LOCAL, "con", None)
    if con is not None:
        return con

    try:
        import duckdb  # type: ignore
    except Exception as e:
        _HF_DUCKDB_ERR = e
        return None

    try:
        con = duckdb.connect(database=":memory:")
        # Best effort: enable httpfs for HTTPS parquet
        try:
            con.execute("INSTALL httpfs;")
        except Exception:
            pass
        try:
            con.execute("LOAD httpfs;")
        except Exception:
            pass

        # A couple of safe perf toggles (no-op on older versions)
        try:
            con.execute("SET preserve_insertion_order=false;")
        except Exception:
            pass
        try:
            con.execute("SET threads=2;")
        except Exception:
            pass

        _HF_LOCAL.con = con
        return con
    except Exception as e:
        _HF_DUCKDB_ERR = e
        return None


def _to_date_str(x: Any) -> str:
    try:
        if hasattr(x, "strftime"):
            return x.strftime("%Y-%m-%d")
    except Exception:
        pass
    return str(x)[:10]


def _f(x: Any) -> float:
    try:
        v = float(x)
        return 0.0 if (not math.isfinite(v)) else v
    except Exception:
        return 0.0


def _bootstrap_history_hf(symbol: str, max_days: int = MAX_HISTORY_DAYS) -> List[Dict[str, Any]]:
    """
    Per-symbol HF bootstrap (kept for compatibility / fallback).
    Prefer using batched prefetch in v3.6.
    """
    symbol = symbol.upper()
    with _HF_SEMA:
        con = _hf_duckdb_conn()
        if con is None:
            return []

        start_date = (datetime.utcnow().date() - timedelta(days=365 * max(1, int(HF_LOOKBACK_YEARS)))).isoformat()

        try:
            rows = con.execute(
                """
                SELECT report_date, open, high, low, close, volume
                FROM read_parquet(?, columns=['symbol','report_date','open','high','low','close','volume'])
                WHERE symbol = ? AND report_date >= ?
                ORDER BY report_date DESC
                LIMIT ?
                """,
                [HF_STOCK_PRICES_PARQUET, symbol, start_date, int(max_days)],
            ).fetchall()
        except Exception as e:
            if VERBOSE_BOOTSTRAP_ERRORS:
                log(f"⚠️ HF/DuckDB bootstrap failed for {symbol}: {e}")
            return []

        if not rows:
            return []

        bars: List[Dict[str, Any]] = []
        for report_date, open_, high, low, close, volume in reversed(rows):
            bars.append(
                {
                    "date": _to_date_str(report_date),
                    "open": _f(open_),
                    "high": _f(high),
                    "low": _f(low),
                    "close": _f(close),
                    "volume": _f(volume),
                }
            )

        # This is a FETCH LIMIT, not a rolling prune.
        return bars[-max_days:] if bars else []


def _bootstrap_histories_hf_batch(symbols: Sequence[str], max_days: int = MAX_HISTORY_DAYS) -> Dict[str, List[Dict[str, Any]]]:
    """
    Batched HF bootstrap: fetch history for many symbols in ONE DuckDB query.

    Returns mapping: SYMBOL -> list[{date, open, high, low, close, volume}] sorted ascending by date.
    """
    syms = [str(s).upper() for s in symbols if str(s).strip()]
    if not syms:
        return {}

    # Limit HF concurrency to avoid a thundering herd of HTTPS range requests.
    with _HF_SEMA:
        con = _hf_duckdb_conn()
        if con is None:
            return {}

        start_date = (datetime.utcnow().date() - timedelta(days=365 * max(1, int(HF_LOOKBACK_YEARS)))).isoformat()

        # Build IN (?, ?, ...) safely for this batch.
        placeholders = ", ".join(["?"] * len(syms))
        sql = f"""
        WITH t AS (
            SELECT
                symbol,
                report_date,
                open, high, low, close, volume,
                row_number() OVER (PARTITION BY symbol ORDER BY report_date DESC) AS rn
            FROM read_parquet(?, columns=['symbol','report_date','open','high','low','close','volume'])
            WHERE symbol IN ({placeholders}) AND report_date >= ?
        )
        SELECT symbol, report_date, open, high, low, close, volume
        FROM t
        WHERE rn <= ?
        ORDER BY symbol, report_date
        """

        params: List[Any] = [HF_STOCK_PRICES_PARQUET]
        params.extend(syms)
        params.append(start_date)
        params.append(int(max_days))

        try:
            rows = con.execute(sql, params).fetchall()
        except Exception as e:
            if VERBOSE_BOOTSTRAP_ERRORS:
                log(f"⚠️ HF/DuckDB batch bootstrap failed for batch(size={len(syms)}): {e}")
            return {}

    out: Dict[str, List[Dict[str, Any]]] = {s: [] for s in syms}
    for sym, report_date, open_, high, low, close, volume in rows or []:
        su = str(sym).upper()
        if su not in out:
            out[su] = []
        out[su].append(
            {
                "date": _to_date_str(report_date),
                "open": _f(open_),
                "high": _f(high),
                "low": _f(low),
                "close": _f(close),
                "volume": _f(volume),
            }
        )
    # out lists are already ascending by date due to ORDER BY symbol, report_date
    return out


def _chunk(seq: Sequence[str], n: int) -> List[List[str]]:
    n = max(1, int(n))
    return [list(seq[i:i+n]) for i in range(0, len(seq), n)]


def _prefetch_hf_bootstrap_map(
    symbols: Sequence[str],
    *,
    batch_size: int = HF_BATCH_SIZE,
    max_workers: int = HF_MAX_CONCURRENCY,
    max_days: int = MAX_HISTORY_DAYS,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Prefetch HF histories for the given symbols using batched DuckDB queries.
    Designed to only be called for symbols that are actually missing enough history.
    """
    syms = [str(s).upper() for s in symbols if str(s).strip()]
    if not syms:
        return {}

    batches = _chunk(syms, max(1, int(batch_size)))
    workers = max(1, int(max_workers))
    workers = min(workers, len(batches))

    log(f"[backfill_history] 🧠 HF batched prefetch: symbols={len(syms)}, batch_size={batch_size}, batches={len(batches)}, workers={workers}")

    out: Dict[str, List[Dict[str, Any]]] = {}

    def _job(batch_syms: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        return _bootstrap_histories_hf_batch(batch_syms, max_days=max_days)

    # Parallelize batches (bounded by HF_MAX_CONCURRENCY)
    if workers <= 1 or len(batches) <= 1:
        for b in batches:
            out.update(_job(b))
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, b) for b in batches]
        for fut in progress_bar(as_completed(futs), desc="HF Prefetch", unit="batch", total=len(futs)):
            try:
                res = fut.result() or {}
                out.update(res)
            except Exception:
                continue
    return out


def _history_unique_days_at_least(hist: List[Dict[str, Any]], n: int) -> bool:
    """
    Fast-ish: count distinct dates and early-exit once >= n.
    Avoids building a giant set when histories are very long.
    """
    n = int(n)
    if n <= 0:
        return True
    seen: set[str] = set()
    for b in hist or []:
        d = b.get("date")
        if not d:
            continue
        ds = str(d)
        if ds not in seen:
            seen.add(ds)
            if len(seen) >= n:
                return True
    return False


def _unique_history_dates(hist: List[Dict[str, Any]]) -> set[str]:
    return {str(b.get("date")) for b in (hist or []) if b.get("date")}


def _merge_histories_prefer_existing(base: List[Dict[str, Any]], existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge two histories by date.
    - base: usually HF/YF bars (bootstrap data)
    - existing: bars already in rolling (preferred if conflict)

    IMPORTANT: Never prunes. (Append-only + dedupe, then sort.)
    """
    by_date: Dict[str, Dict[str, Any]] = {}
    for bar in base or []:
        d = str(bar.get("date") or "")
        if d:
            by_date[d] = bar
    for bar in existing or []:
        d = str(bar.get("date") or "")
        if d:
            by_date[d] = bar  # overwrite with existing
    merged = list(by_date.values())
    merged.sort(key=lambda x: x.get("date") or "")
    return merged


def _patch_recent_days_yf(symbol: str, hist: List[Dict[str, Any]], days: int) -> List[Dict[str, Any]]:
    """
    Optional: patch the last N calendar days using yfinance (rate-limited).
    Only useful if your HF snapshot is stale.
    """
    if days <= 0:
        return hist
    period = f"{max(1, int(days))}d"
    yf_bars = _bootstrap_history_yf(symbol, max_days=MAX_HISTORY_DAYS, period=period)
    if not yf_bars:
        return hist
    return _merge_histories_prefer_existing(base=yf_bars, existing=hist)


def _ensure_bootstrap_history_if_needed(
    symbol: str,
    hist: List[Dict[str, Any]],
    min_days: int,
    *,
    hf_prefetched: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """
    If history is too short (< min_days unique dates), bootstrap.

    Order:
      1) HF dataset via DuckDB/Parquet (batched prefetch map if provided)
      2) yfinance (rate-limited) fallback
      3) Optional recent-day patching via yfinance (OFF by default)

    NEVER prunes rolling history.
    """
    symbol = symbol.upper()
    if _history_unique_days_at_least(hist, int(min_days)):
        return hist

    bootstrap_bars: List[Dict[str, Any]] = []

    if USE_HF_BOOTSTRAP:
        if hf_prefetched and symbol in hf_prefetched:
            bootstrap_bars = hf_prefetched.get(symbol) or []
        else:
            bootstrap_bars = _bootstrap_history_hf(symbol, max_days=MAX_HISTORY_DAYS)

    if not bootstrap_bars:
        bootstrap_bars = _bootstrap_history_yf(symbol, max_days=MAX_HISTORY_DAYS, period="3y")

    if not bootstrap_bars:
        return hist

    merged = _merge_histories_prefer_existing(base=bootstrap_bars, existing=hist)

    if YF_PATCH_RECENT_DAYS > 0:
        merged = _patch_recent_days_yf(symbol, merged, days=YF_PATCH_RECENT_DAYS)

    if VERBOSE_BOOTSTRAP:
        log(f"🧪 Bootstrapped history for {symbol}: {len(merged)} total days (append-only).")

    return merged


# -------------------------------------------------------------------
# Local node helper (replaces ensure_symbol_fields)
# -------------------------------------------------------------------
def _ensure_symbol_node(rolling: Dict[str, Any], sym_u: str) -> Dict[str, Any]:
    node = rolling.get(sym_u)
    if not isinstance(node, dict):
        node = {"symbol": sym_u, "history": []}
    else:
        node.setdefault("symbol", sym_u)
        node.setdefault("history", [])
    rolling[sym_u] = node
    return node


# -------------------------------------------------------------------
# Main backfill routine (same external API)
# -------------------------------------------------------------------
def backfill_symbols(symbols: List[str], min_days: int = 180, max_workers: int = 8) -> int:
    """
    Backfill routine:
      - Always reads rolling_body first.
      - Only bootstraps HF/YF for symbols missing enough history.
      - Never prunes rolling history.
    """
    rolling = _read_rolling_with_retry() or {}
    today = datetime.utcnow().strftime("%Y-%m-%d")
    mode = "full"
    if not rolling:
        mode = "fallback"
        log("⚠️ Rolling cache missing/empty — running in fallback mode (will not overwrite existing rolling with empty).")
    log(f"🧩 Backfill mode: {mode.upper()} | Date: {today}")

    if not symbols:
        symbols = [s for s in rolling.keys() if not str(s).startswith("_")]

    symbols = [str(s).upper() for s in symbols if str(s).strip()]
    total = len(symbols)
    if not total:
        log("⚠️ No symbols to backfill.")
        return 0

    updated = 0
    start = time.time()

    # Determine which symbols actually need HF/YF bootstrap (missing history).
    need_bootstrap: List[str] = []
    for s in symbols:
        node = rolling.get(s)
        hist = node.get("history") if isinstance(node, dict) else []
        if not isinstance(hist, list):
            hist = []
        if not _history_unique_days_at_least(hist, int(min_days)):
            need_bootstrap.append(s)

    hf_prefetched: Dict[str, List[Dict[str, Any]]] = {}
    if USE_HF_BOOTSTRAP and HF_PREFETCH_ENABLED and need_bootstrap:
        try:
            hf_prefetched = _prefetch_hf_bootstrap_map(
                need_bootstrap,
                batch_size=HF_BATCH_SIZE,
                max_workers=max(1, HF_MAX_CONCURRENCY),
                max_days=MAX_HISTORY_DAYS,
            ) or {}
        except Exception as e:
            hf_prefetched = {}
            log(f"[backfill_history] ⚠️ HF prefetch failed (continuing with per-symbol HF/YF): {e}")

    # FULL / FALLBACK MODE — bundle-based refresh + bootstrap
    if mode in ("full", "fallback"):
        log(f"🔧 Starting rolling backfill for {total} symbols (batch SA fetch + HF/YF bootstrap where missing)…")
        sa_bundle = fetch_sa_bundle_parallel(max_workers=max_workers)
        if sa_bundle:
            try:
                build_latest_metrics()
            except Exception as e:
                log(f"⚠️ build_latest_metrics during backfill failed: {e}")
        else:
            log("⚠️ Empty SA bundle.")

        patch_symbols: set[str] = set()
        if YF_PATCH_RECENT_DAYS > 0 and YF_PATCH_MAX_SYMBOLS > 0:
            for s in symbols[:YF_PATCH_MAX_SYMBOLS]:
                patch_symbols.add(str(s).upper())
            log(f"🩹 YF recent patch enabled: days={YF_PATCH_RECENT_DAYS}, max_symbols={len(patch_symbols)}")

        def _process(sym_u: str) -> int:
            # Work off a local node copy to reduce shared-state hazards; write back under lock.
            with _ROLLING_LOCK:
                node = _ensure_symbol_node(rolling, sym_u)
                node_local = dict(node)

            hist = node_local.get("history") or []
            if not isinstance(hist, list):
                hist = []
            hist = _ensure_bootstrap_history_if_needed(sym_u, hist, min_days=min_days, hf_prefetched=hf_prefetched)

            if sym_u in patch_symbols:
                hist = _patch_recent_days_yf(sym_u, hist, days=YF_PATCH_RECENT_DAYS)

            sa = sa_bundle.get(sym_u) if sa_bundle else None
            if not sa:
                if hist:
                    node_local["history"] = hist
                    try:
                        last_bar = hist[-1]
                        node_local["close"] = last_bar.get("close")
                        node_local.setdefault("price", last_bar.get("close"))
                    except Exception:
                        pass
                    with _ROLLING_LOCK:
                        rolling[sym_u] = node_local
                    return 1
                with _ROLLING_LOCK:
                    rolling[sym_u] = node_local
                return 0

            latest_bar = {
                "date": today,
                "open": sa.get("open"),
                "high": sa.get("high"),
                "low": sa.get("low"),
                "close": sa.get("price") or sa.get("close"),
                "volume": sa.get("volume"),
            }

            # Append-only + per-date dedupe (never prune).
            by_date: Dict[str, Dict[str, Any]] = {}
            for bar in hist or []:
                d = str(bar.get("date") or "")
                if d:
                    by_date[d] = bar
            by_date[today] = latest_bar
            hist_new = list(by_date.values())
            hist_new.sort(key=lambda x: x.get("date") or "")

            node_local["history"] = hist_new
            node_local["close"] = latest_bar.get("close")
            # Merge SA snapshot fields (never deletes unknown keys)
            try:
                node_local.update(sa)
            except Exception:
                pass

            with _ROLLING_LOCK:
                rolling[sym_u] = node_local
            return 1

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_process, s): s for s in symbols}
            for fut in progress_bar(as_completed(futs), desc="Backfill (bundle+HF/YF)", unit="sym", total=total):
                try:
                    updated += int(fut.result() or 0)
                except Exception:
                    continue

    # INCREMENTAL MODE — per-symbol repair (kept for compatibility)
    else:
        def _process(sym_u: str) -> int:
            with _ROLLING_LOCK:
                node = _ensure_symbol_node(rolling, sym_u)
                node_local = dict(node)

            hist = node_local.get("history") or []
            if not isinstance(hist, list):
                hist = []
            hist = _ensure_bootstrap_history_if_needed(sym_u, hist, min_days=min_days, hf_prefetched=hf_prefetched)

            if hist and str(hist[-1].get("date")) == today:
                node_local["history"] = hist
                with _ROLLING_LOCK:
                    rolling[sym_u] = node_local
                return 0

            sa = _fetch_from_stockanalysis(sym_u)
            if not sa:
                if hist:
                    node_local["history"] = hist
                    try:
                        last_bar = hist[-1]
                        node_local["close"] = last_bar.get("close")
                        node_local.setdefault("price", last_bar.get("close"))
                    except Exception:
                        pass
                    with _ROLLING_LOCK:
                        rolling[sym_u] = node_local
                    return 1
                with _ROLLING_LOCK:
                    rolling[sym_u] = node_local
                return 0

            latest_bar = {
                "date": today,
                "open": sa.get("open"),
                "high": sa.get("high"),
                "low": sa.get("low"),
                "close": sa.get("price") or sa.get("close"),
                "volume": sa.get("volume"),
            }

            by_date: Dict[str, Dict[str, Any]] = {}
            for bar in hist or []:
                d = str(bar.get("date") or "")
                if d:
                    by_date[d] = bar
            by_date[today] = latest_bar
            hist_new = list(by_date.values())
            hist_new.sort(key=lambda x: x.get("date") or "")

            node_local["history"] = hist_new
            node_local["close"] = latest_bar.get("close") or sa.get("close")
            node_local["marketCap"] = sa.get("marketCap", node_local.get("marketCap"))
            try:
                node_local.update(sa)
            except Exception:
                pass

            with _ROLLING_LOCK:
                rolling[sym_u] = node_local
            return 1

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_process, s): s for s in symbols}
            for fut in progress_bar(as_completed(futs), desc="Backfill (incremental)", unit="sym", total=total):
                try:
                    updated += int(fut.result() or 0)
                except Exception:
                    continue

    # Save to rolling_body (data_pipeline points save_rolling to PATHS["rolling_body"])
    save_rolling(rolling)

    dur = time.time() - start
    log(f"✅ Backfill ({mode}) complete — {updated}/{total} updated in {dur:.1f}s.")
    log("ℹ️ Rolling safety: this backfill never prunes or deletes existing rolling history.")
    if USE_HF_BOOTSTRAP:
        log(f"ℹ️ HF bootstrap enabled via DuckDB/Parquet: {HF_STOCK_PRICES_PARQUET}")
        log(f"ℹ️ HF prefetch: enabled={HF_PREFETCH_ENABLED}, batch_size={HF_BATCH_SIZE}, max_concurrency={max(1, HF_MAX_CONCURRENCY)}")
        if _HF_DUCKDB_ERR is not None:
            log(f"ℹ️ HF DuckDB note: last import/init error seen: {_HF_DUCKDB_ERR}")
    if YF_PATCH_RECENT_DAYS > 0:
        log(f"ℹ️ YF recent patch days={YF_PATCH_RECENT_DAYS}, max_symbols={YF_PATCH_MAX_SYMBOLS} (see env vars).")
    return updated


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AION Rolling Backfill (Batch SA + Batched HF Bootstrap + YF Fallback)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min_days", type=int, default=180)
    args = parser.parse_args()

    symbols = load_universe()

    if not symbols:
        log("⚠️ Universe empty — cannot backfill.")
    else:
        backfill_symbols(symbols, min_days=args.min_days, max_workers=args.workers)
