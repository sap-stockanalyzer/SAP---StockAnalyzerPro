# backend/services/backfill_history.py
"""
backfill_history.py — v3.3 (+ universe auto-prune on YF miss)
(Rolling-Native, Normalized Batch StockAnalysis Bundle, YF History Bootstrap)

Purpose:
- Refreshes and repairs ticker data directly inside Rolling cache.
- BATCH fetches metrics from StockAnalysis (parallel /s/d/<metric> requests)
  using modern endpoints.
- Uses /s/i only for basic metadata (symbol, name, price, volume, marketCap, peRatio, industry).
- Uses /s/d/<metric> for everything else (incl. open/high/low/close, rsi, growth, etc.).
- Normalizes all fetched field names before saving (camelCase → snake_case, rsi → rsi_14).
- Writes directly into rolling.json.gz using the new backend.core.data_pipeline helpers.

NEW in v3.2:
- Automatic 3-year (750 trading days) YFinance bootstrap when history is too short (< min_days).
- History is strictly append-only with per-date dedupe (never wipes existing history).
- Uses utils.progress_bar.progress_bar for nicer progress output.
- FutureWarnings from yfinance/pandas removed (no float(single-element Series), auto_adjust set explicitly).

NEW in v3.3:
- Removed dependency on deprecated ensure_symbol_fields from data_pipeline.
- Local helper _ensure_symbol_node keeps basic symbol/history structure;
  full normalization (predictions/context/news/social/policy) is delegated
  to backend.core.data_pipeline.save_rolling().
- Skips meta keys starting with '_' when deriving symbol list from Rolling.

ADD-ON:
- If YFinance bootstrap returns ZERO bars for a symbol that needs bootstrap,
  auto-prune that symbol from the swing universe JSON after the run.
  (DT is unaffected; DT uses Alpaca.)

"""

from __future__ import annotations

import os
import json
import gzip
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Iterable
from typing import Optional, Set
from pathlib import Path
from threading import Lock

import requests
import yfinance as yf

from backend.core.config import PATHS
from backend.core.data_pipeline import (
    _read_rolling,
    save_rolling,
    log,
)
from backend.services.metrics_fetcher import build_latest_metrics
from utils.progress_bar import progress_bar

UNIVERSE_FILE = PATHS["universe"] / "master_universe.json"

# -------------------------------------------------------------------
# Verbosity controls (tune as you like)
# -------------------------------------------------------------------
VERBOSE_BOOTSTRAP = False          # per-ticker “Bootstrapped history for XYZ…”
VERBOSE_BOOTSTRAP_ERRORS = False   # per-ticker YF failure messages


# -------------------------------------------------------------------
# YFinance safety knobs (rate limit mitigation)
# -------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "") or default).strip())
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, "") or default).strip())
    except Exception:
        return default

def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.environ.get(name, "") or "").strip().lower()
    if v == "":
        return default
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default

# Enable YF bootstrap (kept ON by default for backward compatibility).
# Set AION_YF_BOOTSTRAP=0 to completely disable YFinance usage.
YF_BOOTSTRAP_ENABLED = _env_bool("AION_YF_BOOTSTRAP", default=True)

# Hard cap concurrent YF calls (critical: backfill itself is multi-threaded)
YF_MAX_CONCURRENCY = max(1, _env_int("AION_YF_MAX_CONCURRENCY", 1))
YF_MIN_SPACING_SECONDS = max(0.0, _env_float("AION_YF_MIN_SPACING_SECONDS", 0.15))
YF_MAX_RETRIES = max(0, _env_int("AION_YF_MAX_RETRIES", 5))
YF_BACKOFF_SECONDS = max(0.25, _env_float("AION_YF_BACKOFF_SECONDS", 2.0))

_YF_SEM = None  # initialized lazily to avoid import-order surprises
_YF_LAST_CALL_TS = 0.0
_YF_TS_LOCK = Lock()

def _yf_semaphore() -> "Lock":
    # Lazily create semaphore-like lock pool.
    # Using threading.Semaphore but typed loosely to keep py3.10/3.12 happy.
    global _YF_SEM
    if _YF_SEM is None:
        try:
            import threading
            _YF_SEM = threading.Semaphore(int(YF_MAX_CONCURRENCY))
        except Exception:
            _YF_SEM = Lock()
    return _YF_SEM

# -------------------------------------------------------------------
# NEW: Track symbols that fail YFinance bootstrap (thread-safe)
# -------------------------------------------------------------------
_YF_NO_DATA: set[str] = set()
_YF_NO_DATA_LOCK = Lock()


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
# StockAnalysis endpoints
# -------------------------------------------------------------------
SA_BASE = "https://stockanalysis.com/api/screener"

# Index "base" fields from /s/i
SA_INDEX_FIELDS = [
    "symbol", "name", "price", "change", "volume",
    "marketCap", "peRatio", "industry",
]

# Metrics from /s/d/<metric> (aligned with SA docs)
# NOTE:
#   - rsi normalized → rsi_14
#   - sharesOut used for shares outstanding
#   - open/high/low/close fetched from /s/d/* for fuller bars
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

# How many days of history to keep in rolling
# (Option A — about 3 trading years)
MAX_HISTORY_DAYS = 750

# Directory for audit bundle
METRICS_BUNDLE_DIR = Path("data") / "metrics_cache" / "bundle"
METRICS_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# HTTP helpers
# -------------------------------------------------------------------

def _sa_post_json(path: str, payload: dict | None = None, timeout: int = 20) -> dict | None:
    """Generic helper for StockAnalysis API POST/GET requests."""
    url = f"{SA_BASE}/{path.strip('/')}"
    try:
        if payload is not None:
            r = requests.post(url, json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        # Fallback GET
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log(f"⚠️ SA request failed for {url}: {e}")
    return None


# Simple symbol-level fetch (used only in *rare* incremental mode)
_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}


def _fetch_from_stockanalysis(sym: str) -> Dict[str, Any] | None:
    """
    Lightweight helper to fetch a single symbol snapshot.
    For now, we reuse /s/i batch and cache once per run, then read from it.
    This is only used in the incremental branch, which is rarely hit.
    """
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


# -------------------------------------------------------------------
# Batch SA bundle builders
# -------------------------------------------------------------------

def _fetch_sa_index_batch() -> Dict[str, Dict[str, Any]]:
    """Fetch base index snapshot from /s/i (up to 10k rows)."""
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
    """Fetch a single metric table from /s/d/<metric>."""
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
    """Fetch multiple /s/d/<metric> endpoints in parallel."""
    result: Dict[str, Dict[str, Any]] = {}
    metrics = list(metrics)

    def _job(m):
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
    """Convert camelCase → snake_case and ensure RSI normalized to rsi_14."""
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
    """Merge /s/i index snapshot and /s/d metric tables into a per-symbol bundle."""
    out = dict(index_map)
    for metric, tbl in (metrics_map or {}).items():
        for sym, val in (tbl or {}).items():
            if sym not in out:
                out[sym] = {"symbol": sym}
            out[sym][metric] = val
    return out


def _normalize_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize all field names in the bundle once at the end."""
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
    """Save full bundle snapshot for audit."""
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
    """Fetch index + all metrics, normalize, and return unified bundle."""
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
# YF history bootstrap helpers
# -------------------------------------------------------------------


def _bootstrap_history_yf(symbol: str, max_days: int = MAX_HISTORY_DAYS) -> List[Dict[str, Any]]:
    """
    Fetch up to ~3 years of daily bars from YFinance for a symbol.
    Returns list of {date, open, high, low, close, volume}.

    IMPORTANT:
      - This call is rate-limit prone and must be concurrency-capped.
      - We use a global semaphore + minimum spacing + exponential backoff.
      - If YF is disabled (AION_YF_BOOTSTRAP=0), returns [] immediately.
    """
    if not YF_BOOTSTRAP_ENABLED:
        return []

    symbol = symbol.upper()

    # Optional: yfinance exposes a dedicated error type in newer versions.
    try:
        from yfinance.exceptions import YFRateLimitError  # type: ignore
    except Exception:
        YFRateLimitError = Exception  # type: ignore

    sem = _yf_semaphore()

    def _respect_spacing():
        global _YF_LAST_CALL_TS
        try:
            with _YF_TS_LOCK:
                now = time.time()
                wait = float(YF_MIN_SPACING_SECONDS) - (now - float(_YF_LAST_CALL_TS))
                if wait > 0:
                    time.sleep(wait)
                _YF_LAST_CALL_TS = time.time()
        except Exception:
            pass

    # Retry loop
    tries = 0
    backoff = float(YF_BACKOFF_SECONDS)

    while True:
        tries += 1
        try:
            # Concurrency cap
            try:
                sem.acquire()
            except Exception:
                pass

            _respect_spacing()

            df = yf.download(
                tickers=symbol,
                interval="1d",
                period="3y",         # ~3 calendar years
                auto_adjust=False,   # explicit → no FutureWarning
                progress=False,
                threads=False,       # yfinance internal threading can amplify rate limits
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

            if not bars:
                return []

            return bars[-max_days:]

        except KeyboardInterrupt:
            # If you really did Ctrl+C, honor it.
            raise
        except (YFRateLimitError,) as e:
            # Hard rate limit — backoff and retry.
            if tries <= int(YF_MAX_RETRIES):
                if VERBOSE_BOOTSTRAP_ERRORS:
                    log(f"⚠️ YF rate limited for {symbol} (try {tries}/{YF_MAX_RETRIES}): {e}")
                time.sleep(backoff)
                backoff = min(backoff * 1.8, 60.0)
                continue
            if VERBOSE_BOOTSTRAP_ERRORS:
                log(f"⚠️ YF rate limit giving up for {symbol}: {e}")
            return []
        except Exception as e:
            # Many yfinance versions throw generic Exceptions with "Too Many Requests"
            msg = str(e)
            is_rl = ("Too Many Requests" in msg) or ("rate limit" in msg.lower())
            if is_rl and tries <= int(YF_MAX_RETRIES):
                if VERBOSE_BOOTSTRAP_ERRORS:
                    log(f"⚠️ YF rate limited for {symbol} (try {tries}/{YF_MAX_RETRIES}): {msg}")
                time.sleep(backoff)
                backoff = min(backoff * 1.8, 60.0)
                continue

            if VERBOSE_BOOTSTRAP_ERRORS:
                log(f"⚠️ YF bootstrap failed for {symbol}: {e}")
            return []
        finally:
            try:
                sem.release()
            except Exception:
                pass


def _unique_history_dates(hist: List[Dict[str, Any]]) -> set[str]:
    return {str(b.get("date")) for b in (hist or []) if b.get("date")}


def _ensure_bootstrap_history_if_needed(
    symbol: str,
    hist: List[Dict[str, Any]],
    min_days: int,
) -> List[Dict[str, Any]]:
    """
    If history is too short (< min_days unique dates), bootstrap from YF.

    - Never overwrites existing per-date bars.
    - YF bars are only used for dates missing in existing history.
    - Result is sorted by date and capped at MAX_HISTORY_DAYS.

    ADD-ON:
    - If bootstrap is needed and YF returns zero bars, record this symbol
      for universe pruning after the run.
    """
    symbol = symbol.upper()
    existing_dates = _unique_history_dates(hist)

    if len(existing_dates) >= min_days:
        return hist

    # Need bootstrap
    yf_bars = _bootstrap_history_yf(symbol, max_days=MAX_HISTORY_DAYS)
    if not yf_bars:
        # NEW: record YF miss for later universe prune
        try:
            with _YF_NO_DATA_LOCK:
                _YF_NO_DATA.add(symbol)
        except Exception:
            pass
        return hist

    by_date: Dict[str, Dict[str, Any]] = {}

    # Start with YF bars
    for bar in yf_bars:
        d = str(bar.get("date"))
        if not d:
            continue
        by_date[d] = bar

    # Overlay existing history WITHOUT overwriting YF (YF is used as fallback)
    for bar in hist or []:
        d = str(bar.get("date"))
        if not d:
            continue
        # Prefer existing bar if present
        if d not in by_date:
            by_date[d] = bar

    merged = list(by_date.values())
    merged.sort(key=lambda x: x.get("date") or "")
    merged = merged[-MAX_HISTORY_DAYS:]

    if VERBOSE_BOOTSTRAP:
        log(f"🧪 Bootstrapped history for {symbol}: {len(merged)} days.")

    return merged


# -------------------------------------------------------------------
# Universe pruning helpers (NEW)
# -------------------------------------------------------------------

def _prune_universe_file(path: Path, bad_syms: set[str]) -> int:
    """
    Remove symbols from a universe JSON file.

    Supports:
      - {"symbols": [...]} dict format
      - [...] list format

    Writes a timestamped backup next to the file before overwriting.
    Returns number removed.
    """
    if not path.exists():
        return 0

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"⚠️ Failed to read universe file for pruning {path}: {e}")
        return 0

    wrapper = None
    syms: list[str] = []

    if isinstance(raw, dict) and isinstance(raw.get("symbols"), list):
        wrapper = "dict"
        syms = [str(x) for x in (raw.get("symbols") or [])]
    elif isinstance(raw, list):
        wrapper = "list"
        syms = [str(x) for x in raw]
    else:
        return 0

    bad_u = {str(s).upper() for s in (bad_syms or set())}
    before_u = {str(s).upper() for s in syms}
    keep_u = sorted(before_u - bad_u)

    removed = int(len(before_u) - len(keep_u))
    if removed <= 0:
        return 0

    # backup original
    try:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(path.name + f".bak_{ts}")
        backup_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    except Exception:
        pass

    # write pruned
    try:
        if wrapper == "dict":
            raw["symbols"] = keep_u
            path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        else:
            path.write_text(json.dumps(keep_u, indent=2), encoding="utf-8")
    except Exception as e:
        log(f"⚠️ Failed to write pruned universe file {path}: {e}")
        return 0

    return removed


# -------------------------------------------------------------------
# Local node helper (replaces ensure_symbol_fields)
# -------------------------------------------------------------------

def _ensure_symbol_node(rolling: Dict[str, Any], sym_u: str) -> Dict[str, Any]:
    """
    Minimal per-symbol scaffolding:
      - guarantees 'symbol' and 'history' keys exist.
    All other normalization (predictions/context/news/social/policy, sector
    normalization, etc.) is handled centrally in backend.core.data_pipeline.save_rolling().
    """
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
    Perform full or incremental Rolling backfill.

    Called from:
        backend/jobs/nightly_job.py
        or CLI: python -m backend.services.backfill_history --min_days 180

    Mode:
        - If rolling is empty → 'fallback' full rebuild using SA bundle (+ YF bootstrap)
        - Otherwise → 'full' bundle-based refresh (plus per-symbol YF bootstrap if history < min_days)
        - Incremental branch kept for compatibility, but rarely used.
    """
    rolling = _read_rolling() or {}
    today = datetime.utcnow().strftime("%Y-%m-%d")
    mode = "full"
    if not rolling:
        mode = "fallback"
        log("⚠️ Rolling cache missing — forcing full rebuild.")
    log(f"🧩 Backfill mode: {mode.upper()} | Date: {today}")

    # NEW: track how many were pruned (for end-of-run log)
    pruned_total = 0

    # If caller didn't specify symbols, derive from existing rolling keys (skip meta)
    if not symbols:
        symbols = [s for s in rolling.keys() if not s.startswith("_")]

    total = len(symbols)
    if not total:
        log("⚠️ No symbols to backfill.")
        return 0

    updated = 0
    start = time.time()

    # ----------------------------------------------------------
    # FULL / FALLBACK MODE — bundle-based refresh + bootstrap
    # ----------------------------------------------------------
    if mode in ("full", "fallback"):
        log(f"🔧 Starting full rolling backfill for {total} symbols (batch SA fetch + YF bootstrap)…")
        sa_bundle = fetch_sa_bundle_parallel(max_workers=max_workers)
        if sa_bundle:
            # Optionally rebuild metrics cache for other services.
            try:
                build_latest_metrics()
            except Exception as e:
                log(f"⚠️ build_latest_metrics during backfill failed: {e}")
        else:
            log("⚠️ Empty SA bundle.")

        def _process(sym: str) -> int:
            sym_u = sym.upper()
            # Ensure node structure exists
            node = _ensure_symbol_node(rolling, sym_u)

            # Ensure multi-day history via YF if too short
            hist = node.get("history") or []
            hist = _ensure_bootstrap_history_if_needed(sym_u, hist, min_days=min_days)

            sa = sa_bundle.get(sym_u) if sa_bundle else None
            if not sa:
                # Still save bootstrapped history if we have it
                if hist:
                    node["history"] = hist
                    # Derive close/price from latest history bar as a robust fallback
                    try:
                        last_bar = hist[-1]
                        node["close"] = last_bar.get("close")
                        node.setdefault("price", last_bar.get("close"))
                    except Exception:
                        pass
                    rolling[sym_u] = node
                    return 1
                return 0

            latest_bar = {
                "date": today,
                "open": sa.get("open"),
                "high": sa.get("high"),
                "low": sa.get("low"),
                "close": sa.get("price") or sa.get("close"),
                "volume": sa.get("volume"),
            }

            # Per-date dedupe (append-only semantics) ------------------
            by_date: Dict[str, Dict[str, Any]] = {}
            for bar in hist or []:
                d = str(bar.get("date"))
                if not d:
                    continue
                by_date[d] = bar
            by_date[today] = latest_bar
            hist_new = list(by_date.values())
            hist_new.sort(key=lambda x: x.get("date") or "")
            hist_new = hist_new[-MAX_HISTORY_DAYS:]
            # ---------------------------------------------------------

            node["history"] = hist_new
            node["close"] = latest_bar.get("close")
            node.update(sa)
            rolling[sym_u] = node
            return 1

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_process, s): s for s in symbols}
            for fut in progress_bar(as_completed(futs), desc="Backfill (bundle+YF)", unit="sym", total=total):
                updated += fut.result()

    # ----------------------------------------------------------
    # INCREMENTAL MODE — per-symbol repair (kept for compatibility)
    # ----------------------------------------------------------
    else:
        def _process(sym: str) -> int:
            sym_u = sym.upper()
            node = _ensure_symbol_node(rolling, sym_u)

            hist = node.get("history") or []
            # Ensure enough history first
            hist = _ensure_bootstrap_history_if_needed(sym_u, hist, min_days=min_days)

            # If already have today's bar, skip
            if hist and str(hist[-1].get("date")) == today:
                node["history"] = hist
                rolling[sym_u] = node
                return 0

            sa = _fetch_from_stockanalysis(sym_u)
            if not sa:
                # Still persist bootstrapped history if we have it
                if hist:
                    node["history"] = hist
                    try:
                        last_bar = hist[-1]
                        node["close"] = last_bar.get("close")
                        node.setdefault("price", last_bar.get("close"))
                    except Exception:
                        pass
                    rolling[sym_u] = node
                    return 1
                return 0

            latest_bar = {
                "date": today,
                "open": sa.get("open"),
                "high": sa.get("high"),
                "low": sa.get("low"),
                "close": sa.get("price") or sa.get("close"),
                "volume": sa.get("volume"),
            }

            # Per-date dedupe
            by_date: Dict[str, Dict[str, Any]] = {}
            for bar in hist or []:
                d = str(bar.get("date"))
                if not d:
                    continue
                by_date[d] = bar
            by_date[today] = latest_bar
            hist_new = list(by_date.values())
            hist_new.sort(key=lambda x: x.get("date") or "")
            hist_new = hist_new[-MAX_HISTORY_DAYS:]

            node["history"] = hist_new
            node["close"] = latest_bar.get("close") or sa.get("close")
            node["marketCap"] = sa.get("marketCap", node.get("marketCap"))
            node.update(sa)
            rolling[sym_u] = node
            return 1

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_process, s): s for s in symbols}
            for fut in progress_bar(as_completed(futs), desc="Backfill (incremental)", unit="sym", total=total):
                updated += fut.result()

    # ----------------------------------------------------------
    # Save rolling via new core helper (atomic + backups)
    # ----------------------------------------------------------
    save_rolling(rolling)

    # ----------------------------------------------------------
    # NEW: Prune universe for symbols that YFinance couldn't bootstrap
    # ----------------------------------------------------------
    try:
        with _YF_NO_DATA_LOCK:
            bad = set(_YF_NO_DATA)

        if bad:
            removed = 0
            removed += _prune_universe_file(UNIVERSE_FILE, bad)

            # Optional: if you later add swing/dt split universe files,
            # keep swing in sync automatically if it exists.
            swing_file = PATHS["universe"] / "swing_universe.json"
            if swing_file.exists():
                removed += _prune_universe_file(swing_file, bad)

            pruned_total = int(removed)

            log(
                f"🧹 Universe auto-prune: removed {removed} symbols "
                f"(YFinance bootstrap returned 0 bars)."
            )
    except Exception as e:
        log(f"⚠️ Universe prune step failed: {e}")

    dur = time.time() - start
    log(f"✅ Backfill ({mode}) complete — {updated}/{total} updated in {dur:.1f}s.")

    # NEW: end-of-run prune log (always prints, even if 0)
    log(f"🧹 Pruned from universe this run: {int(pruned_total)}")

    return updated


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AION Rolling Backfill (Batch SA + YF Bootstrap, New Core)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min_days", type=int, default=180)
    args = parser.parse_args()

    # 🚀 Load universe instead of rolling keys
    symbols = load_universe()

    if not symbols:
        log("⚠️ Universe empty — cannot backfill.")
    else:
        backfill_symbols(symbols, min_days=args.min_days, max_workers=args.workers)
