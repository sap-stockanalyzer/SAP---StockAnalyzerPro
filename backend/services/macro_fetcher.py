# backend/services/macro_fetcher.py — v1.4 (FRED-only + throttle + retries + better logging + robust obs parsing)
"""
macro_fetcher.py — v1.4 (FRED-only; no yfinance; rate-limit resilient)

Upgrades (v1.4):
  ✅ Canonicalize SP500 naming:
       - NEW: sp500_close / sp500_daily_pct / sp500_pct_decimal
       - KEEP: spy_close / spy_daily_pct / spy_pct_decimal as *legacy aliases* (surrogate = SP500)
         so you don’t break downstream readers immediately.
  ✅ Log HTTP status + body snippet when we get empty_or_bad_json (no more blindfold)
  ✅ Robust observation parsing:
       - scan up to a configurable window (default 60 days) for the last 2 valid numeric values
       - skip "." and blanks (FRED missing values)
  ✅ Throttle: skip fetch if last macro_state.json is "fresh" (default 6h)
  ✅ Retry w/ exponential backoff on HTTP errors / rate limits (default 3 tries)
  ✅ If fetch returns junk (zeros/empty), DO NOT overwrite existing snapshots
  ✅ Atomic writes (tmp -> replace)
  ✅ Writes canonical macro_state.json AND market_state.json snapshot (fallback)
  ✅ Adds alias keys used downstream: vix, spy_pct, breadth (regime_detector-friendly)

Notes:
  • We do NOT fetch the SPY ETF here. We fetch FRED SP500 (index level).
    For clarity, SP500 is stored canonically as sp500_*.
    Legacy spy_* fields are kept as aliases to avoid breaking older code.
  • "tnx_close" uses DGS10 (10Y yield, percent), not ^TNX directly.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import log, safe_float


# ------------------------------------------------------------
# Config knobs (env)
# ------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        v = str(os.getenv(name, "")).strip()
        if not v:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        v = str(os.getenv(name, "")).strip()
        if not v:
            return int(default)
        return int(float(v))
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.environ.get(name, "") or "").strip().lower()
    if v == "":
        return default
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


# Default: don’t refetch macro more than once every 6 hours
MACRO_MIN_REFRESH_HOURS = _env_float("AION_MACRO_MIN_REFRESH_HOURS", 6.0)
MACRO_MAX_RETRIES = _env_int("AION_MACRO_MAX_RETRIES", 3)
MACRO_RETRY_BASE_SEC = _env_float("AION_MACRO_RETRY_BASE_SEC", 10.0)
MACRO_FORCE = _env_bool("AION_MACRO_FORCE", False)

# How far back we’ll search for numeric values if the latest entries are "."
FRED_WINDOW_DAYS = _env_int("AION_FRED_WINDOW_DAYS", 60)

# FRED API sometimes needs more than 10 obs to find 2 valid numerics (gold does this a lot)
FRED_LIMIT = _env_int("AION_FRED_OBS_LIMIT", 250)

# Logging: how much of the HTTP body to print on failures
FRED_ERR_SNIPPET_CHARS = _env_int("AION_FRED_ERR_SNIPPET_CHARS", 240)

FRED_API_KEY = (os.getenv("FRED_API", "") or "").strip()


# ------------------------------------------------------------
# Atomic write helper
# ------------------------------------------------------------

def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def _is_fresh(path: Path, max_age_hours: float) -> bool:
    """
    Controls "skip if it's < N hours old":
      return age_s < max_age_hours * 3600
    """
    try:
        if not path.exists():
            return False
        age_s = max(0.0, datetime.now(timezone.utc).timestamp() - path.stat().st_mtime)
        return age_s < float(max_age_hours) * 3600.0
    except Exception:
        return False


# ------------------------------------------------------------
# FRED fetch helpers
# ------------------------------------------------------------

def _http_get_text(url: str, timeout_s: float = 20.0) -> Tuple[Optional[int], str]:
    """
    Stdlib-only HTTP GET.

    Returns: (status_code, body_text)
      - status_code may be None if we never got a response.
    """
    try:
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "AION-Analytics/1.0 (macro_fetcher)",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            status = getattr(resp, "status", None)
            body = resp.read().decode("utf-8", errors="ignore")
            return int(status) if status is not None else 200, body

    except Exception as e:
        # urllib gives us HTTPError with status + body sometimes; extract if possible
        try:
            import urllib.error  # type: ignore
            if isinstance(e, urllib.error.HTTPError):
                status = int(getattr(e, "code", 0) or 0)
                try:
                    body = e.read().decode("utf-8", errors="ignore")
                except Exception:
                    body = str(e)
                return status, body
        except Exception:
            pass

        return None, str(e)


def _http_get_json(url: str, timeout_s: float = 20.0) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (parsed_json_or_none, meta)
      meta includes: status, snippet, err
    """
    status, body = _http_get_text(url, timeout_s=timeout_s)
    snippet = (body or "")[: max(0, int(FRED_ERR_SNIPPET_CHARS))]

    try:
        js = json.loads(body) if isinstance(body, str) and body else None
        if isinstance(js, dict):
            return js, {"status": status, "snippet": snippet, "err": ""}
        return None, {"status": status, "snippet": snippet, "err": "non_dict_json_or_empty"}
    except Exception as e:
        return None, {"status": status, "snippet": snippet, "err": f"json_parse_error: {e}"}


def _fred_series_observations(series_id: str, limit: int = 50, window_days: int = 60) -> List[Dict[str, Any]]:
    """
    Returns observations list (most recent first).

    Key behaviors:
      - sort_order=desc so newest is first
      - observation_start set so we don’t drag the whole history
      - logs HTTP status + snippet on bad JSON/schema (debuggable!)
    """
    if not FRED_API_KEY:
        return []

    base = "https://api.stlouisfed.org/fred/series/observations"

    # Observation start: last N days (inclusive).
    # Use UTC date; FRED expects YYYY-MM-DD.
    try:
        start_dt: date = (datetime.now(timezone.utc) - timedelta(days=max(1, int(window_days)))).date()
        observation_start = start_dt.isoformat()
    except Exception:
        observation_start = ""

    params = (
        f"?series_id={series_id}"
        f"&api_key={FRED_API_KEY}"
        f"&file_type=json"
        f"&sort_order=desc"
        f"&limit={max(1, int(limit))}"
    )
    if observation_start:
        params += f"&observation_start={observation_start}"

    url = base + params

    last_meta: Dict[str, Any] = {}
    last_err: Optional[str] = None

    for attempt in range(1, max(1, MACRO_MAX_RETRIES) + 1):
        js, meta = _http_get_json(url, timeout_s=20.0)
        last_meta = meta or {}
        try:
            if not js or "observations" not in js:
                last_err = "empty_or_bad_json"
                raise RuntimeError(last_err)

            obs = js.get("observations") or []
            if not isinstance(obs, list):
                last_err = "bad_observations_schema"
                raise RuntimeError(last_err)

            return [o for o in obs if isinstance(o, dict)]

        except Exception as e:
            last_err = str(e)

        if attempt < MACRO_MAX_RETRIES:
            base_s = float(MACRO_RETRY_BASE_SEC) * (2 ** (attempt - 1))
            jitter = random.uniform(0.0, 0.25 * base_s)
            sleep_s = min(180.0, base_s + jitter)

            st = last_meta.get("status")
            snip = (last_meta.get("snippet") or "").replace("\n", " ")[: max(0, int(FRED_ERR_SNIPPET_CHARS))]
            err = last_meta.get("err", "")

            log(
                f"[macro_fetcher] ⚠️ FRED fetch {series_id} attempt {attempt}/{MACRO_MAX_RETRIES} failed: {last_err} "
                f"(status={st} meta_err={err} snippet='{snip}') — sleeping {sleep_s:.1f}s"
            )
            try:
                time.sleep(sleep_s)
            except Exception:
                pass

    st = last_meta.get("status")
    snip = (last_meta.get("snippet") or "").replace("\n", " ")[: max(0, int(FRED_ERR_SNIPPET_CHARS))]
    err = last_meta.get("err", "")
    log(f"[macro_fetcher] ⚠️ FRED fetch {series_id} failed after retries: {last_err} (status={st} meta_err={err} snippet='{snip}')")
    return []


def _parse_fred_value(v: Any) -> Optional[float]:
    """
    FRED uses "." for missing.
    """
    try:
        s = str(v).strip()
        if not s or s == ".":
            return None
        x = float(s)
        if x != x:
            return None
        return x
    except Exception:
        return None


def _last2_from_fred(series_id: str) -> Tuple[float, float]:
    """
    Returns (last_value, pct_change_vs_prev) where pct is percent.

    Robust behavior:
      - pulls observations over a window (default 60 days)
      - scans until it finds 2 numeric values
      - skips "." / blanks
    """
    obs = _fred_series_observations(series_id, limit=int(FRED_LIMIT), window_days=int(FRED_WINDOW_DAYS))

    vals: List[float] = []
    for o in obs:
        x = _parse_fred_value(o.get("value"))
        if x is None:
            continue
        vals.append(float(x))
        if len(vals) >= 2:
            break

    if len(vals) < 1:
        return 0.0, 0.0
    if len(vals) < 2:
        return float(vals[0]), 0.0

    last = float(vals[0])
    prev = float(vals[1])
    if prev == 0:
        return last, 0.0
    pct = ((last - prev) / prev) * 100.0
    return last, float(pct)


# ------------------------------------------------------------
# Sanity gate
# ------------------------------------------------------------

def _macro_looks_sane(m: Dict[str, Any]) -> bool:
    """
    Consider macro sane if we have at least some real signal.
    Most important: SP500 should not be zero.
    Prefer VIX too, but don't brick everything if only VIX fails.
    """
    try:
        spx = abs(safe_float(m.get("sp500_close", 0.0)))
        if spx <= 0.0:
            return False

        vix = abs(safe_float(m.get("vix_close", 0.0)))
        spx_dec = abs(safe_float(m.get("sp500_pct_decimal", 0.0)))
        breadth = abs(safe_float(m.get("breadth_proxy", 0.0)))

        if vix >= 8.0:
            return True

        return (spx_dec > 0.0001) or (breadth > 0.0001)
    except Exception:
        return False


# ------------------------------------------------------------
# Risk-off (kept stable)
# ------------------------------------------------------------

def _risk_off_score(vix_close: float, spy_pct_dec: float, dxy_pct_dec: float) -> float:
    vix_component = min(max((vix_close - 15.0) / 25.0, 0.0), 1.0)      # 15..40 mapped
    spy_component = min(max((-spy_pct_dec) / 0.03, 0.0), 1.0)          # -3% maps to 1
    dxy_component = min(max((dxy_pct_dec) / 0.01, 0.0), 1.0)           # +1% maps to 1
    return float(min(1.0, 0.45 * vix_component + 0.45 * spy_component + 0.10 * dxy_component))


# ------------------------------------------------------------
# Main builder
# ------------------------------------------------------------

def build_macro_features() -> Dict[str, Any]:
    log("🌐 Fetching macro signals via FRED (VIX, SP500, NASDAQ, 10Y, DXY, Gold, Oil)…")

    # canonical destination
    out_path = PATHS.get("macro_state")
    if not isinstance(out_path, Path):
        out_path = None

    # If no key, best-effort return cached
    if not FRED_API_KEY:
        log("[macro_fetcher] ❌ FRED_API key missing. Set FRED_API in .env.")
        if out_path:
            cached = _read_json_if_exists(out_path)
            if cached:
                return {"status": "skipped", "reason": "missing_fred_key_return_cache", "macro_state": cached}
        return {"status": "error", "error": "missing_fred_api_key"}

    # Throttle: if we have a recent macro_state.json, skip fetch unless forced
    if (not MACRO_FORCE) and out_path and _is_fresh(out_path, MACRO_MIN_REFRESH_HOURS):
        cached = _read_json_if_exists(out_path)
        if cached:
            log(f"[macro_fetcher] ℹ️ macro_state.json is fresh (<{MACRO_MIN_REFRESH_HOURS}h). Skipping fetch.")
            return {"status": "skipped", "reason": "fresh_cache", "macro_state": cached}

    # --- FRED series map ---
    SERIES = {
        "vix": "VIXCLS",
        "sp500": "SP500",
        "nasdaq": "NASDAQCOM",
        "tnx": "DGS10",
        "dxy": "DTWEXBGS",
        "gld": "GOLDAMGBD228NLBM",
        "uso": "DCOILWTICO",
    }

    vix_close, vix_pct = _last2_from_fred(SERIES["vix"])
    spx_close, spx_pct = _last2_from_fred(SERIES["sp500"])
    nas_close, nas_pct = _last2_from_fred(SERIES["nasdaq"])
    tnx_close, tnx_pct = _last2_from_fred(SERIES["tnx"])
    dxy_close, dxy_pct = _last2_from_fred(SERIES["dxy"])
    gld_close, gld_pct = _last2_from_fred(SERIES["gld"])
    uso_close, uso_pct = _last2_from_fred(SERIES["uso"])

    sp500_close = float(safe_float(spx_close))
    sp500_daily_pct = float(safe_float(spx_pct))
    sp500_pct_dec = float(sp500_daily_pct / 100.0)

    dxy_pct_dec = float(safe_float(dxy_pct) / 100.0)

    # Breadth proxy: keep your existing proxy style (SP500 daily change in decimal)
    breadth_proxy = float(sp500_pct_dec)

    volatility = float(max(0.0, min(0.10, float(safe_float(vix_close)) / 100.0)))
    risk_off = _risk_off_score(float(safe_float(vix_close)), float(sp500_pct_dec), float(dxy_pct_dec))

    now_iso_utc = datetime.now(timezone.utc).isoformat()
    now_iso_local = datetime.now(TIMEZONE).isoformat()

    macro_state: Dict[str, Any] = {
        # VIX
        "vix_close": float(safe_float(vix_close)),
        "vix_daily_pct": float(safe_float(vix_pct)),

        # ✅ Canonical SP500 fields (clarity)
        "sp500_close": float(safe_float(sp500_close)),
        "sp500_daily_pct": float(safe_float(sp500_daily_pct)),       # percent
        "sp500_pct_decimal": float(safe_float(sp500_pct_dec)),       # decimal

        # NASDAQ surrogate for QQQ-ish tech risk (optional)
        "qqq_close": float(safe_float(nas_close)),
        "qqq_daily_pct": float(safe_float(nas_pct)),

        # TNX surrogate: 10Y yield (percent)
        "tnx_close": float(safe_float(tnx_close)),
        "tnx_daily_pct": float(safe_float(tnx_pct)),

        # DXY surrogate: trade-weighted USD index
        "dxy_close": float(safe_float(dxy_close)),
        "dxy_daily_pct": float(safe_float(dxy_pct)),
        "dxy_pct_decimal": float(safe_float(dxy_pct_dec)),

        # Gold & Oil (best-effort; may be 0 if series missing/blank)
        "gld_close": float(safe_float(gld_close)),
        "gld_daily_pct": float(safe_float(gld_pct)),
        "uso_close": float(safe_float(uso_close)),
        "uso_daily_pct": float(safe_float(uso_pct)),

        # Breadth proxy
        "breadth_proxy": float(safe_float(breadth_proxy)),

        # Downstream keys
        "volatility": float(safe_float(volatility)),
        "risk_off": float(safe_float(risk_off)),

        # timestamps
        "generated_at": now_iso_local,
        "updated_at": now_iso_utc,

        # provenance
        "source": "fred",
        "fred_series": dict(SERIES),

        # clarity breadcrumbs
        "notes": {
            "sp500_is_index_level_not_spy_etf": True,
            "legacy_spy_fields_are_sp500_aliases": True,
        },
    }

    # ----------------------------
    # Legacy compatibility (do NOT delete yet)
    # ----------------------------
    # Older code expects spy_close/spy_pct_decimal, but we are not fetching SPY ETF.
    # We keep these as aliases to SP500 to avoid breaking downstream,
    # while sp500_* is the canonical truth going forward.
    macro_state["spy_close"] = float(macro_state.get("sp500_close", 0.0))
    macro_state["spy_daily_pct"] = float(macro_state.get("sp500_daily_pct", 0.0))
    macro_state["spy_pct_decimal"] = float(macro_state.get("sp500_pct_decimal", 0.0))

    # Alias keys used by regime_detector / context_state debug expectations
    macro_state["vix"] = float(macro_state.get("vix_close", 0.0))
    macro_state["spy_pct"] = float(macro_state.get("spy_pct_decimal", 0.0))   # still what your regime expects
    macro_state["breadth"] = float(macro_state.get("breadth_proxy", 0.0))

    # ----------------------------
    # Critical: Do NOT overwrite on failure
    # ----------------------------
    if not _macro_looks_sane(macro_state):
        log("[macro_fetcher] ⚠️ Macro fetch looks invalid (zeros/empty). Keeping last snapshot (no overwrite).")
        if out_path:
            cached = _read_json_if_exists(out_path)
            if cached:
                return {"status": "skipped", "reason": "macro_not_sane_keep_last", "macro_state": cached}
        return {"status": "skipped", "reason": "macro_not_sane_no_cache", "macro_state": macro_state}

    # Save canonical macro_state.json
    if out_path:
        try:
            _atomic_write_json(out_path, macro_state)
            log(f"[macro_fetcher] 📈 macro_state.json updated → {out_path}")
        except Exception as e:
            log(f"[macro_fetcher] ⚠️ Failed to write macro_state.json: {e}")

    # Also write market_state-compatible snapshot for regime_detector fallbacks
    try:
        ml_root = PATHS.get("ml_data", Path("ml_data"))
        market_state_path = Path(ml_root) / "market_state.json"
        payload = dict(macro_state)
        payload.setdefault("regime_hint", "neutral")
        _atomic_write_json(market_state_path, payload)
        log(f"[macro_fetcher] 🧭 market_state.json (macro snapshot) updated → {market_state_path}")
    except Exception as e:
        log(f"[macro_fetcher] ⚠️ Failed to write market_state.json macro snapshot: {e}")

    return {"status": "ok", **macro_state}


if __name__ == "__main__":
    res = build_macro_features()
    print(json.dumps(res, indent=2))
