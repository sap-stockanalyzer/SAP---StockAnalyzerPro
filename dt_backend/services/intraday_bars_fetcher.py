"""dt_backend/services/intraday_bars_fetcher.py

Live intraday bars fetcher for dt_backend.

What this does
--------------
* Pulls 1Min and/or 5Min bars from Alpaca Data API in *batches*.
* Merges bars into rolling[symbol] without duplicating timestamps.
* Trims history to a configurable max length (keeps rolling lightweight).

Why it exists
-------------
dt_backend's higher-level pipeline (context/features/prediction/policy)
assumes `rolling[sym]["bars_intraday"]` is being refreshed during market
hours. The original repo had backfill + replay helpers but no robust live
updater.

This module keeps dt_backend self-contained and avoids touching backend
nightly-job code.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import os
import requests

from dt_backend.core.config_dt import (
    DT_PATHS,
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
)
from dt_backend.core.data_pipeline_dt import _read_rolling, save_rolling, ensure_symbol_node
from dt_backend.core.logger_dt import log, warn, error
from dt_backend.core.locks_dt import acquire_lock_file, release_lock_file
from dt_backend.services.dt_truth_store import BARS_FETCH_LOCK_PATH
from dt_backend.core.bars_fetch_state_dt import get_last_end, set_last_end

try:
    from pathlib import Path
except Exception:  # pragma: no cover
    Path = None  # type: ignore

DEFAULT_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"


def _headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY_ID or "",
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET_KEY or "",
    }


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_alpaca_bar(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize Alpaca bar schema into our rolling format."""
    if not isinstance(raw, dict):
        return None
    t = raw.get("t")
    if not t:
        return None
    # Alpaca typically returns RFC3339 like 2025-01-01T14:31:00Z
    ts = str(t).replace("+00:00", "Z")
    out: Dict[str, Any] = {
        "ts": ts,
        "o": raw.get("o"),
        "h": raw.get("h"),
        "l": raw.get("l"),
        "c": raw.get("c"),
        "v": raw.get("v"),
        "n": raw.get("n"),
        "vw": raw.get("vw"),
    }
    return out


def _dedupe_merge(existing: List[Dict[str, Any]], new_bars: List[Dict[str, Any]], *, max_len: int) -> List[Dict[str, Any]]:
    """Merge bars by timestamp, keeping order and trimming."""
    if not existing:
        merged = new_bars
    else:
        seen = {str(b.get("ts") or b.get("t") or "") for b in existing if isinstance(b, dict)}
        merged = list(existing)
        for b in new_bars:
            ts = str(b.get("ts") or "")
            if not ts or ts in seen:
                continue
            merged.append(b)
            seen.add(ts)

    # Sort by timestamp (string sort is OK for ISO8601 UTC)
    try:
        merged.sort(key=lambda x: str(x.get("ts") or ""))
    except Exception:
        pass

    if max_len > 0 and len(merged) > max_len:
        merged = merged[-max_len:]
    return merged


def fetch_bars_batch(
    symbols: List[str],
    *,
    timeframe: str = "1Min",
    lookback_minutes: int = 90,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    limit: int = 10000,
    bars_url: str = DEFAULT_BARS_URL,
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch bars for a batch of symbols from Alpaca.

    Returns:
        {"AAPL": [bar, ...], "MSFT": [bar, ...], ...}
    """
    syms = [s.strip().upper() for s in (symbols or []) if str(s).strip()]
    if not syms:
        return {}

    if not (ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY):
        warn("[bars_fetch] Alpaca API keys missing (ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY).")
        return {}

    end = end_dt or datetime.now(timezone.utc)
    start = start_dt or (end - timedelta(minutes=max(1, int(lookback_minutes))))

    params = {
        "symbols": ",".join(syms),
        "timeframe": timeframe,
        "start": _iso(start),
        "end": _iso(end),
        "limit": int(limit),
        "feed": "iex",  # <-- REQUIRED for non-SIP subscriptions
    }

    try:
        r = requests.get(bars_url, headers=_headers(), params=params, timeout=25)
        if r.status_code != 200:
            raise RuntimeError(f"Alpaca error {r.status_code}: {r.text[:200]}")
        payload = r.json() if r.content else {}
    except Exception as e:
        error(f"[bars_fetch] request failed (tf={timeframe}, n={len(syms)}): {e}", e)
        return {}

    bars = payload.get("bars") or {}
    if not isinstance(bars, dict):
        return {}

    out: Dict[str, List[Dict[str, Any]]] = {}
    for sym in syms:
        raw_list = bars.get(sym) or []
        if not isinstance(raw_list, list):
            continue
        norm: List[Dict[str, Any]] = []
        for raw in raw_list:
            b = _parse_alpaca_bar(raw)
            if b is not None:
                norm.append(b)
        if norm:
            out[sym] = norm
    return out


def update_rolling_with_live_bars(
    *,
    symbols: List[str],
    timeframe: str = "1Min",
    lookback_minutes: int = 90,
    max_len: int = 600,
    batch_size: int = 150,
    bars_url: str = DEFAULT_BARS_URL,
    rolling_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch + merge bars into the rolling cache.

    Parameters
    ----------
    rolling_key:
        If None, uses:
          - "bars_intraday" for 1Min
          - "bars_intraday_5m" for 5Min
        Otherwise uses the provided key.
    """

    if timeframe not in {"1Min", "5Min"}:
        warn(f"[bars_fetch] unsupported timeframe '{timeframe}', expected 1Min/5Min")
        return {"status": "bad_timeframe", "timeframe": timeframe}

    # Phase 1: prevent fetch storms (multi-process) + rate-limit friendliness.
    lock_path = DT_PATHS.get("dt_bars_fetch_lock_file")
    if not lock_path:
        lock_path = BARS_FETCH_LOCK_PATH
    lock = acquire_lock_file(lock_path, timeout_s=0.25) if lock_path else None
    if lock is None and lock_path is not None:
        return {"status": "locked", "timeframe": timeframe}

    try:
        rk = rolling_key
        if rk is None:
            rk = "bars_intraday" if timeframe == "1Min" else "bars_intraday_5m"

        syms = [s.strip().upper() for s in (symbols or []) if str(s).strip()]
        if not syms:
            return {"status": "no_symbols", "timeframe": timeframe}

        rolling = _read_rolling() or {}
        if not isinstance(rolling, dict):
            rolling = {}

        # -----------------------------
        # Rate-limit safety: hard throttle based on last successful end time
        # -----------------------------
        now = datetime.now(timezone.utc)
        last_end = get_last_end(timeframe)

        try:
            gap = float(
                os.getenv(
                    "DT_BARS_MIN_GAP_SEC_1M" if timeframe == "1Min" else "DT_BARS_MIN_GAP_SEC_5M",
                    "",
                )
            )
        except Exception:
            gap = 0.0
        if gap <= 0.0:
            gap = 25.0 if timeframe == "1Min" else 90.0

        if last_end is not None and (now - last_end).total_seconds() < gap:
            return {"status": "throttled", "timeframe": timeframe, "gap_sec": float(gap)}

        # Incremental window: fetch only since last_end (with a small overlap).
        overlap_sec = 120 if timeframe == "1Min" else 600
        start_dt = (last_end - timedelta(seconds=overlap_sec)) if last_end is not None else None
        end_dt = now

        updated_syms = 0
        new_bars_total = 0

        # Batch fetch to stay friendly to Alpaca.
        for i in range(0, len(syms), max(1, int(batch_size))):
            batch = syms[i : i + max(1, int(batch_size))]
            fetched = fetch_bars_batch(
                batch,
                timeframe=timeframe,
                lookback_minutes=lookback_minutes,
                start_dt=start_dt,
                end_dt=end_dt,
                bars_url=bars_url,
            )
            if not fetched:
                continue

            for sym, new_list in fetched.items():
                node = ensure_symbol_node(rolling, sym)
                existing = node.get(rk) or []
                if not isinstance(existing, list):
                    existing = []

                merged = _dedupe_merge(existing, new_list, max_len=max_len)
                if len(merged) != len(existing):
                    node[rk] = merged
                    rolling[sym] = node
                    updated_syms += 1
                    new_bars_total += max(0, len(merged) - len(existing))

        # Persist once per call.
        save_rolling(rolling)
        set_last_end(timeframe, now)
        log(
            f"[bars_fetch] ✅ updated rolling ({timeframe}) syms={updated_syms}/{len(syms)} "
            f"new_bars≈{new_bars_total} key={rk}"
        )

        return {
            "status": "ok",
            "timeframe": timeframe,
            "symbols_requested": len(syms),
            "symbols_updated": int(updated_syms),
            "new_bars_est": int(new_bars_total),
            "rolling_key": rk,
        }
    finally:
        release_lock_file(lock)
