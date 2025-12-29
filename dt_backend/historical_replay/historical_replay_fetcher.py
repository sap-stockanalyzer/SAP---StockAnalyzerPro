# dt_backend/historical_replay/historical_replay_fetcher.py
"""
Historical Replay Data Fetcher (Alpaca Minute Bars, COMPRESSED)
---------------------------------------------------------------

Writes:
    ml_data_dt/intraday/replay/raw_days/YYYY-MM-DD.json.gz

Compressed ~90% smaller than raw JSON.
"""

from __future__ import annotations

import os
import time
import json
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests

from dt_backend.core.config_dt import DT_PATHS, ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY
from dt_backend.core.logger_dt import log

# ============================================================
# CONFIG
# ============================================================

ALPACA_KEY = os.getenv("ALPACA_API_KEY_ID", ALPACA_API_KEY_ID or "")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET_KEY", ALPACA_API_SECRET_KEY or "")

ALPACA_URL = os.getenv("ALPACA_DATA_BARS_URL", "https://data.alpaca.markets/v2/stocks/bars")
MAX_CALLS_PER_MIN = 175
BATCH_SIZE = 100

UNIVERSE_FILE = Path(DT_PATHS["universe_file"])


# ============================================================
# PATH HELPERS
# ============================================================

def _raw_days_dir() -> Path:
    root = Path(DT_PATHS.get("dtml_data", Path("ml_data_dt")))
    out = root / "intraday" / "replay" / "raw_days"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _raw_day_path(date: str) -> Path:
    # ALWAYS GZIP
    return _raw_days_dir() / f"{date}.json.gz"


# ============================================================
# UNIVERSE LOADING
# ============================================================

def load_universe() -> List[str]:
    if not UNIVERSE_FILE.exists():
        log(f"[fetcher] ⚠ universe file missing: {UNIVERSE_FILE}")
        return []

    try:
        data = json.loads(UNIVERSE_FILE.read_text(encoding='utf-8'))
        if isinstance(data, dict) and 'symbols' in data:
            data = data.get('symbols') or []
        syms = [str(s).upper().strip() for s in (data or []) if str(s).strip()]
        log(f"[fetcher] Loaded {len(syms)} symbols")
        return syms
    except Exception as e:
        log(f"[fetcher] ❌ universe load error: {e}")
        return []


# ============================================================
# HTTP HELPERS
# ============================================================

def _alpaca_headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_KEY or "",
        "APCA-API-SECRET-KEY": ALPACA_SECRET or "",
    }


def fetch_bars_for_batch(symbols: List[str], day: str) -> Dict[str, Any]:
    params = {
        "symbols": ",".join(symbols),
        "timeframe": "1Min",
        "start": f"{day}T09:30:00-04:00",
        "end": f"{day}T16:00:00-04:00",
        "limit": 10000,
    }
    r = requests.get(ALPACA_URL, headers=_alpaca_headers(), params=params, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Alpaca error {r.status_code}: {r.text[:200]}")
    return r.json()


# ============================================================
# FETCH SINGLE DAY — GZIP OUTPUT
# ============================================================

def fetch_day(day: str, universe: List[str]) -> Path:
    out_path = _raw_day_path(day)

    # REUSE cache
    if out_path.exists():
        log(f"[fetcher] ℹ cached → {out_path.name}")
        return out_path

    if not universe:
        out_path.write_bytes(gzip.compress(b"[]"))
        return out_path

    batches = [universe[i:i + BATCH_SIZE] for i in range(0, len(universe), BATCH_SIZE)]
    log(f"[fetcher] Fetching {day}: {len(universe)} symbols in {len(batches)} batches")

    results: List[Dict[str, Any]] = []
    calls_made = 0
    window_start = time.time()

    for idx, batch in enumerate(batches):
        # throttle
        now = time.time()
        elapsed = now - window_start
        if calls_made >= MAX_CALLS_PER_MIN:
            wait = max(0.0, 60.0 - elapsed)
            if wait > 0:
                log(f"[fetcher] ⏳ throttling {wait:.1f}s")
                time.sleep(wait)
            calls_made = 0
            window_start = time.time()

        try:
            resp = fetch_bars_for_batch(batch, day)
            calls_made += 1
        except Exception as e:
            log(f"[fetcher] ❌ batch {idx+1} failed: {e}")
            time.sleep(3)
            continue

        bars = resp.get("bars", {})
        for sym in batch:
            results.append({
                "symbol": sym,
                "bars": bars.get(sym, []),
            })

        log(f"[fetcher] ✓ batch {idx+1}/{len(batches)}")

    # SAVE AS .json.gz
    try:
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            json.dump(results, f)
        log(f"[fetcher] 💾 saved {out_path.name} ({len(results)} symbols)")
    except Exception as e:
        log(f"[fetcher] ❌ save gzip error: {e}")

    return out_path


# ============================================================
# FETCH RANGE
# ============================================================

def fetch_range(start: str, end: str, universe: Optional[List[str]] = None) -> List[str]:
    if universe is None:
        universe = load_universe()

    if not universe:
        log("[fetcher] ⚠ universe empty")
        return []

    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end, "%Y-%m-%d")
    if d1 < d0:
        d0, d1 = d1, d0

    days: List[str] = []
    cur = d0
    while cur <= d1:
        ds = cur.strftime("%Y-%m-%d")
        fetch_day(ds, universe)
        days.append(ds)
        cur += timedelta(days=1)

    return days

