"""
utils/refresh_universes_from_alpaca.py

One-shot universe refresh tool:
- Pulls ALL tradable + active Alpaca assets (US equities/ETFs) via Trading API /v2/assets
- Writes universe files in the SAME format your backend expects: {"symbols": [...]}

Outputs (in PATHS["universe"]):
- master_universe.json  (swing-compatible, used by existing code)
- swing_universe.json   (same as master; explicit split)
- dt_universe.json      (filtered "day-trade friendly" universe using Snapshot liquidity)

Designed so you can:
- run it as a CLI tool now
- call refresh_universes(...) from an admin route later
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests

from backend.core.config import (
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    ALPACA_PAPER_BASE_URL,
    PATHS,
)
from backend.core.data_pipeline import log


# ---------------------------------------------------------------------
# Defaults / knobs (env-overridable)
# ---------------------------------------------------------------------

DEFAULT_TRADING_BASE = (ALPACA_PAPER_BASE_URL or "").strip() or "https://paper-api.alpaca.markets"
DEFAULT_DATA_BASE = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets").strip()

# Snapshot batching safety (avoid URL length & API limits headaches)
SNAPSHOT_BATCH = int(os.getenv("AION_UNIVERSE_SNAPSHOT_BATCH", "200"))

# DT liquidity filters (NO price filtering — penny stocks are allowed)
DT_MIN_DAILY_VOLUME = int(os.getenv("AION_DT_MIN_DAILY_VOLUME", "75000"))         # shares
DT_MIN_DOLLAR_VOLUME = float(os.getenv("AION_DT_MIN_DOLLAR_VOLUME", "250000"))   # USD

# Asset filters
INCLUDE_FRACTIONABLE = os.getenv("AION_UNIVERSE_REQUIRE_FRACTIONABLE", "0") == "1"
ONLY_US_EQUITY = os.getenv("AION_UNIVERSE_ONLY_US_EQUITY", "1") == "1"


# ---------------------------------------------------------------------
# Paths (matches your existing universe usage)
# ---------------------------------------------------------------------

UNIVERSE_DIR: Path = Path(PATHS["universe"])
MASTER_FILE: Path = UNIVERSE_DIR / "master_universe.json"
SWING_FILE: Path = UNIVERSE_DIR / "swing_universe.json"
DT_FILE: Path = UNIVERSE_DIR / "dt_universe.json"


# ---------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------

def _alpaca_headers() -> Dict[str, str]:
    if not ALPACA_API_KEY_ID or not ALPACA_API_SECRET_KEY:
        raise RuntimeError("Missing Alpaca keys. Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY.")
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY_ID,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET_KEY,
    }


def _get_json(url: str, *, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Any:
    r = requests.get(url, headers=_alpaca_headers(), params=params, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"GET {url} failed: {r.status_code} {r.text[:500]}")
    return r.json()


def _chunks(items: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------
# Alpaca assets + snapshots
# ---------------------------------------------------------------------

def fetch_tradable_assets(
    *,
    trading_base_url: str = DEFAULT_TRADING_BASE,
    status: str = "active",
    tradable: bool = True,
) -> List[Dict[str, Any]]:
    """
    Pull Alpaca assets.

    Uses Trading API:
      GET {trading_base_url}/v2/assets?status=active&tradable=true

    Returns: list of asset dicts.
    """
    trading_base_url = trading_base_url.rstrip("/")
    url = f"{trading_base_url}/v2/assets"
    params = {
        "status": status,
        "tradable": "true" if tradable else "false",
    }
    assets = _get_json(url, params=params, timeout=45)
    if not isinstance(assets, list):
        raise RuntimeError(f"Unexpected assets payload type: {type(assets)}")
    return assets


def _asset_symbol(a: Dict[str, Any]) -> str:
    return str(a.get("symbol") or "").upper().strip()


def _asset_class(a: Dict[str, Any]) -> str:
    # Alpaca uses 'class' in many payloads for asset class
    return str(a.get("class") or a.get("asset_class") or "").lower().strip()


def _asset_ok_base(a: Dict[str, Any]) -> bool:
    # Must have symbol
    sym = _asset_symbol(a)
    if not sym:
        return False

    # Tradable & active (we already query for this, but belt+suspenders)
    if str(a.get("status") or "").lower() != "active":
        return False
    if bool(a.get("tradable")) is False:
        return False

    if ONLY_US_EQUITY:
        if _asset_class(a) not in ("us_equity", "usequity"):
            return False

    if INCLUDE_FRACTIONABLE:
        if bool(a.get("fractionable")) is False:
            return False

    # Avoid obviously broken symbols
    if " " in sym:
        return False
    return True


def fetch_snapshots(
    symbols: List[str],
    *,
    data_base_url: str = DEFAULT_DATA_BASE,
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch snapshots in batches:
      GET https://data.alpaca.markets/v2/stocks/snapshots?symbols=AAPL,TSLA,...

    Returns: dict keyed by symbol -> snapshot object
    """
    data_base_url = data_base_url.rstrip("/")
    url = f"{data_base_url}/v2/stocks/snapshots"

    out: Dict[str, Dict[str, Any]] = {}
    syms = [s.upper().strip() for s in symbols if s and s.strip()]
    for batch in _chunks(syms, SNAPSHOT_BATCH):
        params = {"symbols": ",".join(batch)}
        js = _get_json(url, params=params, timeout=45)
        if isinstance(js, dict):
            # API returns { "AAPL": {...}, "TSLA": {...} }
            for k, v in js.items():
                if isinstance(v, dict):
                    out[str(k).upper()] = v
        time.sleep(0.05)  # tiny politeness delay
    return out


def _snapshot_liquidity(snapshot: Dict[str, Any]) -> Tuple[float, float]:
    """
    Returns (daily_volume_shares, approx_dollar_volume).

    dailyBar.v = daily volume (shares)
    latestTrade.p = last trade price (or fallback to dailyBar.c close)
    """
    vol = 0.0
    px = None

    try:
        daily = snapshot.get("dailyBar") or {}
        vol = float(daily.get("v") or 0.0)
    except Exception:
        vol = 0.0

    try:
        lt = snapshot.get("latestTrade") or {}
        px = float(lt.get("p")) if lt.get("p") is not None else None
    except Exception:
        px = None

    if px is None:
        try:
            daily = snapshot.get("dailyBar") or {}
            px = float(daily.get("c")) if daily.get("c") is not None else None
        except Exception:
            px = None

    if px is None or not (px > 0):
        # dollar volume unknown, but keep volume
        return vol, 0.0

    return vol, float(px) * float(vol)


# ---------------------------------------------------------------------
# Universe build logic
# ---------------------------------------------------------------------

@dataclass
class UniverseResult:
    total_assets: int
    base_symbols: List[str]
    swing_symbols: List[str]
    dt_symbols: List[str]
    wrote: List[str]


def build_universes(
    assets: List[Dict[str, Any]],
    *,
    trading_base_url: str = DEFAULT_TRADING_BASE,
    data_base_url: str = DEFAULT_DATA_BASE,
    dt_min_daily_volume: int = DT_MIN_DAILY_VOLUME,
    dt_min_dollar_volume: float = DT_MIN_DOLLAR_VOLUME,
) -> UniverseResult:
    """
    - base = all active+tradable assets (filtered to us_equity by default)
    - swing = base (you said you want everything tradable for swing/backfill)
    - dt = liquidity-filtered subset using snapshots
    """
    base: List[str] = []
    for a in assets:
        if _asset_ok_base(a):
            base.append(_asset_symbol(a))

    base = sorted(set(base))
    swing = list(base)

    # DT: liquidity filter via snapshots
    dt: List[str] = []
    if base:
        log(f"[universe] Fetching snapshots for DT filtering: symbols={len(base)} batch={SNAPSHOT_BATCH}")
        snaps = fetch_snapshots(base, data_base_url=data_base_url)

        kept = 0
        missing = 0
        for sym in base:
            snap = snaps.get(sym)
            if not snap:
                missing += 1
                continue

            vol, dvol = _snapshot_liquidity(snap)
            if (vol >= float(dt_min_daily_volume)) or (dvol >= float(dt_min_dollar_volume)):
                dt.append(sym)
                kept += 1

        dt = sorted(set(dt))
        log(
            f"[universe] DT filter done: kept={kept} missing_snapshots={missing} "
            f"(min_vol={dt_min_daily_volume}, min_$vol={dt_min_dollar_volume:,.0f})"
        )

    return UniverseResult(
        total_assets=int(len(assets)),
        base_symbols=base,
        swing_symbols=swing,
        dt_symbols=dt,
        wrote=[],
    )


def write_universe_files(res: UniverseResult) -> UniverseResult:
    """
    Writes:
      - master_universe.json  (swing)
      - swing_universe.json   (swing)
      - dt_universe.json      (dt)
    """
    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    def payload(symbols: List[str], kind: str) -> Dict[str, Any]:
        return {
            "generated_at": now,
            "source": "alpaca",
            "kind": kind,
            "symbols": symbols,
        }

    _atomic_write_json(MASTER_FILE, payload(res.swing_symbols, "master/swing"))
    _atomic_write_json(SWING_FILE, payload(res.swing_symbols, "swing"))
    _atomic_write_json(DT_FILE, payload(res.dt_symbols, "dt"))

    res.wrote = [str(MASTER_FILE), str(SWING_FILE), str(DT_FILE)]
    return res


def refresh_universes(
    *,
    trading_base_url: str = DEFAULT_TRADING_BASE,
    data_base_url: str = DEFAULT_DATA_BASE,
    write_files: bool = True,
) -> UniverseResult:
    log(f"[universe] Refreshing universes from Alpaca…")
    log(f"[universe] trading_base_url={trading_base_url}")
    log(f"[universe] data_base_url={data_base_url}")
    assets = fetch_tradable_assets(trading_base_url=trading_base_url)

    res = build_universes(
        assets,
        trading_base_url=trading_base_url,
        data_base_url=data_base_url,
    )

    log(f"[universe] Base symbols: {len(res.base_symbols)} (swing/master)")
    log(f"[universe] DT symbols:   {len(res.dt_symbols)}")

    if write_files:
        res = write_universe_files(res)
        log(f"[universe] ✅ Wrote universe files:")
        for p in res.wrote:
            log(f"[universe]   - {p}")

    return res


# ---------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Refresh AION universes from Alpaca (master/swing + dt).")
    parser.add_argument("--trading-base", type=str, default=DEFAULT_TRADING_BASE, help="Trading API base URL (paper or live).")
    parser.add_argument("--data-base", type=str, default=DEFAULT_DATA_BASE, help="Market Data base URL.")
    parser.add_argument("--no-write", action="store_true", help="Do everything except writing files.")
    args = parser.parse_args()

    result = refresh_universes(
        trading_base_url=str(args.trading_base).strip(),
        data_base_url=str(args.data_base).strip(),
        write_files=(not args.no_write),
    )

    print(json.dumps({
        "total_assets": result.total_assets,
        "base_symbols": len(result.base_symbols),
        "swing_symbols": len(result.swing_symbols),
        "dt_symbols": len(result.dt_symbols),
        "wrote": result.wrote,
    }, indent=2))
