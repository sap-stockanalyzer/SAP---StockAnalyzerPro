# dt_backend/engines/broker_api.py — v2.3
"""Abstract broker API for AION dt_backend.

Design summary
--------------
- Strategy-level source of truth is a *local per-bot ledger* (cash/positions/fills)
  so one bot can't spend the whole broker account.
- Orders can still route to Alpaca paper (account-level) when API keys exist.

v2.3 — Broker equity/PnL cache (for risk rails)
----------------------------------------------
Adds an account snapshot cache (TTL default 180s) so other components (notably
risk_rails_dt) can use broker-reported equity/PnL instead of only local ledger
estimates.

Public additions:
- get_account_cached(ttl_sec=180) -> dict
- get_equity_cached(ttl_sec=180) -> float
- BrokerAPI.get_account_cached(ttl_sec=180)

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import json
import os
import time
import uuid
import urllib.request
import urllib.error

from dt_backend.core import DT_PATHS, log
from dt_backend.core.time_override_dt import utc_iso

from dt_backend.core.position_registry import (
    load_registry,
    save_registry,
    reconcile_with_alpaca_positions,
    can_sell_qty,
    reserve_on_fill,
)

# ROOT secrets (env-only)
try:
    from admin_keys import (
        ALPACA_API_KEY_ID,
        ALPACA_API_SECRET_KEY,
        ALPACA_PAPER_BASE_URL,
        # legacy aliases
        ALPACA_KEY,
        ALPACA_SECRET,
    )
except Exception:
    ALPACA_API_KEY_ID = ""
    ALPACA_API_SECRET_KEY = ""
    ALPACA_PAPER_BASE_URL = ""
    ALPACA_KEY = ""
    ALPACA_SECRET = ""


# =========================
# Models
# =========================


@dataclass
class Position:
    symbol: str
    qty: float
    avg_price: float


@dataclass
class Order:
    symbol: str
    side: str  # "BUY" or "SELL"
    qty: float
    limit_price: float | None = None


# =========================
# Helpers
# =========================


def _utc_now_iso() -> str:
    # Replay/backtest can drive time via DT_NOW_UTC.
    return utc_iso()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


# =========================
# Bot identity + ledger path
# =========================


def _bot_id() -> str:
    bid = _env("DT_BOT_ID", "default")
    bid = "".join(c for c in bid if c.isalnum() or c in ("_", "-", "."))
    return bid or "default"


def _strategy_owner() -> str:
    """Strategy ownership tag written into the shared registry.

    DT should use 'DT'. Swing should use 'SWING' (or 'SW'), etc.
    """
    tag = _env("AION_STRATEGY_TAG", "") or _env("DT_STRATEGY_TAG", "") or "DT"
    tag = "".join(c for c in tag.upper() if c.isalnum() or c in {"_", "-"})
    return tag or "DT"



def _resolve_ledger_path() -> Path:
    override = _env("DT_BOT_LEDGER_PATH", "")
    if override:
        return Path(override)

    da = DT_PATHS.get("da_brains")
    if isinstance(da, Path):
        return da / "intraday" / "brokers" / f"bot_{_bot_id()}.json"

    # fallback
    return Path("bot_ledger.json")


LEDGER_PATH: Path = _resolve_ledger_path()

# Backward-compat alias
PAPER_STATE_PATH: Path = LEDGER_PATH


def _alpaca_keys() -> tuple[str, str]:
    key = (ALPACA_API_KEY_ID or ALPACA_KEY or "").strip()
    secret = (ALPACA_API_SECRET_KEY or ALPACA_SECRET or "").strip()
    return key, secret


def _alpaca_enabled() -> bool:
    k, s = _alpaca_keys()
    return bool(k and s)

# (… file continues exactly as you pasted …)

# =========================
# Strategy ownership registry reconcile (v2.4)
# =========================

_OWNERSHIP_RECONCILE_TS: float = 0.0
_OWNERSHIP_RECONCILE_LAST: dict = {}


def reconcile_ownership_cached(*, ttl_sec: int = 180, force: bool = False) -> dict:
    """Refresh ownership registry against Alpaca positions (safety only)."""
    global _OWNERSHIP_RECONCILE_TS, _OWNERSHIP_RECONCILE_LAST

    if not _alpaca_enabled():
        return {}

    now = time.time()
    ttl = max(1, int(ttl_sec))
    if (not force) and _OWNERSHIP_RECONCILE_LAST and (now - _OWNERSHIP_RECONCILE_TS) < ttl:
        return _OWNERSHIP_RECONCILE_LAST

    try:
        broker_pos = _alpaca_positions()
        reg = load_registry()
        qty_map = {sym: float(p.qty) for sym, p in broker_pos.items()}
        summary = reconcile_with_alpaca_positions(reg, qty_map)
        save_registry(reg)
        _OWNERSHIP_RECONCILE_LAST = summary
        _OWNERSHIP_RECONCILE_TS = now
        if isinstance(summary, dict) and summary.get('mismatch_symbols'):
            log(f"[broker_ownership] 🧯 mismatch detected: {summary.get('mismatch_symbols')}")
        return summary
    except Exception as e:
        log(f"[broker_ownership] ⚠️ reconcile failed: {e}")
        return _OWNERSHIP_RECONCILE_LAST if isinstance(_OWNERSHIP_RECONCILE_LAST, dict) else {}

# =========================
# Execution helpers
# =========================

def _ledger_apply_fill(state: Dict[str, Any], sym: str, side: str, filled_qty: float, fill_price: float) -> Dict[str, Any]:
    # ... (snip) ...
    owner = _strategy_owner()
    reg = load_registry()
    if side == "SELL":
        allowed = can_sell_qty(reg, sym, owner)
        if allowed <= 0.0:
            return {"status": "rejected", "reason": "not_owned_by_strategy", "symbol": sym, "strategy": owner}
        if filled_qty > allowed:
            filled_qty = allowed
    # ... (snip) ...
