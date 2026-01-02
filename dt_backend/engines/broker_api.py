# dt_backend/engines/broker_api.py — v2.2
"""
Abstract broker API for AION dt_backend.

v2.2 — Alpaca PAPER execution + local per-bot allowance ledger
--------------------------------------------------------------
Goals:
  ✅ Keep *per-bot* cash + positions locally (so one bot can't spend the whole Alpaca account)
  ✅ Still place real PAPER orders on Alpaca (https://paper-api.alpaca.markets/v2)
  ✅ Same public interface used by the engine:
        - get_cash() -> float
        - get_positions() -> Dict[str, Position]
        - submit_order(order: Order, last_price: float | None = None) -> Dict[str, Any]

How it works:
  • Each bot process identifies itself via env var DT_BOT_ID (default: "default").
  • Each bot has a local ledger JSON (cash/positions/fills) under:
        <DT_PATHS["da_brains"]>/intraday/brokers/bot_<DT_BOT_ID>.json
    (or DT_BOT_LEDGER_PATH override env var)

  • get_cash/get_positions read ONLY the local ledger.
  • submit_order:
      1) checks allowance (ledger cash / ledger positions)
      2) submits order to Alpaca paper (if keys exist)
      3) polls briefly for fills
      4) updates local ledger from the filled quantity + price
      5) (optional) cancels remaining if partial/unfilled

Allowances:
  • Set DT_BOT_CASH_CAP to seed/reset initial cash if the ledger is missing.
    Example: DT_BOT_CASH_CAP=2500  (per-bot "allowance")
  • You can top-up by editing the ledger file or writing an admin endpoint later.

Important:
  • Alpaca positions are *account-level*; this ledger is the strategy-level source of truth.
  • Best practice: avoid multiple bots trading the same symbol simultaneously unless you add symbol ownership/netting.

Secrets:
  • Uses ROOT admin_keys.py (env-only) for Alpaca keys/base URL.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone
import os
import json
import time
import uuid
import urllib.request
import urllib.error

from dt_backend.core import DT_PATHS, log
from dt_backend.core.time_override_dt import utc_iso, now_utc

# ---------------------------------------------------------------------------
# Public interface (compat)
# ---------------------------------------------------------------------------
#
# Some older / external modules expect a BrokerAPI class with methods.
# This file originally provided free functions (get_cash/get_positions/submit_order).
# To keep the system stable, we provide a small wrapper class that delegates to
# the existing functions.

# ROOT secrets (env-only)
try:
    from admin_keys import (
        ALPACA_API_KEY_ID,
        ALPACA_API_SECRET_KEY,
        ALPACA_PAPER_BASE_URL,
        # legacy aliases (in case other code expects them)
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
    side: str           # "BUY" or "SELL"
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
        return default


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


# =========================
# Bot identity + ledger path
# =========================

def _bot_id() -> str:
    bid = _env("DT_BOT_ID", "default")
    bid = "".join(c for c in bid if c.isalnum() or c in ("_", "-", "."))
    return bid or "default"


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

# Backward-compat alias (older code imports PAPER_STATE_PATH)
# In v2.2 this points at the per-bot ledger for the current DT_BOT_ID.
PAPER_STATE_PATH: Path = LEDGER_PATH


def _starting_cash_cap() -> float:
    """
    Seeds initial cash if ledger doesn't exist.
    Defaults to 100k if DT_BOT_CASH_CAP not provided.
    """
    cap = _env("DT_BOT_CASH_CAP", "")
    if cap:
        return max(0.0, _safe_float(cap, 0.0))
    return 100_000.0


def _default_ledger() -> Dict[str, Any]:
    cap = _starting_cash_cap()
    now = _utc_now_iso()
    return {
        "bot_id": _bot_id(),
        "cash": float(cap),
        "positions": {},   # { "AAPL": {"qty": 10, "avg_price": 180.0} }
        "fills": [],       # list of fill dicts
        "meta": {
            "created_at": now,
            "updated_at": now,
            "cash_cap": float(cap),
            "venue": "alpaca_paper+local" if _alpaca_enabled() else "local_only",
        },
    }


def _read_ledger() -> Dict[str, Any]:
    if not LEDGER_PATH.exists():
        return _default_ledger()
    try:
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return _default_ledger()

        data.setdefault("bot_id", _bot_id())
        data.setdefault("cash", _starting_cash_cap())
        data.setdefault("positions", {})
        data.setdefault("fills", [])
        meta = data.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("created_at", _utc_now_iso())
        meta.setdefault("updated_at", _utc_now_iso())
        meta.setdefault("cash_cap", _starting_cash_cap())
        meta.setdefault("venue", "alpaca_paper+local" if _alpaca_enabled() else "local_only")
        data["meta"] = meta

        if not isinstance(data.get("positions"), dict):
            data["positions"] = {}
        if not isinstance(data.get("fills"), list):
            data["fills"] = []

        return data
    except Exception as e:
        log(f"[broker_ledger] ⚠️ failed to read ledger: {e}")
        return _default_ledger()


def _save_ledger(state: Dict[str, Any]) -> None:
    try:
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        state = state if isinstance(state, dict) else _default_ledger()
        state.setdefault("meta", {})
        state["meta"]["updated_at"] = _utc_now_iso()
        tmp = LEDGER_PATH.with_suffix(LEDGER_PATH.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        tmp.replace(LEDGER_PATH)
    except Exception as e:
        log(f"[broker_ledger] ⚠️ failed to save ledger: {e}")


# =========================
# Alpaca PAPER client (stdlib only)
# =========================

def _alpaca_keys() -> tuple[str, str]:
    key = (ALPACA_API_KEY_ID or ALPACA_KEY or "").strip()
    secret = (ALPACA_API_SECRET_KEY or ALPACA_SECRET or "").strip()
    return key, secret


def _alpaca_enabled() -> bool:
    k, s = _alpaca_keys()
    return bool(k and s)


def _alpaca_base_v2() -> str:
    base = (ALPACA_PAPER_BASE_URL or "").strip() or "https://paper-api.alpaca.markets/v2"
    base = base.rstrip("/")
    if not base.endswith("/v2"):
        base = base + "/v2"
    return base


def _alpaca_headers() -> Dict[str, str]:
    k, s = _alpaca_keys()
    return {
        "APCA-API-KEY-ID": k,
        "APCA-API-SECRET-KEY": s,
        "Content-Type": "application/json",
    }


def _http_json(method: str, path: str, payload: Dict[str, Any] | None = None, timeout: float = 15.0) -> Any:
    url = _alpaca_base_v2() + path
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method.upper())
    for hk, hv in _alpaca_headers().items():
        req.add_header(hk, hv)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if not raw:
                return None
            try:
                return json.loads(raw)
            except Exception:
                return raw
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(f"alpaca {method} {path} {e.code}: {body[:500]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"alpaca {method} {path} urlerror: {e}")


def _alpaca_post(path: str, payload: Dict[str, Any]) -> Any:
    return _http_json("POST", path, payload=payload, timeout=20.0)


def _alpaca_get(path: str) -> Any:
    return _http_json("GET", path, payload=None, timeout=15.0)


def _alpaca_delete(path: str) -> Any:
    return _http_json("DELETE", path, payload=None, timeout=15.0)


def _fmt_qty(q: float) -> str:
    if q <= 0:
        return "0"
    s = f"{q:.8f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _client_order_id() -> str:
    # Alpaca client_order_id max length is 48 chars; keep it tight.
    bid = _bot_id()[:14]
    suffix = uuid.uuid4().hex[:16]
    return f"DT{bid}-{suffix}"[:48]


def _poll_order(order_id: str, max_wait_s: float = 4.0) -> Dict[str, Any]:
    deadline = time.time() + max_wait_s
    last: Dict[str, Any] = {}
    while time.time() < deadline:
        o = _alpaca_get(f"/orders/{order_id}")
        if isinstance(o, dict):
            last = o
            status = str(o.get("status", "")).lower()
            filled_qty = _safe_float(o.get("filled_qty") or 0.0, 0.0)
            if status in {"filled", "canceled", "rejected"}:
                return o
            if filled_qty > 0 and status in {"partially_filled", "accepted", "new"}:
                return o
        time.sleep(0.35)
    return last or {"status": "unknown"}


def _cancel_order(order_id: str) -> None:
    try:
        _alpaca_delete(f"/orders/{order_id}")
    except Exception as e:
        log(f"[broker_alpaca] ⚠️ cancel failed for {order_id}: {e}")


# =========================
# Local allowance ledger API (public read)
# =========================

def get_positions() -> Dict[str, Position]:
    state = _read_ledger()
    positions_raw = state.get("positions") or {}
    out: Dict[str, Position] = {}
    if not isinstance(positions_raw, dict):
        return out
    for sym, node in positions_raw.items():
        if not isinstance(node, dict):
            continue
        qty = _safe_float(node.get("qty", 0.0), 0.0)
        avg = _safe_float(node.get("avg_price", 0.0), 0.0)
        sym2 = str(sym).upper().strip()
        if sym2 and qty != 0:
            out[sym2] = Position(symbol=sym2, qty=qty, avg_price=avg)
    return out


def get_cash() -> float:
    state = _read_ledger()
    return _safe_float(state.get("cash", 0.0), 0.0)


def get_ledger_state() -> Dict[str, Any]:
    """Return the full per-bot local ledger (cash/positions/fills/meta).

    This is intentionally read-only; mutations happen through submit_order().
    """
    return _read_ledger()



# =========================
# Execution: Alpaca if enabled, else pure local simulation
# =========================

def _ledger_apply_fill(state: Dict[str, Any], sym: str, side: str, filled_qty: float, fill_price: float) -> Dict[str, Any]:
    positions = state.get("positions") or {}
    if not isinstance(positions, dict):
        positions = {}
    fills = state.get("fills") or []
    if not isinstance(fills, list):
        fills = []

    cash = _safe_float(state.get("cash", 0.0), 0.0)

    pos = positions.get(sym) or {"qty": 0.0, "avg_price": 0.0}
    pos_qty = _safe_float(pos.get("qty", 0.0), 0.0)
    pos_avg = _safe_float(pos.get("avg_price", 0.0), 0.0)

    realized_pnl = 0.0

    if side == "BUY":
        cost = fill_price * filled_qty
        cash -= cost
        new_qty = pos_qty + filled_qty
        if new_qty <= 0:
            positions.pop(sym, None)
        else:
            new_avg = ((pos_avg * pos_qty) + cost) / new_qty if pos_qty > 0 else fill_price
            positions[sym] = {"qty": new_qty, "avg_price": new_avg}

    else:  # SELL
        proceeds = fill_price * filled_qty
        cash += proceeds
        realized_pnl = (fill_price - pos_avg) * filled_qty
        new_qty = pos_qty - filled_qty
        if new_qty <= 0:
            positions.pop(sym, None)
        else:
            positions[sym] = {"qty": new_qty, "avg_price": pos_avg}

    fill = {
        "t": _utc_now_iso(),
        "bot_id": _bot_id(),
        "symbol": sym,
        "side": side,
        "qty": float(filled_qty),
        "price": float(fill_price),
        "realized_pnl": float(realized_pnl),
        "venue": "alpaca_paper" if _alpaca_enabled() else "local",
    }
    fills.append(fill)

    state["cash"] = float(cash)
    state["positions"] = positions
    state["fills"] = fills
    return state


def submit_order(order: Order, last_price: float | None = None) -> Dict[str, Any]:
    """
    Submit an order with a per-bot local allowance ledger.

    Steps:
      1) Validate side/qty
      2) Validate against local ledger cash/positions
      3) If Alpaca enabled -> send order to Alpaca paper, poll for fill, cancel remainder
         Else -> local simulation fill (requires last_price for price)
      4) Apply fill to local ledger and return normalized fill dict

    Note:
      - For LIMIT orders, if last_price is missing we reject (keeps old behavior).
    """
    sym = str(order.symbol).upper().strip()
    side = str(order.side).upper().strip()
    qty_req = _safe_float(order.qty, 0.0)

    if not sym:
        return {"status": "rejected", "reason": "bad_symbol"}
    if side not in {"BUY", "SELL"}:
        return {"status": "rejected", "reason": "bad_side"}
    if qty_req <= 0:
        return {"status": "rejected", "reason": "bad_qty"}

    state = _read_ledger()
    cash = _safe_float(state.get("cash", 0.0), 0.0)
    positions = state.get("positions") or {}
    if not isinstance(positions, dict):
        positions = {}

    pos = positions.get(sym) or {"qty": 0.0, "avg_price": 0.0}
    pos_qty = _safe_float(pos.get("qty", 0.0), 0.0)
    pos_avg = _safe_float(pos.get("avg_price", 0.0), 0.0)

    # For local validation, we need a reference price to check affordability on BUY.
    # Use last_price when present; else for limit order use limit_price.
    ref_price = None
    if order.limit_price is not None:
        ref_price = _safe_float(order.limit_price, None)  # type: ignore[arg-type]
    elif last_price is not None:
        ref_price = _safe_float(last_price, None)  # type: ignore[arg-type]

    # Allowance checks (local)
    if side == "BUY":
        if ref_price is None:
            return {"status": "rejected", "reason": "no_price_for_buy_check"}
        est_cost = ref_price * qty_req
        if est_cost > cash:
            return {"status": "rejected", "reason": "insufficient_cash_allowance"}
    else:  # SELL
        if pos_qty <= 0:
            return {"status": "rejected", "reason": "no_position_allowance"}
        if qty_req > pos_qty:
            qty_req = pos_qty  # clamp to local position

    # ===== Execute on Alpaca if enabled =====
    if _alpaca_enabled():
        # Keep behavior consistent: require last_price for limit guardrails (optional)
        if order.limit_price is not None and last_price is None:
            return {"status": "rejected", "reason": "no_price_for_limit"}

        alpaca_side = "buy" if side == "BUY" else "sell"
        payload: Dict[str, Any] = {
            "symbol": sym,
            "side": alpaca_side,
            "time_in_force": "day",
            "client_order_id": _client_order_id(),
        }

        if order.limit_price is not None:
            payload["type"] = "limit"
            payload["limit_price"] = str(_safe_float(order.limit_price, 0.0))
            payload["qty"] = _fmt_qty(qty_req)
        else:
            payload["type"] = "market"
            payload["qty"] = _fmt_qty(qty_req)

        try:
            created = _alpaca_post("/orders", payload)
            if not isinstance(created, dict) or not created.get("id"):
                return {"status": "rejected", "reason": "alpaca_no_order_id", "raw": created}

            oid = str(created["id"])
            final = _poll_order(oid, max_wait_s=4.0)

            status = str(final.get("status", "")).lower()
            filled_qty = _safe_float(final.get("filled_qty") or 0.0, 0.0)
            filled_avg = final.get("filled_avg_price")
            fill_price = _safe_float(filled_avg, 0.0)

            # If Alpaca didn't fill quickly, cancel and reject (keeps "instant fill" semantics)
            if filled_qty <= 0:
                _cancel_order(oid)
                return {"status": "rejected", "reason": f"alpaca_not_filled_fast", "id": oid, "alpaca_status": status}

            # If partial fill, cancel remainder so local ledger matches the executed qty
            if status != "filled":
                _cancel_order(oid)

            # Apply fill to local ledger
            state2 = _ledger_apply_fill(state, sym, side, filled_qty, fill_price)
            _save_ledger(state2)

            # Compute realized pnl from local basis (pos_avg before fill for SELL)
            realized_pnl = 0.0
            if side == "SELL":
                realized_pnl = (fill_price - pos_avg) * filled_qty

            out = {
                "status": "filled",
                "id": oid,
                "t": str(final.get("filled_at") or final.get("submitted_at") or _utc_now_iso()),
                "symbol": sym,
                "side": side,
                "qty": float(filled_qty),
                "price": float(fill_price),
                "realized_pnl": float(realized_pnl),
                "venue": "alpaca_paper",
                "bot_id": _bot_id(),
                "client_order_id": payload.get("client_order_id"),
            }
            log(f"[broker_alpaca] ✅ filled {side} {filled_qty} {sym} @ {fill_price} (bot={_bot_id()})")
            return out

        except Exception as e:
            log(f"[broker_alpaca] ❌ submit failed for {sym}: {e}")
            return {"status": "rejected", "reason": "alpaca_error", "detail": str(e)[:500], "bot_id": _bot_id()}

    # ===== Local simulation fallback =====
    if last_price is None:
        return {"status": "rejected", "reason": "no_price_local"}

    fill_price = _safe_float(last_price, 0.0)
    # Limit behavior: only fill if favorable vs last_price (old logic)
    if order.limit_price is not None:
        lp = _safe_float(order.limit_price, 0.0)
        if side == "BUY" and fill_price > lp:
            return {"status": "rejected", "reason": "limit_not_reached"}
        if side == "SELL" and fill_price < lp:
            return {"status": "rejected", "reason": "limit_not_reached"}
        fill_price = lp

    filled_qty = qty_req

    state2 = _ledger_apply_fill(state, sym, side, filled_qty, fill_price)
    _save_ledger(state2)

    realized_pnl = 0.0
    if side == "SELL":
        realized_pnl = (fill_price - pos_avg) * filled_qty

    log(f"[broker_local] ✅ filled {side} {filled_qty} {sym} @ {fill_price} (bot={_bot_id()})")
    return {
        "status": "filled",
        "t": _utc_now_iso(),
        "symbol": sym,
        "side": side,
        "qty": float(filled_qty),
        "price": float(fill_price),
        "realized_pnl": float(realized_pnl),
        "venue": "local",
        "bot_id": _bot_id(),
    }


class BrokerAPI:
    """Thin wrapper around the module-level broker functions.

    This exists for compatibility with executors that were written against a
    class-based interface.
    """

    def get_cash(self) -> float:
        return get_cash()

    def get_positions(self) -> Dict[str, Position]:
        return get_positions()

    def submit_order(self, order: Order, last_price: float | None = None) -> Dict[str, Any]:
        return submit_order(order, last_price=last_price)
