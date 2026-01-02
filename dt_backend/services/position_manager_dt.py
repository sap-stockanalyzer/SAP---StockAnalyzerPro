# dt_backend/services/position_manager_dt.py — v1.0 (Phase 5)
"""Synthetic bracket + position state machine for dt_backend.

This module upgrades execution behavior without relying on broker-native
bracket orders. It stores a small per-bot position state file and evaluates
exit conditions each cycle.

State file
----------
Written under <DT_PATHS['da_brains']>/intraday/positions/dt_positions.json.

Each open position entry looks like:

    {
      "AAPL": {
        "status": "OPEN"|"CLOSING"|"CLOSED",
        "side": "BUY"|"SELL",
        "qty": 10.0,
        "entry_price": 180.12,
        "entry_ts": "...Z",
        "stop": 176.00,
        "take_profit": 186.00,
        "trail": true,
        "trail_atr_mult": 1.2,
        "partials": true,
        "partial_taken": false,
        "time_stop_min": 45,
        "scratch_min": 12,
        "scratch_atr_frac": 0.15,
        "bot": "ORB",
        "reason": "...",
        "last_exit_ts": "...Z"    # set on close
      }
    }

Design goals
------------
* Safe-by-default: never raises in the main loop.
* Deterministic and observable: logs exits/updates to dt_trades.jsonl.
* Broker-agnostic: uses only BrokerAPI.submit_order() market/limit.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from dt_backend.core import DT_PATHS
from dt_backend.core.time_override_dt import now_utc as _now_utc_override
from dt_backend.core.logger_dt import log
from dt_backend.services.dt_truth_store import append_trade_event


def _utc_iso(now_utc: Optional[datetime] = None) -> str:
    # Replay/backtest can drive time via DT_NOW_UTC.
    dt = now_utc or _now_utc_override()
    return dt.isoformat().replace("+00:00", "Z")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _intraday_dir() -> Path:
    """Resolve intraday artifact directory.

    Replay/backtest can override the artifact root by setting
        DT_TRUTH_DIR=/abs/path/to/run_dir
    """
    override = (os.getenv("DT_TRUTH_DIR", "") or "").strip()
    if override:
        base = Path(override) / "intraday"
        base.mkdir(parents=True, exist_ok=True)
        return base

    da = DT_PATHS.get("da_brains")
    if isinstance(da, Path):
        base = da / "intraday"
    else:
        base = Path("da_brains") / "intraday"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _pos_state_path() -> Path:
    p = _intraday_dir() / "positions" / "dt_positions.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_positions_state() -> Dict[str, Any]:
    p = _pos_state_path()
    try:
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_positions_state(state: Dict[str, Any]) -> None:
    p = _pos_state_path()
    try:
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception as e:
        log(f"[pos_mgr] ⚠️ failed to write dt_positions: {e}")


def _ny_tz():
    if ZoneInfo is not None:
        return ZoneInfo("America/New_York")
    return timezone.utc


def _is_eod_flatten_time(now_utc: datetime, *, minutes_before_close: int) -> bool:
    """Return True if we are within N minutes of the 16:00 NY close."""
    ny = _ny_tz()
    now_ny = now_utc.astimezone(ny)
    close_ny = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    delta = close_ny - now_ny
    return 0 <= delta.total_seconds() <= max(0, int(minutes_before_close)) * 60


def _extract_last_price(node: Dict[str, Any]) -> float:
    feat = node.get("features_dt")
    if isinstance(feat, dict):
        px = _safe_float(feat.get("last_price"), 0.0)
        if px > 0:
            return px

    # fallback to last close from bars
    for k in ("bars_intraday_5m", "bars_intraday"):
        bars = node.get(k)
        if isinstance(bars, list) and bars:
            b = bars[-1] if isinstance(bars[-1], dict) else None
            if b:
                px = _safe_float(b.get("c") or b.get("close"), 0.0)
                if px > 0:
                    return px
    return 0.0


def _extract_atr(node: Dict[str, Any]) -> float:
    feat = node.get("features_dt")
    if isinstance(feat, dict):
        v = _safe_float(feat.get("atr_14"), 0.0)
        return v
    return 0.0


def _parse_iso(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = str(ts).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


@dataclass
class ExitSummary:
    evaluated: int = 0
    exits_sent: int = 0
    partials_sent: int = 0
    eod_flattens: int = 0


def record_entry(
    *,
    symbol: str,
    side: str,
    qty: float,
    entry_price: float,
    risk: Dict[str, Any] | None,
    bot: str | None = None,
    reason: str | None = None,
    trail_atr_mult: float = 1.2,
    scratch_min: int = 12,
    scratch_atr_frac: float = 0.15,
    now_utc: Optional[datetime] = None,
    meta: Optional[Dict[str, Any]] = None,
    confidence: Optional[float] = None,
) -> None:
    """Create/overwrite a position state entry for a new fill.

    now_utc/meta exist to make historical replay deterministic and
    to attach regime/time-window tags without polluting core logic.
    """
    symbol = str(symbol).upper().strip()
    if not symbol or qty <= 0 or entry_price <= 0:
        return

    ts = _utc_iso(now_utc)
    risk = risk if isinstance(risk, dict) else {}
    meta = meta if isinstance(meta, dict) else {}

    st = read_positions_state()
    st[symbol] = {
        "status": "OPEN",
        "side": str(side).upper(),
        "qty": float(qty),
        "entry_price": float(entry_price),
        "entry_ts": ts,
        "stop": risk.get("stop"),
        "take_profit": risk.get("take_profit"),
        "trail": bool(risk.get("trail") is True),
        "trail_atr_mult": float(_safe_float(risk.get("trail_atr_mult"), trail_atr_mult)),
        "partials": bool(risk.get("partials") is True),
        "partial_taken": False,
        "time_stop_min": risk.get("time_stop_min"),
        "scratch_min": int(_safe_float(risk.get("scratch_min"), scratch_min)),
        "scratch_atr_frac": float(_safe_float(risk.get("scratch_atr_frac"), scratch_atr_frac)),
        "bot": str(bot).upper() if bot else None,
        "reason": str(reason)[:280] if reason else None,
        "last_exit_ts": None,
        "last_exit_reason": None,
        "max_favorable": float(entry_price),
        "min_favorable": float(entry_price),
        "meta": meta,
        "confidence": float(confidence) if confidence is not None else None,
    }
    write_positions_state(st)

    append_trade_event({
        "ts": ts,
        "type": "bracket_set",
        "symbol": symbol,
        "side": str(side).upper(),
        "qty": float(qty),
        "entry_price": float(entry_price),
        "stop": st[symbol].get("stop"),
        "take_profit": st[symbol].get("take_profit"),
        "trail": st[symbol].get("trail"),
        "time_stop_min": st[symbol].get("time_stop_min"),
        "bot": st[symbol].get("bot"),
        "reason": st[symbol].get("reason"),
        "meta": meta,
        "confidence": float(confidence) if confidence is not None else None,
    })



def record_exit(symbol: str, *, reason: str, now_utc: Optional[datetime] = None) -> None:
    st = read_positions_state()
    sym = str(symbol).upper().strip()
    if sym in st and isinstance(st[sym], dict):
        st[sym]["status"] = "CLOSED"
        st[sym]["last_exit_ts"] = _utc_iso(now_utc)
        st[sym]["last_exit_reason"] = str(reason)[:120]
        write_positions_state(st)


def recent_exit_info(symbol: str) -> Tuple[Optional[datetime], str]:
    st = read_positions_state()
    sym = str(symbol).upper().strip()
    if sym not in st or not isinstance(st[sym], dict):
        return None, ""
    ts = _parse_iso(st[sym].get("last_exit_ts"))
    side = str(st[sym].get("side") or "").upper()
    return ts, side


def process_exits(
    *,
    rolling: Dict[str, Any],
    broker: Any,
    dry_run: bool,
    eod_flatten: bool,
    eod_flatten_minutes: int,
    now_utc: Optional[datetime] = None,
) -> ExitSummary:
    """Evaluate open positions and submit synthetic exits.

    Exits are sent as market orders via BrokerAPI.
    """
    out = ExitSummary()
    now = now_utc or _now_utc_override()
    state = read_positions_state()
    if not isinstance(state, dict) or not state:
        return out

    positions = broker.get_positions() if hasattr(broker, "get_positions") else {}
    positions = positions if isinstance(positions, dict) else {}

    # EOD flatten signal
    do_eod = bool(eod_flatten and _is_eod_flatten_time(now, minutes_before_close=int(eod_flatten_minutes)))

    for sym, ps in list(state.items()):
        if not isinstance(sym, str) or not isinstance(ps, dict):
            continue
        sym_u = sym.upper()
        if str(ps.get("status") or "").upper() != "OPEN":
            continue

        out.evaluated += 1

        pos = positions.get(sym_u)
        pos_qty = _safe_float(getattr(pos, "qty", 0.0) if pos is not None else 0.0, 0.0)
        if pos is None or pos_qty == 0.0:
            # Position gone; close state.
            ps["status"] = "CLOSED"
            ps["last_exit_ts"] = _utc_iso(now)
            ps["last_exit_reason"] = "ledger_position_missing"
            state[sym_u] = ps
            continue

        node = rolling.get(sym_u) if isinstance(rolling, dict) else None
        node = node if isinstance(node, dict) else {}
        last = _extract_last_price(node)
        atr = _extract_atr(node)
        if last <= 0:
            continue

        side = str(ps.get("side") or "BUY").upper()
        entry = _safe_float(ps.get("entry_price"), 0.0)
        stop = _safe_float(ps.get("stop"), 0.0) if ps.get("stop") is not None else 0.0
        tp = _safe_float(ps.get("take_profit"), 0.0) if ps.get("take_profit") is not None else 0.0

        # Track favorable excursion for trailing.
        ps["max_favorable"] = max(_safe_float(ps.get("max_favorable"), entry), last)
        ps["min_favorable"] = min(_safe_float(ps.get("min_favorable"), entry), last)

        # EOD flatten overrides all.
        if do_eod:
            exit_reason = "eod_flatten"
            exit_side = "SELL" if side == "BUY" else "BUY"
            qty = float(pos_qty)
            append_trade_event({"ts": _utc_iso(now), "bot": ps.get("bot"), "meta": ps.get("meta"), "type": "exit_signal", "symbol": sym_u, "reason": exit_reason, "last": last, "qty": qty})
            if not dry_run:
                broker.submit_order(type("O", (), {"symbol": sym_u, "side": exit_side, "qty": qty, "limit_price": None})(), last_price=last)
            out.exits_sent += 1
            out.eod_flattens += 1
            ps["status"] = "CLOSED"
            ps["last_exit_ts"] = _utc_iso(now)
            ps["last_exit_reason"] = exit_reason
            state[sym_u] = ps
            continue

        # Time stop
        t0 = _parse_iso(ps.get("entry_ts"))
        tstop = int(_safe_float(ps.get("time_stop_min"), 0.0))
        if t0 is not None and tstop > 0:
            if (now - t0).total_seconds() >= tstop * 60:
                exit_reason = "time_stop"
                exit_side = "SELL" if side == "BUY" else "BUY"
                qty = float(pos_qty)
                append_trade_event({"ts": _utc_iso(now), "bot": ps.get("bot"), "meta": ps.get("meta"), "type": "exit_signal", "symbol": sym_u, "reason": exit_reason, "last": last, "qty": qty})
                if not dry_run:
                    broker.submit_order(type("O", (), {"symbol": sym_u, "side": exit_side, "qty": qty, "limit_price": None})(), last_price=last)
                out.exits_sent += 1
                ps["status"] = "CLOSED"
                ps["last_exit_ts"] = _utc_iso(now)
                ps["last_exit_reason"] = exit_reason
                state[sym_u] = ps
                continue

        # Scratch rule: if it doesn't go anywhere, exit.
        scratch_min = int(_safe_float(ps.get("scratch_min"), 0.0))
        scratch_atr_frac = float(_safe_float(ps.get("scratch_atr_frac"), 0.0))
        if t0 is not None and scratch_min > 0 and atr > 0 and scratch_atr_frac > 0:
            if (now - t0).total_seconds() >= scratch_min * 60:
                # Require at least X*ATR favorable move, else scratch.
                min_move = scratch_atr_frac * atr
                favorable = (last - entry) if side == "BUY" else (entry - last)
                if favorable < min_move:
                    exit_reason = "scratch"
                    exit_side = "SELL" if side == "BUY" else "BUY"
                    qty = float(pos_qty)
                    append_trade_event({"ts": _utc_iso(now), "bot": ps.get("bot"), "meta": ps.get("meta"), "type": "exit_signal", "symbol": sym_u, "reason": exit_reason, "last": last, "qty": qty, "atr": atr})
                    if not dry_run:
                        broker.submit_order(type("O", (), {"symbol": sym_u, "side": exit_side, "qty": qty, "limit_price": None})(), last_price=last)
                    out.exits_sent += 1
                    ps["status"] = "CLOSED"
                    ps["last_exit_ts"] = _utc_iso(now)
                    ps["last_exit_reason"] = exit_reason
                    state[sym_u] = ps
                    continue

        # Trailing stop update
        if bool(ps.get("trail") is True) and atr > 0:
            trail_mult = float(_safe_float(ps.get("trail_atr_mult"), 1.2))
            if side == "BUY":
                new_stop = _safe_float(ps.get("max_favorable"), last) - trail_mult * atr
                if stop <= 0 or new_stop > stop:
                    stop = max(0.01, float(new_stop))
                    ps["stop"] = stop
            else:
                new_stop = _safe_float(ps.get("min_favorable"), last) + trail_mult * atr
                if stop <= 0 or new_stop < stop:
                    stop = max(0.01, float(new_stop))
                    ps["stop"] = stop

        # Take profit -> partials
        if tp > 0 and bool(ps.get("partials") is True) and not bool(ps.get("partial_taken") is True):
            hit_tp = (last >= tp) if side == "BUY" else (last <= tp)
            if hit_tp:
                exit_side = "SELL" if side == "BUY" else "BUY"
                qty = max(0.0, float(pos_qty) * 0.5)
                if qty > 0:
                    append_trade_event({"ts": _utc_iso(now), "type": "partial_signal", "bot": ps.get("bot"), "meta": ps.get("meta"), "symbol": sym_u, "reason": "tp_partial", "last": last, "qty": qty, "tp": tp})
                    if not dry_run:
                        broker.submit_order(type("O", (), {"symbol": sym_u, "side": exit_side, "qty": qty, "limit_price": None})(), last_price=last)
                    out.partials_sent += 1
                    ps["partial_taken"] = True
                    # After partial, let trail manage the rest.
                    ps["take_profit"] = None

        # Hard stop
        if stop > 0:
            stop_hit = (last <= stop) if side == "BUY" else (last >= stop)
            if stop_hit:
                exit_reason = "stop_hit"
                exit_side = "SELL" if side == "BUY" else "BUY"
                qty = float(pos_qty)
                append_trade_event({"ts": _utc_iso(now), "bot": ps.get("bot"), "meta": ps.get("meta"), "type": "exit_signal", "symbol": sym_u, "reason": exit_reason, "last": last, "qty": qty, "stop": stop})
                if not dry_run:
                    broker.submit_order(type("O", (), {"symbol": sym_u, "side": exit_side, "qty": qty, "limit_price": None})(), last_price=last)
                out.exits_sent += 1
                ps["status"] = "CLOSED"
                ps["last_exit_ts"] = _utc_iso(now)
                ps["last_exit_reason"] = exit_reason
                state[sym_u] = ps
                continue

        # If no partials, full TP exit
        if tp > 0 and not bool(ps.get("partials") is True):
            hit_tp = (last >= tp) if side == "BUY" else (last <= tp)
            if hit_tp:
                exit_reason = "take_profit"
                exit_side = "SELL" if side == "BUY" else "BUY"
                qty = float(pos_qty)
                append_trade_event({"ts": _utc_iso(now), "bot": ps.get("bot"), "meta": ps.get("meta"), "type": "exit_signal", "symbol": sym_u, "reason": exit_reason, "last": last, "qty": qty, "tp": tp})
                if not dry_run:
                    broker.submit_order(type("O", (), {"symbol": sym_u, "side": exit_side, "qty": qty, "limit_price": None})(), last_price=last)
                out.exits_sent += 1
                ps["status"] = "CLOSED"
                ps["last_exit_ts"] = _utc_iso(now)
                ps["last_exit_reason"] = exit_reason
                state[sym_u] = ps
                continue

        state[sym_u] = ps

    write_positions_state(state)
    return out
