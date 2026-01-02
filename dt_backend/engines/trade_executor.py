# dt_backend/engines/trade_executor.py — v0.1 (Phase 0 executor)
"""Intraday trade executor for dt_backend.

Phase 0 goals
-------------
* Deterministic: every attempted action is logged to dt_trades.jsonl.
* Safe-by-default: DRY RUN unless explicitly enabled.
* Minimal assumptions: consumes execution_dt when present, otherwise policy_dt.

This module is intentionally conservative.
It does not try to be "smart" yet — it only turns intents into orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from dt_backend.core.data_pipeline_dt import _read_rolling
from dt_backend.core.logger_dt import log
from dt_backend.core.time_override_dt import now_utc as _now_utc_override
from dt_backend.services.dt_truth_store import append_trade_event, bump_metric

from dt_backend.engines.broker_api import BrokerAPI, Order
from dt_backend.services.position_manager_dt import (
    process_exits,
    record_entry,
    record_exit,
    recent_exit_info,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ExecutionConfig:
    """Execution controls.

    Notes:
        - dry_run=True means we only log intents; no broker calls.
        - max_orders_per_cycle is a safety cap.
        - allow_shorts=False means SELL only closes existing long positions.
        - min_confidence prevents micro-trades.
    """

    dry_run: bool = True
    max_orders_per_cycle: int = 5
    allow_shorts: bool = False
    min_confidence: float = 0.25
    default_qty: float = 1.0

    # Phase 5: synthetic brackets + state machine
    enable_brackets: bool = True
    eod_flatten: bool = True
    eod_flatten_minutes: int = 5

    # Anti-flip hysteresis (direction changes) after an exit
    min_flip_minutes: int = 12

    # Fallback risk if a plan doesn't specify stop/tp
    fallback_stop_atr: float = 1.25
    fallback_tp_atr: float = 1.75

    # Position management defaults (can be overridden per-plan)
    trail_atr_mult: float = 1.2
    scratch_min: int = 12
    scratch_atr_frac: float = 0.15


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _extract_intent(node: Dict[str, Any]) -> Tuple[str, float, float]:
    """Return (side, size, confidence)."""

    # Prefer execution_dt (sizing + cooldown already applied).
    ex = node.get("execution_dt")
    if isinstance(ex, dict):
        side = str(ex.get("side") or "").upper()
        size = _safe_float(ex.get("size"), 0.0)
        conf = _safe_float(ex.get("confidence_adj"), 0.0)
        if side in {"BUY", "SELL", "FLAT"}:
            return side, size, conf

    # Fallback: policy_dt.
    pol = node.get("policy_dt")
    if isinstance(pol, dict):
        side = str(pol.get("action") or pol.get("intent") or "").upper()
        conf = _safe_float(pol.get("confidence"), 0.0)
        # policy_dt has no explicit size; treat confidence as a weak size proxy.
        size = max(0.0, min(1.0, conf))
        if side in {"BUY", "SELL", "HOLD", "STAND_DOWN"}:
            return ("FLAT" if side in {"HOLD", "STAND_DOWN"} else side), size, conf

    return "FLAT", 0.0, 0.0


def _extract_last_price(node: Dict[str, Any]) -> float:
    # Prefer features_dt
    feat = node.get("features_dt")
    if isinstance(feat, dict):
        try:
            px = float(feat.get("last_price") or 0.0)
            if px > 0:
                return px
        except Exception:
            pass

    # Fallback to last close from bars
    for k in ("bars_intraday_5m", "bars_intraday"):
        bars = node.get(k)
        if isinstance(bars, list) and bars:
            b = bars[-1] if isinstance(bars[-1], dict) else None
            if b:
                try:
                    px = float(b.get("c") or b.get("close") or 0.0)
                    if px > 0:
                        return px
                except Exception:
                    pass
    return 0.0


def _extract_atr(node: Dict[str, Any]) -> float:
    feat = node.get("features_dt")
    if isinstance(feat, dict):
        try:
            return float(feat.get("atr_14") or 0.0)
        except Exception:
            return 0.0
    return 0.0

def _entry_meta_from_global(rolling: Dict[str, Any], now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    """Pull lightweight tags from _GLOBAL_DT for analytics/replay."""
    g = rolling.get("_GLOBAL_DT") if isinstance(rolling, dict) else None
    g = g if isinstance(g, dict) else {}
    rd = g.get("regime_dt") if isinstance(g.get("regime_dt"), dict) else {}
    mr = g.get("micro_regime_dt") if isinstance(g.get("micro_regime_dt"), dict) else {}
    state = g.get("dt_state") if isinstance(g.get("dt_state"), dict) else {}
    out = {
        "regime": rd.get("label"),
        "day_type": rd.get("day_type"),
        "regime_conf": rd.get("confidence"),
        "micro": mr.get("label"),
        "time_window": mr.get("time_window"),
        "risk_mode": state.get("risk_mode"),
        "version": state.get("version"),
    }
    if now_utc is not None:
        out["cycle_ts"] = now_utc.isoformat(timespec="seconds").replace("+00:00", "Z")
    return out



def _plan_risk(node: Dict[str, Any], *, side: str, last_price: float, atr: float, cfg: ExecutionConfig) -> Dict[str, Any]:
    """Return a normalized risk dict for synthetic brackets."""
    plan = node.get("execution_plan_dt")
    risk = plan.get("risk") if isinstance(plan, dict) else None
    risk = risk if isinstance(risk, dict) else {}

    stop = risk.get("stop")
    tp = risk.get("take_profit")

    # Fallback to ATR-based if missing
    if (stop is None or tp is None) and (atr > 0 and last_price > 0):
        if side.upper() == "BUY":
            stop_f = last_price - cfg.fallback_stop_atr * atr
            tp_f = last_price + cfg.fallback_tp_atr * atr
        else:
            stop_f = last_price + cfg.fallback_stop_atr * atr
            tp_f = last_price - cfg.fallback_tp_atr * atr
        if stop is None:
            stop = max(0.01, float(stop_f))
        if tp is None:
            tp = max(0.01, float(tp_f))

    out = {
        "stop": float(stop) if stop is not None else None,
        "take_profit": float(tp) if tp is not None else None,
        "time_stop_min": risk.get("time_stop_min"),
        "partials": bool(risk.get("partials") is True),
        "trail": bool(risk.get("trail") is True),
        "trail_atr_mult": float(_safe_float(risk.get("trail_atr_mult"), cfg.trail_atr_mult)),
        "scratch_min": int(_safe_float(risk.get("scratch_min"), cfg.scratch_min)),
        "scratch_atr_frac": float(_safe_float(risk.get("scratch_atr_frac"), cfg.scratch_atr_frac)),
    }
    return out


def _qty_from_size(size: float, cfg: ExecutionConfig) -> float:
    # Phase 0 uses a simple mapping; we will upgrade sizing later.
    size = max(0.0, min(1.0, float(size)))
    return max(0.0, float(cfg.default_qty) * size)


def execute_from_policy(cfg: Optional[ExecutionConfig] = None, *, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    """Execute one cycle worth of intents.

    Returns a summary dict.
    """

    cfg = cfg or ExecutionConfig()
    rolling = _read_rolling() or {}
    if not isinstance(rolling, dict) or not rolling:
        log("[dt_exec] ⚠️ rolling empty; nothing to execute")
        return {"status": "empty", "orders": 0, "dry_run": cfg.dry_run}

    broker = BrokerAPI()

    # Replay/backtest can drive time via DT_NOW_UTC; fall back to real time.
    ts_now = now_utc or _now_utc_override()

    # Phase 5: manage exits (synthetic brackets, time-stops, scratch, EOD flatten)
    exit_summary = process_exits(
        rolling=rolling,
        broker=broker,
        dry_run=bool(cfg.dry_run),
        eod_flatten=bool(cfg.eod_flatten),
        eod_flatten_minutes=int(cfg.eod_flatten_minutes),
        now_utc=now_utc,
    )
    try:
        if exit_summary.exits_sent or exit_summary.partials_sent or exit_summary.eod_flattens:
            log(
                f"[dt_exec] 🧯 exits: eval={exit_summary.evaluated} "
                f"partials={exit_summary.partials_sent} exits={exit_summary.exits_sent} eod={exit_summary.eod_flattens}"
            )
    except Exception:
        pass

    orders = 0
    considered = 0
    blocked = 0

    # Iterate symbols deterministically.
    for sym in sorted([s for s in rolling.keys() if isinstance(s, str) and not s.startswith("_")]):
        if orders >= max(0, int(cfg.max_orders_per_cycle)):
            break
        node = rolling.get(sym)
        if not isinstance(node, dict):
            continue

        side, size, conf = _extract_intent(node)
        considered += 1

        # Nothing to do.
        if side == "FLAT" or size <= 0.0:
            continue

        if conf < float(cfg.min_confidence):
            blocked += 1
            append_trade_event(
                {
                    "type": "no_trade",
                    "symbol": sym,
                    "reason": f"conf<{cfg.min_confidence}",
                    "side": side,
                    "confidence": conf,
                    "size": size,
                }
            )
            continue

        positions_now = broker.get_positions()
        pos = positions_now.get(sym) if isinstance(positions_now, dict) else None

        # Anti-flip hysteresis: avoid rapid direction changes after an exit.
        try:
            last_exit_ts, last_side = recent_exit_info(sym)
            if last_exit_ts is not None and last_side and last_side in {"BUY", "SELL"}:
                if side in {"BUY", "SELL"} and side != last_side:
                    age_min = (ts_now - last_exit_ts).total_seconds() / 60.0
                    if age_min < float(cfg.min_flip_minutes):
                        blocked += 1
                        append_trade_event(
                            {
                                "type": "no_trade",
                                "symbol": sym,
                                "reason": f"flip_cooldown<{cfg.min_flip_minutes}m",
                                "side": side,
                                "confidence": conf,
                                "size": size,
                            }
                        )
                        continue
        except Exception:
            pass

        # Don't stack entries in the same direction (keeps it sane).
        try:
            if side == "BUY" and pos is not None and getattr(pos, "qty", 0.0) > 0:
                blocked += 1
                append_trade_event({"type": "no_trade", "symbol": sym, "reason": "already_long", "side": side, "confidence": conf, "size": size})
                continue
        except Exception:
            pass

        # Phase 0 only closes longs on SELL unless allow_shorts is enabled.
        if side == "SELL" and (pos is None or pos.qty <= 0.0) and not cfg.allow_shorts:
            blocked += 1
            append_trade_event(
                {
                    "type": "no_trade",
                    "symbol": sym,
                    "reason": "sell_without_position",
                    "side": side,
                    "confidence": conf,
                    "size": size,
                }
            )
            continue

        qty = _qty_from_size(size, cfg)
        if qty <= 0.0:
            continue

        append_trade_event(
            {
                "type": "intent",
                "symbol": sym,
                "side": side,
                "qty": qty,
                "confidence": conf,
                "size": size,
                "dry_run": bool(cfg.dry_run),
                "bot": (node.get("execution_dt") or {}).get("bot") if isinstance(node.get("execution_dt"), dict) else None,
                "risk": (node.get("execution_dt") or {}).get("risk") if isinstance(node.get("execution_dt"), dict) else None,
            }
        )

        if cfg.dry_run:
            orders += 1
            bump_metric("dry_run_intents", 1.0)
            continue

        try:
            last_px = _extract_last_price(node)
            atr = _extract_atr(node)
            order = Order(symbol=sym, side=side, qty=float(qty))
            res = broker.submit_order(order, last_price=(last_px if last_px > 0 else None))
            orders += 1
            bump_metric("orders_submitted", 1.0)
            append_trade_event(
                {
                    "type": "order_result",
                    "symbol": sym,
                    "side": side,
                    "qty": qty,
                    "result": res,
                }
            )

            # Phase 5: on entry fills, record synthetic bracket state.
            try:
                if bool(cfg.enable_brackets) and isinstance(res, dict) and str(res.get("status") or "").lower() == "filled":
                    filled_qty = _safe_float(res.get("qty"), 0.0)
                    fill_price = _safe_float(res.get("price"), 0.0)

                    # Entry: BUY always means long entry. SELL could be exit (if we had a position) or short entry.
                    is_entry = side == "BUY" or (side == "SELL" and cfg.allow_shorts and (pos is None or getattr(pos, "qty", 0.0) <= 0))

                    if is_entry and filled_qty > 0 and fill_price > 0:
                        risk = _plan_risk(node, side=side, last_price=fill_price, atr=atr, cfg=cfg)
                        plan = node.get("execution_plan_dt") if isinstance(node.get("execution_plan_dt"), dict) else {}
                        bot = str(plan.get("bot") or "") if isinstance(plan, dict) else ""
                        reason = str(plan.get("reason") or "") if isinstance(plan, dict) else ""
                        meta = _entry_meta_from_global(rolling, now_utc)
                        try:
                            # Phase 7 calibration wants the raw entry confidence at time of decision.
                            meta["base_conf"] = float(conf)
                            meta["confidence"] = float(conf)
                            # Also stash calibrated prob if the policy provided it.
                            p = node.get("policy_dt") if isinstance(node.get("policy_dt"), dict) else {}
                            if isinstance(p, dict) and p.get("p_hit") is not None:
                                meta["p_hit"] = float(p.get("p_hit") or 0.0)
                            # Phase 9 researcher wants a few entry-time features for safe filter proposals.
                            feats = node.get("features_dt") if isinstance(node.get("features_dt"), dict) else {}
                            meta["entry_features"] = {
                                "rel_volume": float(feats.get("rel_volume") or 0.0),
                                "atr": float(feats.get("atr") or 0.0),
                                "vwap_dev": float(feats.get("vwap_dev") or 0.0),
                            }
                            if bot:
                                meta["bot"] = str(bot).upper()
                        except Exception:
                            pass

                        record_entry(
                            symbol=sym,
                            side=side,
                            qty=float(filled_qty),
                            entry_price=float(fill_price),
                            risk=risk,
                            bot=(bot or None),
                            reason=(reason or None),
                            trail_atr_mult=float(cfg.trail_atr_mult),
                            scratch_min=int(cfg.scratch_min),
                            scratch_atr_frac=float(cfg.scratch_atr_frac),
                            now_utc=now_utc,
                            meta=meta,
                            confidence=float(conf),
                        )
                    else:
                        # If this SELL likely closed a long, mark exit in state.
                        if side == "SELL" and (pos is not None and getattr(pos, "qty", 0.0) > 0) and filled_qty > 0:
                            record_exit(sym, reason="manual_sell", now_utc=now_utc)
            except Exception:
                pass
        except Exception as e:
            blocked += 1
            bump_metric("order_errors", 1.0)
            append_trade_event(
                {
                    "type": "order_error",
                    "symbol": sym,
                    "side": side,
                    "qty": qty,
                    "error": str(e),
                }
            )

    out = {
        "status": "ok",
        "considered": int(considered),
        "orders": int(orders),
        "blocked": int(blocked),
        "dry_run": bool(cfg.dry_run),
        "exits": {
            "evaluated": int(exit_summary.evaluated),
            "partials_sent": int(exit_summary.partials_sent),
            "exits_sent": int(exit_summary.exits_sent),
            "eod_flattens": int(exit_summary.eod_flattens),
        },
    }
    log(f"[dt_exec] ✅ execute_from_policy done: {out}")
    return out
