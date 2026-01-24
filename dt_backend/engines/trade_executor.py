# dt_backend/engines/trade_executor.py — v0.3 (Phase 0 executor + LANE-AWARE + LIQUIDATION DEDUP)
"""Intraday trade executor for dt_backend.

Phase 0 goals
-------------
* Deterministic: every attempted action is logged to dt_trades.jsonl.
* Safe-by-default: DRY RUN unless explicitly enabled.
* Minimal assumptions: consumes execution_dt when present, otherwise policy_dt.

v0.2 updates
------------
- Lane-aware: optional `symbols=[...]` to execute only a scoped universe.
- Meta fixes: pull risk_mode/version from dt_state + daily_plan_dt (not a phantom g["dt_state"]).
- Feature key fixes: atr_14 / vwap_dist instead of atr / vwap_dev.
- Time consistency: use the effective cycle timestamp for exits + flip cooldown.

v0.3 updates
------------
- Liquidation de-dup + inflight tracking:
    * Prevents re-sending liquidation orders for the same symbols every cycle.
    * Tracks "inflight" liquidations in dt_state.json with TTL.
    * Optionally records accept/reject counts when broker returns status.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from dt_backend.core.data_pipeline_dt import _read_rolling, save_rolling
from dt_backend.core.logger_dt import log, debug
from dt_backend.core.time_override_dt import now_utc as _now_utc_override
from dt_backend.core.constants_dt import HOLD_MIN_TIME_MINUTES, POSITION_MAX_FRACTION
from dt_backend.services.dt_truth_store import append_trade_event, bump_metric
from dt_backend.services import execution_ledger
from dt_backend.core.constants_dt import (
    ORDERS_MAX_PER_CYCLE,
    TRADE_GAP_MIN_MINUTES,
    ORDER_TIMEOUT_SEC,
    HOLD_MIN_TIME_MINUTES,
    CONFIDENCE_MIN_EXEC,
    POSITION_DEFAULT_QTY,
)

# Optional: dt_state tags (safe if unavailable)
try:
    from dt_backend.services.dt_truth_store import read_dt_state, update_dt_state  # type: ignore
except Exception:  # pragma: no cover
    read_dt_state = None  # type: ignore
    update_dt_state = None  # type: ignore

from dt_backend.engines.broker_api import BrokerAPI, Order, get_positions_scoped
from dt_backend.services.position_manager_dt import (
    process_exits,
    record_entry,
    record_exit,
    recent_exit_info,
)
from dt_backend.utils.trading_utils_dt import sort_by_ranking_metric

# Slack alerting for visibility
try:
    from backend.monitoring.alerting import alert_dt, alert_error
except ImportError:
    alert_dt = None  # type: ignore
    alert_error = None  # type: ignore

# Slack-aware logger for error forwarding
try:
    from backend.monitoring.log_aggregator import get_aggregator
except ImportError:
    get_aggregator = None  # type: ignore

# Decision recorder for replay/analysis
try:
    from dt_backend.services.decision_recorder import DecisionRecorder
except ImportError:
    DecisionRecorder = None  # type: ignore

# Feature importance tracking for ML interpretability
try:
    from dt_backend.ml.feature_importance_tracker import get_tracker as get_feature_tracker
except ImportError:
    get_feature_tracker = None  # type: ignore


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
    max_orders_per_cycle: int = ORDERS_MAX_PER_CYCLE
    allow_shorts: bool = False
    min_confidence: float = CONFIDENCE_MIN_EXEC
    default_qty: float = POSITION_DEFAULT_QTY

    # Phase 5: synthetic brackets + state machine
    enable_brackets: bool = True
    eod_flatten: bool = True
    eod_flatten_minutes: int = 5

    # Anti-flip hysteresis (direction changes) after an exit
    min_flip_minutes: int = 12
    
    # Minimum hold time before any exit is considered (prevents fast BUY->SELL flips)
    min_hold_time_minutes: int = 10
    
    # Hard stop loss threshold (percentage) - exits allowed before min_hold_time if loss exceeds this
    hard_stop_loss_pct: float = 2.0

    # Fallback risk if a plan doesn't specify stop/tp
    fallback_stop_atr: float = 1.25
    fallback_tp_atr: float = 1.75

    # Position management defaults (can be overridden per-plan)
    trail_atr_mult: float = 1.2
    scratch_min: int = 12
    scratch_atr_frac: float = 0.15
    
    # Bug Fix #1: Minimum hold time
    min_hold_time_minutes: int = HOLD_MIN_TIME_MINUTES
    
    # Bug Fix #2: Position sizing parameters
    min_phit: float = 0.50  # Minimum P(Hit) to consider trading
    max_symbol_fraction: float = 0.15  # Maximum fraction of portfolio per symbol


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


def _env_bool(name: str, default: str = "0") -> bool:
    v = _env(name, default)
    return v.lower() in {"1", "true", "yes", "y", "on"}


def _cfg_from_env() -> ExecutionConfig:
    cfg = ExecutionConfig()
    cfg.dry_run = _env_bool("DT_DRY_RUN", "1")
    cfg.max_orders_per_cycle = int(_safe_float(_env("DT_MAX_ORDERS_PER_CYCLE", str(cfg.max_orders_per_cycle)), cfg.max_orders_per_cycle))
    cfg.allow_shorts = _env_bool("DT_ALLOW_SHORTS", "0")
    # allow both legacy and new knob names
    mc = _env("DT_EXEC_MIN_CONF", _env("DT_MIN_CONFIDENCE", str(cfg.min_confidence)))
    cfg.min_confidence = float(_safe_float(mc, cfg.min_confidence))
    dq = _env("DT_EXEC_DEFAULT_QTY", str(cfg.default_qty))
    cfg.default_qty = float(_safe_float(dq, cfg.default_qty))
    # minimum hold time before exit
    mht = _env("DT_MIN_HOLD_TIME_MINUTES", str(cfg.min_hold_time_minutes))
    cfg.min_hold_time_minutes = int(_safe_float(mht, cfg.min_hold_time_minutes))
    # hard stop loss percentage
    hsl = _env("DT_HARD_STOP_LOSS_PCT", str(cfg.hard_stop_loss_pct))
    cfg.hard_stop_loss_pct = float(_safe_float(hsl, cfg.hard_stop_loss_pct))
    return cfg


# ---------------------------------------------------------------------------
# Liquidation (operator lever)
# ---------------------------------------------------------------------------


def _liquidation_settings() -> Dict[str, Any]:
    return {
        "enabled": _env_bool("DT_LIQUIDATE_ENABLED", "0"),
        "target_positions": int(_safe_float(_env("DT_LIQUIDATE_TARGET_POSITIONS", "-1"), -1)),
        "scope": _env("DT_LIQUIDATE_SCOPE", "ALL").upper(),  # ACTIVE|CARRY|ALL
        "max_orders": int(_safe_float(_env("DT_LIQUIDATE_MAX_ORDERS_PER_CYCLE", "25"), 25)),
        # de-dup / inflight tracking
        "dedup": _env_bool("DT_LIQUIDATE_DEDUP_ENABLED", "1"),
        "ttl_sec": int(_safe_float(_env("DT_LIQUIDATE_INFLIGHT_TTL_SEC", "180"), 180)),
    }


def _parse_utc_iso(s: Any) -> Optional[datetime]:
    if not s:
        return None
    try:
        txt = str(s).strip()
        if not txt:
            return None
        # Accept both Z and +00:00
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        return datetime.fromisoformat(txt).astimezone(timezone.utc)
    except Exception:
        return None


def _load_inflight() -> Dict[str, Dict[str, Any]]:
    """Load liquidation inflight map from dt_state.json.

    Shape:
        {
          "AAPL": {"ts": "2026-01-08T16:41:20Z", "side": "SELL", "qty": 10.0},
          ...
        }
    """
    if read_dt_state is None:
        return {}
    try:
        st = read_dt_state() or {}
        m = st.get("liquidation_inflight")
        return m if isinstance(m, dict) else {}
    except Exception:
        return {}


def _save_inflight(m: Dict[str, Dict[str, Any]]) -> None:
    if update_dt_state is None:
        return
    try:
        update_dt_state({"liquidation_inflight": m})
    except Exception:
        pass


def _prune_inflight(m: Dict[str, Dict[str, Any]], *, now_utc: datetime, ttl_sec: int) -> Dict[str, Dict[str, Any]]:
    if ttl_sec <= 0:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for sym, v in (m or {}).items():
        if not isinstance(sym, str) or not isinstance(v, dict):
            continue
        ts = _parse_utc_iso(v.get("ts"))
        if ts is None:
            continue
        age = (now_utc - ts).total_seconds()
        if age <= float(ttl_sec):
            out[sym.upper()] = v
    return out


def _run_liquidation_if_enabled(broker: BrokerAPI, cfg: ExecutionConfig, *, now_utc: datetime) -> Dict[str, Any]:
    """If liquidation is enabled, sell down positions to target count.

    This runs even during stand_down. It is capped per cycle.

    v0.3: de-dups by tracking inflight symbols in dt_state.json with TTL so we
    don't resend the same liquidation orders every cycle.
    """
    s = _liquidation_settings()
    if not s["enabled"]:
        return {"status": "disabled"}

    scope = str(s["scope"] or "ALL").upper()
    target = int(s["target_positions"])
    max_orders = int(s["max_orders"])
    dedup = bool(s.get("dedup"))
    ttl_sec = int(s.get("ttl_sec") or 180)

    log(f"[dt_exec] 🚨 Liquidation mode enabled: scope={scope} target={target} max={max_orders}")

    inflight = _load_inflight() if dedup else {}
    inflight = _prune_inflight(inflight, now_utc=now_utc, ttl_sec=ttl_sec) if dedup else {}

    # Pull positions from the requested scope (ledger view).
    pos = get_positions_scoped(scope)
    syms = sorted([k for k, p in pos.items() if isinstance(k, str) and float(getattr(p, "qty", 0.0) or 0.0) != 0.0])

    keep = set()
    if target >= 0:
        keep = set(syms[:target])

    to_sell = [sym for sym in syms if sym not in keep]

    sent = 0
    skipped_inflight = 0
    accepted = 0
    rejected = 0
    errors = 0

    for sym in to_sell:
        if sent >= max_orders:
            break

        sym_u = str(sym).upper().strip()
        p = pos.get(sym)
        qty = float(getattr(p, "qty", 0.0)) if p else 0.0
        if qty == 0.0:
            continue

        # De-dup: if we already fired a liquidation order recently for this symbol, skip.
        if dedup and sym_u in inflight:
            skipped_inflight += 1
            continue

        side = "SELL" if qty > 0 else "BUY"  # cover shorts if any
        order = Order(symbol=sym_u, side=side, qty=abs(qty), limit_price=None)

        log(f"[dt_exec] 🚨 Liquidating position: {sym_u} {side} {abs(qty)}")
        append_trade_event(
            {
                "type": "liquidate_intent",
                "symbol": sym_u,
                "side": side,
                "qty": abs(qty),
                "scope": scope,
                "target_positions": target,
                "dedup": dedup,
                "ttl_sec": ttl_sec,
            }
        )

        # Mark inflight BEFORE submit so we don't spam duplicates even if submit is slow.
        if dedup:
            inflight[sym_u] = {
                "ts": now_utc.isoformat(timespec="seconds").replace("+00:00", "Z"),
                "side": side,
                "qty": float(abs(qty)),
            }
            _save_inflight(inflight)

        if cfg.dry_run:
            sent += 1
            continue

        try:
            res = broker.submit_order(order, last_price=None)

            # Best-effort status accounting
            st = ""
            try:
                st = str(res.get("status") or res.get("state") or "").lower() if isinstance(res, dict) else ""
            except Exception:
                st = ""

            if st in {"accepted", "new", "partially_filled", "filled", "pending_new", "submitted"}:
                accepted += 1
            elif st:
                rejected += 1

            append_trade_event(
                {
                    "type": "liquidate_order",
                    "symbol": sym_u,
                    "side": side,
                    "qty": abs(qty),
                    "result": res,
                    "scope": scope,
                }
            )

            # If broker clearly rejected, remove inflight so it can retry (or operator can investigate)
            if dedup and st and st in {"rejected", "canceled", "cancelled", "expired"}:
                inflight.pop(sym_u, None)
                _save_inflight(inflight)

            sent += 1
        except Exception as e:
            errors += 1
            log(f"[dt_exec] ⚠️ Liquidation error for {sym_u}: {e}", level="error")
            if get_aggregator is not None:
                get_aggregator().forward_log("ERROR", f"Liquidation error for {sym_u}: {e}", "dt_exec")
            append_trade_event(
                {
                    "type": "liquidate_error",
                    "symbol": sym_u,
                    "side": side,
                    "qty": abs(qty),
                    "error": str(e),
                    "scope": scope,
                }
            )
            # Don't keep it inflight if submission threw.
            if dedup:
                inflight.pop(sym_u, None)
                _save_inflight(inflight)

    # Final prune/save (keeps state bounded)
    if dedup:
        inflight = _prune_inflight(inflight, now_utc=now_utc, ttl_sec=ttl_sec)
        _save_inflight(inflight)

    summary = {
        "status": "ok",
        "enabled": True,
        "scope": scope,
        "target_positions": target,
        "positions_seen": len(syms),
        "orders_sent": sent,
        "orders_cap": max_orders,
        "dedup": dedup,
        "inflight_ttl_sec": ttl_sec,
        "inflight_now": len(inflight) if dedup else 0,
        "skipped_inflight": skipped_inflight,
        "accepted": accepted,
        "rejected": rejected,
        "errors": errors,
    }
    
    if sent > 0:
        log(f"[dt_exec] 🚨 Liquidation complete: sent={sent} accepted={accepted} rejected={rejected} errors={errors}")
    
    return summary


def _extract_intent(node: Dict[str, Any]) -> Tuple[str, float, float]:
    """Return (side, size, confidence).

    Prefers execution_dt (already sized + cooldown applied). Falls back to policy_dt.
    """

    ex = node.get("execution_dt")
    if isinstance(ex, dict):
        side = str(ex.get("side") or "").upper()
        size = _safe_float(ex.get("size"), 0.0)
        # Prefer p_hit if present; otherwise confidence_adj.
        conf = _safe_float(ex.get("p_hit"), _safe_float(ex.get("confidence_adj"), 0.0))
        if side in {"BUY", "SELL", "FLAT"}:
            return side, size, conf

    pol = node.get("policy_dt")
    if isinstance(pol, dict):
        side = str(pol.get("action") or pol.get("intent") or "").upper()
        conf = _safe_float(pol.get("p_hit"), _safe_float(pol.get("confidence"), 0.0))
        # policy_dt has no explicit size; treat confidence as a weak size proxy.
        size = max(0.0, min(1.0, conf))
        if side in {"BUY", "SELL", "HOLD", "STAND_DOWN"}:
            return ("FLAT" if side in {"HOLD", "STAND_DOWN"} else side), size, conf

    return "FLAT", 0.0, 0.0


def _extract_last_price(node: Dict[str, Any]) -> float:
    feat = node.get("features_dt")
    if isinstance(feat, dict):
        try:
            px = float(feat.get("last_price") or 0.0)
            if px > 0:
                return px
        except Exception:
            pass

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


# ---------------------------------------------------------------------------
# Bug Fix #1: Minimum hold time enforcement
# ---------------------------------------------------------------------------


def _can_exit_position(node: Dict, entry_ts: str, cfg: ExecutionConfig) -> Tuple[bool, str]:
    """
    Check if position held long enough to exit.
    
    Returns: (can_exit: bool, reason: str)
    """
    if not entry_ts:
        return True, "no_entry_timestamp"
    
    try:
        entry_dt = datetime.fromisoformat(entry_ts.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        hold_minutes = (now - entry_dt).total_seconds() / 60.0
        
        if hold_minutes < cfg.min_hold_time_minutes:
            return False, f"held_{hold_minutes:.0f}m_<_{cfg.min_hold_time_minutes}m"
        
        return True, None
    except Exception as e:
        log(f"[dt_exec] ⚠️ Error checking hold time: {e}", level="error")
        if get_aggregator is not None:
            get_aggregator().forward_log("ERROR", f"Hold time check error: {e}", "dt_exec")
        return True, None


# ---------------------------------------------------------------------------
# Bug Fix #2: Conviction-based position sizing
# ---------------------------------------------------------------------------


def _size_from_phit_with_conviction(
    phit: float,
    expected_r: float,
    vol_bkt: str,
    position_qty: float = 0.0,
    cfg: ExecutionConfig = None,
) -> float:
    """
    Calculate position size based on conviction (P(Hit)) with position-aware scaling.
    
    Logic:
    - If no position: size based on P(Hit) conviction
    - If small position (<50% of target): scale up
    - If at target position: reduce size
    - If above target: reduce or reverse
    """
    cfg = cfg or ExecutionConfig()
    phit = _safe_float(phit, 0.5)
    
    # Check minimum P(Hit) threshold
    if phit < cfg.min_phit:
        return 0.0
    
    # Base edge from calibrated probability
    # P(Hit) = 0.52 → edge = 0.04
    # P(Hit) = 0.75 → edge = 0.50
    edge = max(0.0, min(1.0, (phit - 0.5) / 0.5))
    
    # Risk-reward scaling
    er = max(0.5, min(2.0, _safe_float(expected_r, 1.0)))
    r_factor = 0.5 + 0.5 * (er / 2.0)  # 0.5..1.0
    
    # Volatility scaling
    vol_scale = {
        "high": 0.4,     # High vol: reduce size
        "medium": 0.7,   # Medium vol: normal size
        "low": 1.0,      # Low vol: full size
    }.get(vol_bkt, 0.7)
    
    # Base size from conviction
    base_size = cfg.max_symbol_fraction * edge * r_factor * vol_scale
    
    # Position-aware scaling
    if position_qty > 0:
        # Calculate target position (full conviction)
        target_qty = cfg.max_symbol_fraction * POSITION_MAX_FRACTION
        current_fraction = position_qty / target_qty if target_qty > 0 else 0
        
        if current_fraction >= 0.9:
            # Already at/above target: reduce size
            base_size *= 0.3
        elif current_fraction >= 0.5:
            # Halfway to target: maintain pace
            base_size = base_size  # No change
        elif current_fraction >= 0.25:
            # Quarter to target: scale up moderately
            base_size *= 1.2
        else:
            # Well below target: scale up aggressively
            base_size *= 1.5
    
    return max(0.0, min(cfg.max_symbol_fraction, base_size))


def _entry_meta_from_global(rolling: Dict[str, Any], now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    """Pull lightweight tags from _GLOBAL_DT + dt_state for analytics/replay."""
    g = rolling.get("_GLOBAL_DT") if isinstance(rolling, dict) else None
    g = g if isinstance(g, dict) else {}

    rd = g.get("regime_dt") if isinstance(g.get("regime_dt"), dict) else {}
    mr = g.get("micro_regime_dt") if isinstance(g.get("micro_regime_dt"), dict) else {}
    plan = g.get("daily_plan_dt") if isinstance(g.get("daily_plan_dt"), dict) else {}

    # dt_state.json is the source of truth for version/risk switches
    st = {}
    try:
        if read_dt_state is not None:
            st = read_dt_state() or {}
    except Exception:
        st = {}

    out = {
        "regime": rd.get("label"),
        "day_type": rd.get("day_type"),
        "regime_conf": rd.get("confidence"),
        "micro": mr.get("label"),
        "time_window": mr.get("time_window"),
        "risk_mode": (plan.get("risk_mode") or st.get("risk_mode")),
        "version": st.get("version"),
        "lane": st.get("lane"),
        "cycle_seq": st.get("cycle_seq"),
        "cycle_id": st.get("cycle_id"),
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
    size = max(0.0, min(1.0, float(size)))
    return max(0.0, float(cfg.default_qty) * size)


def execute_from_policy(
    cfg: Optional[ExecutionConfig] = None,
    *,
    now_utc: Optional[datetime] = None,
    symbols: Optional[List[str]] = None,
    max_symbols: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute one cycle worth of intents.

    Args:
        cfg: ExecutionConfig (dry run by default).
        now_utc: optional forced time (replay/backtest).
        symbols: optional explicit universe (lane-aware). If provided, only these symbols are considered.
        max_symbols: optional cap applied after symbol filtering.

    Returns a summary dict.
    """
    cfg = _cfg_from_env() if cfg is None else cfg
    rolling = _read_rolling() or {}
    
    # Build initial symbol list for cycle start logging
    if not isinstance(rolling, dict) or not rolling:
        log("[dt_exec] ⚠️ rolling empty; nothing to execute")
        return {"status": "empty", "orders": 0, "dry_run": cfg.dry_run}
    
    initial_symbols = [s for s in rolling.keys() if isinstance(s, str) and not s.startswith("_")]
    log(f"[dt_exec] 🚀 Starting execution cycle: symbols={len(initial_symbols)}")
    
    # Validate rolling structure before execution (PR #4)
    try:
        from dt_backend.core.schema_validator_dt import validate_rolling, ValidationError
        validate_rolling(rolling)
        log("[dt_exec] ✅ Rolling validation passed")
    except ValidationError as e:
        log(f"[dt_exec] ❌ Validation error: {e}", level="error")
        if get_aggregator is not None:
            get_aggregator().forward_log("ERROR", f"Validation error: {e}", "dt_exec")
        return {"error": str(e), "trades": [], "orders": 0}
    except Exception as e:
        log(f"[dt_exec] ⚠️ Validation exception: {e}", level="error")
        if get_aggregator is not None:
            get_aggregator().forward_log("ERROR", f"Validation exception: {e}", "dt_exec")
        # Continue execution despite validation errors

    # Initialize decision recorder for this cycle
    recorder = None
    cycle_id = None
    try:
        if DecisionRecorder is not None:
            recorder = DecisionRecorder()
            cycle_id = recorder.start_cycle()
            debug(f"[dt_exec] 📝 Decision recording started for cycle {cycle_id}")
    except Exception as e:
        log(f"[dt_exec] ⚠️ Failed to initialize decision recorder: {e}")

    broker = BrokerAPI()

    # Replay/backtest can drive time via DT_NOW_UTC; fall back to real time.
    ts_now = now_utc or _now_utc_override()

    # Emergency / operator-controlled liquidation (sell down to target positions).
    liquidation_summary = _run_liquidation_if_enabled(broker, cfg, now_utc=ts_now)

    # Phase 5: manage exits (synthetic brackets, time-stops, scratch, EOD flatten)
    exit_summary = process_exits(
        rolling=rolling,
        broker=broker,
        dry_run=bool(cfg.dry_run),
        eod_flatten=bool(cfg.eod_flatten),
        eod_flatten_minutes=int(cfg.eod_flatten_minutes),
        now_utc=ts_now,
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

    # Build deterministic symbol list
    if isinstance(symbols, list) and symbols:
        wanted = {str(s).strip().upper() for s in symbols if str(s).strip()}
        sym_list = [s for s in rolling.keys() if isinstance(s, str) and not s.startswith("_") and s.upper() in wanted]
    else:
        sym_list = [s for s in rolling.keys() if isinstance(s, str) and not s.startswith("_")]

    # CRITICAL: Sort by signal strength + confidence, NOT alphabetically
    # Alphabetical sorting causes "A" ticker bias (AAPL/AMD always first)
    # Human day traders prioritize highest-conviction setups, not alphabet order
    sym_list = list(set([str(s).upper() for s in sym_list]))
    sym_list = sort_by_ranking_metric(sym_list, rolling)
    
    if max_symbols is not None:
        sym_list = sym_list[: max(0, int(max_symbols))]

    # Record symbol selection decision
    try:
        if recorder is not None and sym_list:
            # Build ranking dict for selected symbols
            ranking = {}
            for sym in sym_list:
                node = rolling.get(sym)
                if isinstance(node, dict):
                    _, _, conf = _extract_intent(node)
                    ranking[sym] = float(conf)
            
            recorder.record_symbol_selection(
                selected_symbols=sym_list[:max_symbols] if max_symbols else sym_list,
                ranking=ranking,
                max_symbols=max_symbols,
                total_candidates=len(sym_list),
            )
    except Exception as e:
        log(f"[dt_exec] ⚠️ Failed to record symbol selection: {e}")

    # Iterate symbols by ranking (highest conviction first).
    for sym in sym_list:
        if orders >= max(0, int(cfg.max_orders_per_cycle)):
            break
        node = rolling.get(sym)
        if not isinstance(node, dict):
            continue

        side, size, conf = _extract_intent(node)
        considered += 1
        
        # Log intent for this symbol
        if side != "FLAT" and size > 0.0:
            debug(f"[dt_exec] 📊 {sym}: intent={side} size={size:.3f} conf={conf:.2%}")
        
        # Log feature importance for this trading decision
        try:
            if get_feature_tracker is not None:
                feats = node.get("features_dt", {})
                if isinstance(feats, dict) and feats:
                    tracker = get_feature_tracker()
                    tracker.log_prediction(
                        symbol=sym,
                        features_dict=feats,
                        prediction=side,
                        confidence=conf,
                        metadata={"cycle": "execution"}
                    )
        except Exception as e:
            debug(f"[dt_exec] Feature importance logging failed for {sym}: {e}")

        # Nothing to do.
        if side == "FLAT" or size <= 0.0:
            continue

        if conf < float(cfg.min_confidence):
            blocked += 1
            debug(f"[dt_exec] ⚠️ Risk check failed: {sym} conf={conf:.2%} < min={cfg.min_confidence:.2%}")
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
            
            # Track missed opportunity if confidence was high
            try:
                if conf >= 0.60:
                    from dt_backend.ml.missed_opportunity_tracker import track_missed_signal
                    
                    signal = {
                        "symbol": sym,
                        "label": side,
                        "confidence": conf,
                        "price": node.get("bars_1m", {}).get("close", 0.0),
                        "timestamp": _now_utc_override().isoformat(),
                        "lgb_prob": conf,
                    }
                    track_missed_signal(signal, f"conf<{cfg.min_confidence}")
            except Exception as e:
                debug(f"[dt_exec] Failed to track missed signal for {sym}: {e}")
                pass
            
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
                        log(f"[dt_exec] 🔄 Trade gap enforcement for {sym}: cooldown {age_min:.1f}m < {cfg.min_flip_minutes}m")
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

        # ============================================================================
        # BUG FIX #1: Minimum Hold Time Check
        # ============================================================================
        # Prevent 2-minute buy/sell flips by enforcing minimum hold time
        if side == "SELL":
            try:
                # Get position entry timestamp
                position_dt = node.get("position_dt", {})
                entry_ts = position_dt.get("entry_ts") if isinstance(position_dt, dict) else None
                
                # If no entry_ts in position_dt, try to get it from position manager
                if not entry_ts and pos is not None:
                    try:
                        entry_ts = getattr(pos, "entry_ts", None)
                    except Exception:
                        pass
                
                if entry_ts:
                    can_exit, reason = _can_exit_position(
                        node=node,
                        entry_ts=entry_ts,
                        cfg=cfg
                    )
                    
                    if not can_exit:
                        blocked += 1
                        log(f"[dt_exec] ⏱️ Hold time check for {sym}: {reason}")
                        append_trade_event(
                            {
                                "type": "no_trade",
                                "symbol": sym,
                                "reason": f"min_hold_time: {reason}",
                                "side": side,
                                "confidence": conf,
                                "size": size,
                            }
                        )
                        log(f"[exec] ⏸ {sym}: {reason} - holding position")
                        continue
            except Exception as e:
                log(f"[exec] ⚠️ Error checking hold time for {sym}: {e}", level="error")
                if get_aggregator is not None:
                    get_aggregator().forward_log("ERROR", f"Hold time validation error for {sym}: {e}", "dt_exec")

        # ============================================================================
        # INTELLIGENT POSITION HOLDING (Human Day Trader Logic)
        # ============================================================================
        # If we have an open position and signal says SELL, apply hold strategy
        # instead of immediately exiting. This prevents mechanical "buy cycle N, sell cycle N+1"
        # behavior and lets winning positions run.
        try:
            if pos is not None and getattr(pos, "qty", 0.0) > 0:  # We have a LONG position
                if side == "SELL":
                    # FIRST: Check minimum hold time to prevent fast flips
                    # Don't exit position within first N minutes unless hard stop is hit
                    from dt_backend.services.position_manager_dt import read_positions_state
                    pos_state = read_positions_state()
                    ps = pos_state.get(sym.upper(), {})
                    
                    if isinstance(ps, dict):
                        entry_ts = _parse_utc_iso(ps.get("entry_ts"))
                        if entry_ts is not None:
                            hold_minutes = (ts_now - entry_ts).total_seconds() / 60.0
                            
                            # If held less than minimum hold time, block exit (unless hard stop)
                            if hold_minutes < cfg.min_hold_time_minutes:
                                # Check if this is a hard stop scenario (large loss)
                                entry_price = float(getattr(pos, "avg_price", 0.0))
                                last_price = _extract_last_price(node)
                                pnl_pct = 0.0
                                if entry_price > 0 and last_price > 0:
                                    pnl_pct = ((last_price - entry_price) / entry_price) * 100.0
                                
                                # Only allow exit if hard stop hit (loss exceeds configured threshold)
                                # Otherwise enforce minimum hold time
                                if pnl_pct > -cfg.hard_stop_loss_pct:
                                    blocked += 1
                                    
                                    # Update position hold tracking
                                    try:
                                        from dt_backend.services.position_manager_dt import update_position_hold_info
                                        update_position_hold_info(
                                            sym,
                                            hold_reason="min_hold_time_not_met",
                                            current_pnl_pct=pnl_pct,
                                            now_utc=ts_now,
                                        )
                                    except Exception as e:
                                        log(f"[dt_exec] ⚠️ Error updating position hold info for {sym}: {e}")
                                    
                                    append_trade_event({
                                        "type": "position_hold",
                                        "symbol": sym,
                                        "reason": "min_hold_time_not_met",
                                        "hold_minutes": hold_minutes,
                                        "min_hold_minutes": cfg.min_hold_time_minutes,
                                        "pnl_pct": pnl_pct,
                                        "hold_strategy": "enforce_min_hold",
                                    })
                                    continue
                    
                    # Check if BUY signal is still active (different from execution intent)
                    # Look at raw policy_dt for underlying signal strength
                    policy = node.get("policy_dt", {})
                    if isinstance(policy, dict):
                        raw_intent = str(policy.get("action") or "").upper()
                        raw_conf = _safe_float(policy.get("confidence"), 0.0)
                        
                        # If BUY signal still active with confidence, HOLD position
                        if raw_intent == "BUY" and raw_conf >= float(cfg.min_confidence):
                            blocked += 1
                            
                            # Calculate current PnL
                            entry_price = float(getattr(pos, "avg_price", 0.0))
                            last_price = _extract_last_price(node)
                            pnl_pct = 0.0
                            if entry_price > 0 and last_price > 0:
                                pnl_pct = ((last_price - entry_price) / entry_price) * 100.0
                            
                            # Update position hold tracking
                            try:
                                from dt_backend.services.position_manager_dt import update_position_hold_info
                                update_position_hold_info(
                                    sym,
                                    hold_reason="buy_signal_still_active",
                                    current_pnl_pct=pnl_pct,
                                    now_utc=ts_now,
                                )
                            except Exception:
                                pass
                            
                            append_trade_event({
                                "type": "position_hold",
                                "symbol": sym,
                                "reason": "buy_signal_still_active",
                                "raw_intent": raw_intent,
                                "raw_confidence": raw_conf,
                                "execution_side": side,
                                "hold_strategy": "signal_active",
                                "pnl_pct": pnl_pct,
                            })
                            continue
                        
                        # Check profit/loss for intelligent exit decisions
                        try:
                            entry_price = float(getattr(pos, "avg_price", 0.0))
                            last_price = _extract_last_price(node)
                            
                            if entry_price > 0 and last_price > 0:
                                pnl_pct = ((last_price - entry_price) / entry_price) * 100.0
                                
                                # Winning trade (>0.5% gain) - HOLD with trailing stop
                                # Let winners run, don't exit just because rank dropped
                                if pnl_pct > 0.5:
                                    blocked += 1
                                    
                                    # Update position hold tracking
                                    try:
                                        from dt_backend.services.position_manager_dt import update_position_hold_info
                                        update_position_hold_info(
                                            sym,
                                            hold_reason="winning_trade_let_run",
                                            current_pnl_pct=pnl_pct,
                                            now_utc=ts_now,
                                        )
                                    except Exception:
                                        pass
                                    
                                    append_trade_event({
                                        "type": "position_hold",
                                        "symbol": sym,
                                        "reason": "winning_trade_let_run",
                                        "pnl_pct": pnl_pct,
                                        "entry_price": entry_price,
                                        "current_price": last_price,
                                        "hold_strategy": "trail_winner",
                                    })
                                    # Trailing stop is handled by position_manager_dt.py
                                    continue
                                
                                # Breakeven or small loss (-1% to 0%) - wait for reversal
                                # Don't exit on noise, give it a chance to recover
                                if -1.0 <= pnl_pct <= 0.0:
                                    # Check time in position
                                    from dt_backend.services.position_manager_dt import read_positions_state
                                    pos_state = read_positions_state()
                                    ps = pos_state.get(sym.upper(), {})
                                    
                                    if isinstance(ps, dict):
                                        entry_ts = _parse_utc_iso(ps.get("entry_ts"))
                                        if entry_ts is not None:
                                            hold_minutes = (ts_now - entry_ts).total_seconds() / 60.0
                                            
                                            # If held < 15 minutes, give it more time
                                            if hold_minutes < 15.0:
                                                blocked += 1
                                                
                                                # Update position hold tracking
                                                try:
                                                    from dt_backend.services.position_manager_dt import update_position_hold_info
                                                    update_position_hold_info(
                                                        sym,
                                                        hold_reason="breakeven_wait_reversal",
                                                        current_pnl_pct=pnl_pct,
                                                        now_utc=ts_now,
                                                    )
                                                except Exception:
                                                    pass
                                                
                                                append_trade_event({
                                                    "type": "position_hold",
                                                    "symbol": sym,
                                                    "reason": "breakeven_wait_reversal",
                                                    "pnl_pct": pnl_pct,
                                                    "hold_minutes": hold_minutes,
                                                    "hold_strategy": "wait_reversal",
                                                })
                                                continue
                                
                                # Large loss (< -1%) - allow exit via stop loss from position_manager
                                # But also respect hard stops from position_manager_dt
                        except Exception as e:
                            # Log error but continue - don't block trading on PnL calculation issues
                            log(f"[dt_exec] ⚠️ Error calculating PnL for {sym}: {e}", level="error")
                            if get_aggregator is not None:
                                get_aggregator().forward_log("ERROR", f"PnL calculation error for {sym}: {e}", "dt_exec")
                            pass
        except Exception as e:
            # Log error but continue - don't block trading on position hold logic issues
            log(f"[dt_exec] ⚠️ Error in position hold logic for {sym}: {e}", level="error")
            if get_aggregator is not None:
                get_aggregator().forward_log("ERROR", f"Position hold logic error for {sym}: {e}", "dt_exec")
            pass

        # Don't stack entries in the same direction (keeps it sane).
        try:
            if side == "BUY" and pos is not None and getattr(pos, "qty", 0.0) > 0:
                blocked += 1
                debug(f"[dt_exec] 🚫 Position limit: {sym} already long, not stacking")
                append_trade_event({"type": "no_trade", "symbol": sym, "reason": "already_long", "side": side, "confidence": conf, "size": size})
                continue
        except Exception:
            pass

        # Phase 0 only closes longs on SELL unless allow_shorts is enabled.
        if side == "SELL" and (pos is None or getattr(pos, "qty", 0.0) <= 0.0) and not cfg.allow_shorts:
            blocked += 1
            debug(f"[dt_exec] 🚫 Risk check: {sym} SELL without position (shorts disabled)")
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

        # ============================================================================
        # BUG FIX #2: Conviction-Based Position Sizing
        # ============================================================================
        # Calculate qty based on conviction (P(Hit)) with position-aware scaling
        try:
            # Get current position qty for position-aware scaling
            current_qty = float(getattr(pos, "qty", 0.0)) if pos else 0.0
            
            # Extract policy features for conviction sizing
            policy = node.get("policy_dt", {})
            phit = _safe_float(policy.get("p_hit"), conf) if isinstance(policy, dict) else conf
            
            # Get expected R from execution plan or features
            plan = node.get("execution_plan_dt", {})
            features = node.get("features_dt", {})
            expected_r = 1.0
            if isinstance(plan, dict) and plan.get("risk"):
                risk = plan.get("risk")
                if isinstance(risk, dict):
                    stop = risk.get("stop")
                    tp = risk.get("take_profit")
                    last_px = _extract_last_price(node)
                    if stop and tp and last_px > 0:
                        r_dist = abs(tp - last_px)
                        stop_dist = abs(last_px - stop)
                        if stop_dist > 0:
                            expected_r = r_dist / stop_dist
            
            # Get volatility bucket
            vol_bkt = "medium"
            if isinstance(features, dict):
                atr = _safe_float(features.get("atr_14"), 0.0)
                last_px = _extract_last_price(node)
                if atr > 0 and last_px > 0:
                    atr_pct = (atr / last_px) * 100.0
                    if atr_pct > 3.0:
                        vol_bkt = "high"
                    elif atr_pct < 1.5:
                        vol_bkt = "low"
            
            # Calculate conviction-aware size (fraction of portfolio)
            size_fraction = _size_from_phit_with_conviction(
                phit=phit,
                expected_r=expected_r,
                vol_bkt=vol_bkt,
                position_qty=abs(current_qty),
                cfg=cfg
            )
            
            debug(f"[dt_exec] 🎯 Conviction sizing: phit={phit:.2%} → size={size_fraction:.3f}")
            
            # Convert size fraction to share quantity
            # Get account equity for position sizing
            last_px = _extract_last_price(node)
            if last_px > 0 and size_fraction > 0:
                # Estimate equity (can be improved with actual account value)
                equity = 100000.0  # Default assumption
                try:
                    account = broker.get_account()
                    if isinstance(account, dict):
                        equity = _safe_float(account.get("equity", 100000.0), 100000.0)
                except Exception as e:
                    debug(f"[dt_exec] Failed to get account equity for {sym}: {e}")
                    pass
                
                qty = max(1, int(size_fraction * equity / last_px))
            else:
                # Fallback to old method if something went wrong
                qty = _qty_from_size(size, cfg)
        except Exception as e:
            log(f"[exec] ⚠️ Error in conviction sizing for {sym}: {e}", level="error")
            if get_aggregator is not None:
                get_aggregator().forward_log("ERROR", f"Conviction sizing error for {sym}: {e}", "dt_exec")
            # Fallback to old method
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

        # NEW: Daily trade limit per symbol
        max_daily = int(os.getenv("DT_MAX_TRADES_PER_SYMBOL_PER_DAY", "2") or "2")
        if max_daily > 0:
            try:
                from dt_backend.services.dt_truth_store import count_trades_today
                trades_today = count_trades_today(sym)
                
                if trades_today >= max_daily:
                    blocked += 1
                    log(f"[dt_exec] 🛑 Emergency stop: {sym} reached daily trade limit ({trades_today}/{max_daily})")
                    plan = node.get("execution_plan_dt") if isinstance(node.get("execution_plan_dt"), dict) else {}
                    append_trade_event({
                        "type": "no_trade",
                        "symbol": sym,
                        "reason": f"max_daily_trades({trades_today}/{max_daily})",
                        "side": side,
                        "bot": plan.get("bot") if isinstance(plan, dict) else None,
                    })
                    continue  # Skip this symbol
            except Exception as e:
                debug(f"[dt_exec] Failed to check daily trade limit for {sym}: {e}")
                pass
        
        # NEW (Phase 3): Max loss per symbol per day check
        max_loss_per_symbol = _safe_float(os.getenv("DT_MAX_LOSS_PER_SYMBOL_DAY", "500.0"), 500.0)
        if max_loss_per_symbol > 0:
            try:
                from dt_backend.services.dt_truth_store import get_symbol_pnl_today
                symbol_pnl = get_symbol_pnl_today(sym)
                
                if symbol_pnl < -abs(max_loss_per_symbol):
                    blocked += 1
                    log(f"[dt_exec] 🛑 Emergency stop: {sym} max daily loss hit pnl=${symbol_pnl:.2f} limit=${max_loss_per_symbol:.2f}")
                    append_trade_event({
                        "type": "no_trade",
                        "symbol": sym,
                        "reason": f"max_daily_loss_per_symbol pnl={symbol_pnl:.2f} limit={max_loss_per_symbol:.2f}",
                        "side": side,
                        "confidence": conf,
                        "size": size,
                    })
                    continue  # Skip this symbol
            except Exception as e:
                debug(f"[dt_exec] Failed to check max loss per symbol for {sym}: {e}")
                pass

        try:
            last_px = _extract_last_price(node)
            atr = _extract_atr(node)
            order = Order(symbol=sym, side=side, qty=float(qty))
            
            # Initialize exec_id to None for error handling
            exec_id = None
            
            # ==========================================================================
            # SAGA PATTERN: 3-Phase Commit
            # ==========================================================================
            # Phase 1: Record pending BEFORE broker API call
            exec_id = execution_ledger.record_pending(
                symbol=sym,
                side=side,
                qty=qty,
                price=last_px if last_px > 0 else 0.0,
                bot=(node.get("execution_plan_dt") or {}).get("bot") if isinstance(node.get("execution_plan_dt"), dict) else None,
                confidence=conf,
                meta=_entry_meta_from_global(rolling, ts_now),
                now_utc=ts_now,
            )
            debug(f"[dt_exec] 📝 Phase 1 (pending): {exec_id} {side} {qty} {sym}")
            
            # Submit order to broker
            log(f"[dt_exec] 📤 Submitting order: {sym} {side} {qty} @ ${last_px:.2f}")
            res = broker.submit_order(order, last_price=(last_px if last_px > 0 else None))
            orders += 1
            bump_metric("orders_submitted", 1.0)
            log(f"[dt_exec] ✅ Order submitted: {sym} {side} {qty}")
            append_trade_event(
                {
                    "type": "order_result",
                    "symbol": sym,
                    "side": side,
                    "qty": qty,
                    "result": res,
                    "execution_id": exec_id,  # Link to ledger
                }
            )

            # Phase 5: on entry fills, record synthetic bracket state.
            try:
                if bool(cfg.enable_brackets) and isinstance(res, dict) and str(res.get("status") or "").lower() == "filled":
                    filled_qty = _safe_float(res.get("qty"), 0.0)
                    fill_price = _safe_float(res.get("price"), 0.0)

                    is_entry = side == "BUY" or (side == "SELL" and cfg.allow_shorts and (pos is None or getattr(pos, "qty", 0.0) <= 0))

                    if is_entry and filled_qty > 0 and fill_price > 0:
                        # Phase 2: Record confirmed AFTER broker confirms fill
                        broker_order_id = str(res.get("order_id") or res.get("id") or "unknown")
                        execution_ledger.record_confirmed(
                            execution_id=exec_id,
                            broker_order_id=broker_order_id,
                            fill_price=fill_price,
                            now_utc=ts_now,
                        )
                        debug(f"[dt_exec] ✅ Phase 2 (confirmed): {exec_id} filled @ {fill_price}")
                        
                        risk = _plan_risk(node, side=side, last_price=fill_price, atr=atr, cfg=cfg)
                        plan = node.get("execution_plan_dt") if isinstance(node.get("execution_plan_dt"), dict) else {}
                        bot = str(plan.get("bot") or "") if isinstance(plan, dict) else ""
                        reason = str(plan.get("reason") or "") if isinstance(plan, dict) else ""

                        meta = _entry_meta_from_global(rolling, ts_now)
                        try:
                            meta["base_conf"] = float(conf)
                            meta["confidence"] = float(conf)
                            p = node.get("policy_dt") if isinstance(node.get("policy_dt"), dict) else {}
                            if isinstance(p, dict) and p.get("p_hit") is not None:
                                meta["p_hit"] = float(p.get("p_hit") or 0.0)
                            
                            # Send Slack alert for position entry
                            if alert_dt is not None and not cfg.dry_run:
                                try:
                                    feats = node.get("features_dt", {})
                                    signal_strength = _safe_float(feats.get("signal_strength"), 0.0) if isinstance(feats, dict) else 0.0
                                    alert_dt(
                                        f"Position Opened: {sym}",
                                        f"{side} {filled_qty} shares @ ${fill_price:.2f}",
                                        level="info",
                                        context={
                                            "Bot": bot or "N/A",
                                            "Confidence": f"{conf:.1%}",
                                            "Signal Strength": f"{signal_strength:.3f}",
                                            "Stop": f"${risk.get('stop', 0.0):.2f}" if risk.get('stop') else "N/A",
                                            "Take Profit": f"${risk.get('take_profit', 0.0):.2f}" if risk.get('take_profit') else "N/A",
                                            "Reason": reason[:100] if reason else "N/A",
                                        }
                                    )
                                except Exception:
                                    pass

                            feats = node.get("features_dt") if isinstance(node.get("features_dt"), dict) else {}
                            meta["entry_features"] = {
                                "rel_volume": float(feats.get("rel_volume") or 0.0),
                                "atr_14": float(feats.get("atr_14") or 0.0),
                                "vwap_dist": float(feats.get("vwap_dist") or 0.0),
                                "squeeze_on": float(feats.get("squeeze_on") or 0.0),
                            }
                            if bot:
                                meta["bot"] = str(bot).upper()
                        except Exception:
                            pass

                        # Phase 3: Record atomic update (position + truth store)
                        # Note: record_entry now does atomic locking internally
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
                            now_utc=ts_now,
                            meta=meta,
                            confidence=float(conf),
                        )
                        
                        # Mark as recorded in saga ledger
                        position_snapshot = {
                            "symbol": sym,
                            "side": side,
                            "qty": filled_qty,
                            "entry_price": fill_price,
                            "ts": ts_now.isoformat(timespec="seconds").replace("+00:00", "Z"),
                        }
                        execution_ledger.record_recorded(
                            execution_id=exec_id,
                            position_state=position_snapshot,
                            now_utc=ts_now,
                        )
                        debug(f"[dt_exec] 🎉 Phase 3 (recorded): {exec_id} complete")

                        # Record entry decision for replay
                        try:
                            if recorder is not None:
                                recorder.record_entry(
                                    symbol=sym,
                                    side=side,
                                    qty=float(filled_qty),
                                    price=float(fill_price),
                                    reason=reason or "signal",
                                    confidence=float(conf),
                                    bot=bot,
                                    stop=risk.get("stop"),
                                    take_profit=risk.get("take_profit"),
                                )
                        except Exception as e:
                            log(f"[dt_exec] ⚠️ Failed to record entry decision: {e}")

                        # Update position_dt in rolling cache so policy sees the position
                        try:
                            # Determine signed qty: positive for LONG, negative for SHORT
                            if isinstance(node, dict):
                                position_qty = float(filled_qty) if side == "BUY" else -float(filled_qty)
                                position_side = "LONG" if side == "BUY" else "SHORT"

                                node["position_dt"] = {
                                    "qty": position_qty,
                                    "avg_price": float(fill_price),
                                    "side": position_side,
                                    "ts": ts_now.isoformat(timespec="seconds").replace("+00:00", "Z"),
                                }
                                rolling[sym] = node
                        except Exception:
                            pass
                        
                        # Mark symbol as recently acted upon
                        try:
                            from dt_backend.core.time_override_dt import now_utc
                            if isinstance(node, dict):
                                plan = node.get("execution_plan_dt") if isinstance(node.get("execution_plan_dt"), dict) else {}
                                node["_last_action_ts"] = now_utc().isoformat().replace("+00:00", "Z")
                                node["_last_action_bot"] = str(plan.get("bot") or "").upper() if isinstance(plan, dict) else ""
                                rolling[sym] = node
                        except Exception:
                            pass
                    else:
                        # Not an entry, mark as completed (exit)
                        if side == "SELL" and (pos is not None and getattr(pos, "qty", 0.0) > 0) and filled_qty > 0:
                            # Phase 2: Confirmed
                            broker_order_id = str(res.get("order_id") or res.get("id") or "unknown")
                            execution_ledger.record_confirmed(
                                execution_id=exec_id,
                                broker_order_id=broker_order_id,
                                fill_price=fill_price if fill_price > 0 else last_px,
                                now_utc=ts_now,
                            )
                            
                            record_exit(sym, reason="manual_sell", now_utc=ts_now)
                            
                            # Phase 3: Recorded
                            execution_ledger.record_recorded(
                                execution_id=exec_id,
                                position_state={"symbol": sym, "side": "FLAT", "qty": 0.0},
                                now_utc=ts_now,
                            )

                            # Clear position_dt after exit
                            try:
                                if isinstance(node, dict):
                                    node["position_dt"] = {
                                        "qty": 0.0,
                                        "avg_price": 0.0,
                                        "side": "FLAT",
                                        "ts": ts_now.isoformat(timespec="seconds").replace("+00:00", "Z"),
                                    }
                                    rolling[sym] = node
                            except Exception:
                                pass
                        else:
                            # Order filled but not an entry or exit we track, mark as recorded
                            execution_ledger.record_recorded(
                                execution_id=exec_id,
                                position_state={"info": "non-tracked fill"},
                                now_utc=ts_now,
                            )
                else:
                    # Order not filled (rejected, pending, etc.)
                    # Mark as failed in saga ledger
                    status = str(res.get("status") or res.get("state") or "unknown").lower() if isinstance(res, dict) else "unknown"
                    if status not in {"filled", "partially_filled"}:
                        execution_ledger.record_failed(
                            execution_id=exec_id,
                            error_msg=f"Order not filled: status={status}",
                            now_utc=ts_now,
                        )
                        log(f"[dt_exec] ❌ Order not filled: {exec_id} status={status}")
            except Exception as e:
                # Something went wrong during position update, record failure
                log(f"[dt_exec] ⚠️ Error in saga phase 2/3 for {exec_id}: {e}", level="error")
                if get_aggregator is not None:
                    stack_trace = traceback.format_exc()
                    get_aggregator().forward_log("ERROR", f"Saga phase 2/3 error for {exec_id}: {e}\n{stack_trace}", "dt_exec")
                execution_ledger.record_failed(
                    execution_id=exec_id,
                    error_msg=f"Phase 2/3 error: {str(e)[:200]}",
                    now_utc=ts_now,
                )
        except Exception as e:
            blocked += 1
            bump_metric("order_errors", 1.0)
            
            # Log error with full details and send to Slack
            log(f"[dt_exec] ⚠️ Error processing {sym}: {e}", level="error")
            if get_aggregator is not None:
                stack_trace = traceback.format_exc()
                get_aggregator().forward_log("ERROR", f"Order error for {sym}: {e}\n{stack_trace}", "dt_exec")
            
            # Try to record failure in ledger if we have exec_id
            if exec_id is not None:
                try:
                    execution_ledger.record_failed(
                        execution_id=exec_id,
                        error_msg=f"Broker submit error: {str(e)[:200]}",
                        now_utc=ts_now,
                    )
                except Exception:
                    pass
            
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
        "symbols": len(sym_list),
        "liquidation": liquidation_summary,
        "exits": {
            "evaluated": int(exit_summary.evaluated),
            "partials_sent": int(exit_summary.partials_sent),
            "exits_sent": int(exit_summary.exits_sent),
            "eod_flattens": int(exit_summary.eod_flattens),
        },
    }

    # Save rolling cache with updated position_dt fields for next cycle
    try:
        save_rolling(rolling)
        debug("[dt_exec] 💾 saved rolling cache with position updates")
    except Exception as e:
        log(f"[dt_exec] ⚠️ failed to save rolling cache: {e}")
    
    # Check for feature importance drift (ML interpretability)
    try:
        if get_feature_tracker is not None and orders > 0:
            tracker = get_feature_tracker()
            if tracker.detect_drift(threshold=0.15):
                log("[dt_exec] ⚠️ Feature importance drift detected - consider retraining")
    except Exception as e:
        debug(f"[dt_exec] Feature drift check failed: {e}")

    debug(f"[dt_exec] ✅ execute_from_policy done: {out}")
    log(f"[dt_exec] ✅ Execution complete: orders={orders} blocked={blocked} considered={considered}")
    return out
