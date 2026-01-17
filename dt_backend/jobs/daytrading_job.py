# dt_backend/jobs/daytrading_job.py — v3.3 (FAST-LANE / SLOW-LANE + CANDIDATE UNIVERSE)
"""Main intraday trading loop for AION dt_backend.

Pipeline:
    rolling → context_dt → features_dt → predictions_dt
            → regime → policy_dt → execution_dt → signals → (optional) broker execution

v3.3 additions
--------------
- Fast/Slow lane orchestration (speed)
- Candidate universe support:
    context_dt writes rolling["_GLOBAL_DT"]["candidate_universe_dt"]["symbols"]
    fast lane prefers this list for the per-cycle universe
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Sequence

# Hot-reload knobs from dt_knobs.env each cycle (so operators can flip switches
# without restarting the scheduler).
try:
    from dt_backend.utils.knob_loader_dt import maybe_reload_dt_knobs
except Exception:  # pragma: no cover
    maybe_reload_dt_knobs = None  # type: ignore

# Slack alerting for cycle tracking
try:
    from backend.monitoring.alerting import alert_dt, alert_error
except ImportError:
    alert_dt = None  # type: ignore
    alert_error = None  # type: ignore

from dt_backend.core import (
    log,
    warn,
    DT_PATHS,
    build_intraday_context,
    classify_intraday_regime,
    apply_intraday_policy,
)
from dt_backend.core.execution_dt import run_execution_intraday
from dt_backend.core.meta_controller_dt import ensure_daily_plan
from dt_backend.engines.feature_engineering import build_intraday_features
from dt_backend.ml import score_intraday_tickers, build_intraday_signals
from dt_backend.engines.trade_executor import ExecutionConfig, execute_from_policy

from dt_backend.services.dt_truth_store import (
    acquire_lock,
    CYCLE_LOCK_PATH,
    append_trade_event,
    update_dt_state,
    read_dt_state,
    write_metrics_snapshot,
)

from dt_backend.core.data_pipeline_dt import _read_rolling, save_rolling

# Phase 6.5 (shadow A/B): optional, safe-by-default
try:
    from dt_backend.shadow.shadow_cycle_dt import run_shadow_cycle
except Exception:
    run_shadow_cycle = None


# Phase 4.5: contextual bandit updater (shadow-first)
try:
    from dt_backend.bandit.contextual_bandit_dt import update_bandit_from_trades
except Exception:  # pragma: no cover
    update_bandit_from_trades = None  # type: ignore

# Phase 0: hard risk rails (kill-switch, cooldown, etc.)
try:
    from dt_backend.risk.risk_rails_dt import assess_and_update_risk_rails
except Exception:  # pragma: no cover
    assess_and_update_risk_rails = None  # type: ignore

# Phase 4: configuration validation (startup check)
try:
    from dt_backend.core.knob_validator_dt import validate_knobs
    knob_errors = validate_knobs()
    if knob_errors:
        for err in knob_errors:
            if err.startswith("WARNING:"):
                warn(f"[daytrading_job] ⚠️ Config: {err}")
            else:
                warn(f"[daytrading_job] ❌ Config error: {err}")
except Exception:  # pragma: no cover
    pass


# ----------------------------
# Fast lane / slow lane helpers
# ----------------------------

def _env_int(name: str, default: int) -> int:
    try:
        raw = (os.getenv(name, "") or "").strip()
        return int(float(raw)) if raw else int(default)
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


def _lane_config() -> Dict[str, Any]:
    return {
        "enabled": _env_bool("DT_FAST_LANE", True),
        "fast_n": _env_int("DT_FAST_SYMBOLS", 450),
        "slow_n": _env_int("DT_SLOW_SYMBOLS", 2500),
        "slow_every_n": max(1, _env_int("DT_SLOW_EVERY_N", 5)),
        "shadow_only_on_slow": _env_bool("DT_SHADOW_ONLY_ON_SLOW", False),
        # Universe source preference: candidate (default) -> plan -> fallback
        "universe_source": (os.getenv("DT_LANE_UNIVERSE_SOURCE", "candidate") or "candidate").strip().lower(),
        # Refresh candidates on slow cycles (recommended)
        "refresh_candidates_on_slow": _env_bool("DT_REFRESH_CANDIDATES_ON_SLOW", True),
    }


def _get_cycle_seq() -> int:
    try:
        st = read_dt_state() or {}
        seq = int(st.get("cycle_seq") or 0)
        return max(0, seq)
    except Exception:
        return 0


def _bump_cycle_seq(cycle_id: str) -> int:
    seq = _get_cycle_seq() + 1
    try:
        update_dt_state({"cycle_seq": seq, "cycle_id": cycle_id})
    except Exception:
        pass
    return seq


def _safe_syms(seq: Any) -> List[str]:
    if not isinstance(seq, (list, tuple)):
        return []
    out = []
    for s in seq:
        try:
            t = str(s).strip().upper()
            if t:
                out.append(t)
        except Exception:
            continue
    # deterministic
    return sorted(set(out))


def _get_candidate_symbols(rolling: Dict[str, Any], *, n: Optional[int] = None) -> Optional[List[str]]:
    gdt = rolling.get("_GLOBAL_DT") if isinstance(rolling.get("_GLOBAL_DT"), dict) else {}
    cu = gdt.get("candidate_universe_dt")
    if isinstance(cu, dict):
        syms = _safe_syms(cu.get("symbols") or cu.get("universe") or cu.get("candidates"))
        if syms:
            if n is not None:
                syms = syms[: max(0, int(n))]
            return syms
    if isinstance(cu, list):
        syms = _safe_syms(cu)
        if syms:
            if n is not None:
                syms = syms[: max(0, int(n))]
            return syms
    return None


def _choose_plan_symbols(plan: Dict[str, Any], *, n: Optional[int]) -> Optional[List[str]]:
    try:
        uni = plan.get("universe") if isinstance(plan, dict) else None
        syms = _safe_syms(uni)
        if syms:
            if n is not None:
                syms = syms[: max(0, int(n))]
            return syms
    except Exception:
        pass
    return None


def _call_maybe(fn, *, symbols: Optional[List[str]] = None, max_symbols: Optional[int] = None, **kwargs):
    """Backward-compatible caller:
      1) fn(symbols=..., max_symbols=..., **kwargs)
      2) fn(max_symbols=..., **kwargs)
      3) fn(**kwargs)
    """
    if symbols is not None:
        try:
            return fn(symbols=symbols, max_symbols=max_symbols, **kwargs)
        except TypeError:
            pass
    if max_symbols is not None:
        try:
            return fn(max_symbols=max_symbols, **kwargs)
        except TypeError:
            pass
    return fn(**kwargs)


def _append_cycle_decisions(cycle_id: str, rolling: Dict[str, Any]) -> None:
    if not isinstance(rolling, dict) or not rolling:
        return

    scored = []
    for sym, node in rolling.items():
        if not isinstance(sym, str) or sym.startswith("_"):
            continue
        if not isinstance(node, dict):
            continue
        p = node.get("policy_dt")
        if not isinstance(p, dict):
            continue
        try:
            score = float(p.get("score") or 0.0)
        except Exception:
            score = 0.0
        scored.append((sym, abs(score)))

    scored.sort(key=lambda t: t[1], reverse=True)
    top_n = int(os.getenv("DT_DECISION_LOG_TOP_N", "25") or "25")
    top_syms = set(sym for sym, _ in scored[: max(0, top_n)])

    append_trade_event({
        "type": "cycle",
        "cycle_id": cycle_id,
        "symbols": len([k for k in rolling.keys() if isinstance(k, str) and not k.startswith("_")]),
    })

    for sym, node in rolling.items():
        if not isinstance(sym, str) or sym.startswith("_"):
            continue
        if not isinstance(node, dict):
            continue

        p = node.get("policy_dt")
        if not isinstance(p, dict):
            continue
        ex = node.get("execution_dt")
        ex = ex if isinstance(ex, dict) else {}

        action = str(p.get("action") or "").upper()
        trade_gate = bool(p.get("trade_gate") is True)
        reason = str(p.get("reason") or "").strip()

        cooled = False
        try:
            if action in {"BUY", "SELL"} and trade_gate and str(ex.get("side") or "").upper() == "FLAT" and bool(ex.get("cooldown")):
                cooled = True
        except Exception:
            cooled = False

        if not (trade_gate or sym in top_syms or action == "STAND_DOWN" or cooled):
            continue

        append_trade_event({
            "type": "decision",
            "cycle_id": cycle_id,
            "symbol": sym,
            "action": action,
            "trade_gate": trade_gate,
            "confidence": p.get("confidence"),
            "score": p.get("score"),
            "reason": reason,
            "execution_side": ex.get("side"),
            "execution_size": ex.get("size"),
            "execution_cooldown": ex.get("cooldown"),
            "bot": (p.get("bot") if isinstance(p, dict) else None) or ex.get("bot"),
            "risk": ex.get("risk"),
        })


def run_daytrading_cycle(
    execute: bool = False,
    max_symbols: Optional[int] = None,
    max_positions: int = 50,
    execution_cfg: ExecutionConfig | None = None,
) -> Dict[str, Any]:
    cycle_id = uuid.uuid4().hex[:12]
    
    # Check emergency stop (Phase 4)
    try:
        from dt_backend.risk.emergency_stop_dt import check_emergency_stop
        is_stopped, reason = check_emergency_stop()
        if is_stopped:
            log(f"[daytrading_job] 🛑 EMERGENCY STOP: {reason}")
            append_trade_event({
                "type": "emergency_stop",
                "cycle_id": cycle_id,
                "reason": reason,
            })
            return {"status": "stopped", "reason": reason, "cycle_id": cycle_id}
    except Exception:
        # Never let emergency stop check break trading cycles
        pass

    # Hot-reload dt_knobs.env for operator toggles (liquidation, rails, sizing, etc.).
    # This makes edits to dt_knobs.env take effect on the very next cycle.
    try:
        if maybe_reload_dt_knobs is not None:
            _ = maybe_reload_dt_knobs()
    except Exception:
        # Never let knob reload break trading cycles.
        pass

    # Lock to prevent overlapping cycles across processes.
    if str(os.getenv("DT_CYCLE_LOCK", "1")).strip().lower() not in {"0", "false", "no", "off"}:
        lk = acquire_lock(CYCLE_LOCK_PATH, timeout_s=float(os.getenv("DT_CYCLE_LOCK_TIMEOUT", "5")))
        if not lk.acquired:
            warn(f"[daytrading_job] ⚠️ cycle overlap detected; skipping cycle_id={cycle_id}")
            append_trade_event({"type": "cycle_skipped", "cycle_id": cycle_id, "reason": "cycle_lock_busy"})
            return {"status": "skipped", "reason": "cycle_lock_busy", "cycle_id": cycle_id}
    else:
        lk = None

    log(f"[daytrading_job] 🚀 starting intraday cycle cycle_id={cycle_id}")

    os.environ.pop("DT_ROLLING_PATH", None)
    os.environ.setdefault("DT_USE_LOCK", (os.getenv("DT_USE_LOCK") or "1"))

    lane = _lane_config()
    cycle_seq = _bump_cycle_seq(cycle_id)

    try:
        update_dt_state({
            "component": "daytrading_cycle",
            "cycle_id": cycle_id,
            "cycle_seq": cycle_seq,
            "execute": bool(execute),
            "max_symbols": max_symbols,
        })

        # Daily plan (Phase 4)
        plan_force = str(os.getenv("DT_FORCE_NEW_PLAN", "0")).strip().lower() in {"1", "true", "yes", "y"}
        plan = ensure_daily_plan(force=plan_force)

        # Lane decision
        lane_label = "legacy"
        is_slow = False
        fast_n = None
        slow_n = None

        if lane.get("enabled"):
            slow_every = int(lane.get("slow_every_n") or 5)
            is_slow = (cycle_seq % max(1, slow_every) == 0)
            lane_label = "SLOW" if is_slow else "FAST"
            fast_n = int(lane.get("fast_n") or 450)
            slow_n = int(lane.get("slow_n") or 2500)
        lane_max_symbols = max_symbols

        # Persist plan once per date
        try:
            st = read_dt_state()
            prev_date = str((st.get("daily_plan_dt") or {}).get("date") or "") if isinstance(st.get("daily_plan_dt"), dict) else ""
            if isinstance(plan, dict) and plan.get("date") and plan.get("date") != prev_date:
                update_dt_state({
                    "daily_plan_dt": plan,
                    "risk_mode": plan.get("risk_mode"),
                    "enabled_bots": plan.get("enabled_bots"),
                    "universe_size": len(plan.get("universe") or []) if isinstance(plan.get("universe"), list) else None,
                })
                append_trade_event({
                    "type": "daily_plan",
                    "cycle_id": cycle_id,
                    "date": plan.get("date"),
                    "risk_mode": plan.get("risk_mode"),
                    "enabled_bots": plan.get("enabled_bots"),
                    "bot_weights": plan.get("bot_weights"),
                    "universe_size": len(plan.get("universe") or []) if isinstance(plan.get("universe"), list) else None,
                    "reason": plan.get("reason"),
                })
        except Exception:
            pass

        # Record lane
        try:
            update_dt_state({
                "lane": lane_label,
                "lane_fast_enabled": bool(lane.get("enabled")),
                "lane_slow_every_n": lane.get("slow_every_n"),
            })
            append_trade_event({"type": "lane", "cycle_id": cycle_id, "cycle_seq": cycle_seq, "lane": lane_label})
        except Exception:
            pass

        # Phase 0 — Hard risk rails
        try:
            if assess_and_update_risk_rails is not None:
                risk_rails_summary = assess_and_update_risk_rails()
                rolling_now = _read_rolling() or {}
                gdt = rolling_now.get("_GLOBAL_DT") if isinstance(rolling_now.get("_GLOBAL_DT"), dict) else {}
                gdt["risk_rails_dt"] = risk_rails_summary
                gdt["stand_down"] = bool(risk_rails_summary.get("stand_down")) if isinstance(risk_rails_summary, dict) else False
                gdt["stand_down_reason"] = str(risk_rails_summary.get("reason")) if isinstance(risk_rails_summary, dict) else ""
                rolling_now["_GLOBAL_DT"] = gdt
                save_rolling(rolling_now)
                append_trade_event({
                    "type": "risk_rail",
                    "cycle_id": cycle_id,
                    "event": "assessed",
                    "stand_down": gdt.get("stand_down"),
                    "reason": gdt.get("stand_down_reason"),
                })
        except Exception:
            pass

        # ---------------------------------------------
        # Context refresh strategy:
        #   - Slow cycle: refresh context broadly and rebuild candidate universe
        #   - Fast cycle: refresh context only for chosen symbols
        # ---------------------------------------------
        rolling_pre = _read_rolling() or {}
        candidate_pre = _get_candidate_symbols(rolling_pre)

        # If slow cycle, optionally refresh broad context first.
        ctx_summary = None
        if lane.get("enabled") and is_slow and bool(lane.get("refresh_candidates_on_slow")):
            lane_max_symbols = slow_n
            ctx_summary = _call_maybe(build_intraday_context, symbols=None, max_symbols=lane_max_symbols)

        # Choose symbol lane universe
        rolling_now = _read_rolling() or {}
        universe_source = str(lane.get("universe_source") or "candidate").lower()

        symbols_lane: Optional[List[str]] = None
        if lane.get("enabled"):
            n = slow_n if is_slow else fast_n
            if universe_source in {"candidate", "candidates"}:
                symbols_lane = _get_candidate_symbols(rolling_now, n=n)
                if not symbols_lane:
                    symbols_lane = _get_candidate_symbols(rolling_pre, n=n)
            if not symbols_lane:
                symbols_lane = _choose_plan_symbols(plan if isinstance(plan, dict) else {}, n=n)
            lane_max_symbols = n

        # If we didn't run broad context above, run lane-scoped context now.
        if ctx_summary is None:
            ctx_summary = _call_maybe(build_intraday_context, symbols=symbols_lane, max_symbols=lane_max_symbols)

        # Lane-scoped pipeline (backward-compatible calls)
        feat_summary = _call_maybe(build_intraday_features, symbols=symbols_lane, max_symbols=lane_max_symbols)
        score_summary = _call_maybe(score_intraday_tickers, symbols=symbols_lane, max_symbols=lane_max_symbols)
        regime_summary = classify_intraday_regime()

        # Policy (scoped if signature supports it)
        try:
            policy_summary = _call_maybe(
                apply_intraday_policy,
                symbols=symbols_lane,
                max_symbols=lane_max_symbols,
                max_positions=max_positions,
            )
        except TypeError:
            policy_summary = apply_intraday_policy(max_positions=max_positions)

        # Execution + signals
        try:
            exec_dt_summary = _call_maybe(run_execution_intraday, symbols=symbols_lane, max_symbols=lane_max_symbols)
        except TypeError:
            exec_dt_summary = run_execution_intraday()

        try:
            signals_summary = _call_maybe(build_intraday_signals, symbols=symbols_lane, max_symbols=lane_max_symbols)
        except TypeError:
            signals_summary = build_intraday_signals()

        # Shadow mode (optional)
        shadow_summary = None
        try:
            shadow_on = str(os.getenv("DT_SHADOW_ENABLED", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}
            if lane.get("shadow_only_on_slow") and not is_slow:
                shadow_on = False
            if shadow_on and run_shadow_cycle is not None:
                every_n = int(os.getenv("DT_SHADOW_EVERY_N", "1") or "1")
                h = sum(bytearray(str(cycle_id).encode("utf-8", errors="ignore")))
                if every_n <= 1 or (h % max(1, every_n) == 0):
                    live_path = DT_PATHS.get("rolling_intraday_file")
                    if live_path is not None:
                        shadow_summary = run_shadow_cycle(
                            cycle_id=cycle_id,
                            live_rolling_path=live_path,
                            max_symbols=lane_max_symbols,
                            max_positions=max_positions,
                        )
        except Exception:
            shadow_summary = None

        # Persist latest view
        try:
            update_dt_state({
                "regime_dt": regime_summary,
                "micro_regime_dt": (((_read_rolling() or {}).get("_GLOBAL_DT") or {}) if isinstance((_read_rolling() or {}).get("_GLOBAL_DT"), dict) else {}).get("micro_regime_dt"),
                "daily_plan_dt": (((_read_rolling() or {}).get("_GLOBAL_DT") or {}) if isinstance((_read_rolling() or {}).get("_GLOBAL_DT"), dict) else {}).get("daily_plan_dt"),
                "policy_summary": policy_summary,
                "lane": lane_label,
                "lane_symbols": len(symbols_lane) if isinstance(symbols_lane, list) else None,
                "lane_max_symbols": lane_max_symbols,
                "lane_universe_source": universe_source,
            })
        except Exception:
            pass

        # Snapshot decisions + metrics
        try:
            rolling = _read_rolling() or {}
            _append_cycle_decisions(cycle_id, rolling)
            metrics_snap = write_metrics_snapshot(rolling=rolling)
            try:
                g = rolling.get("_GLOBAL_DT") if isinstance(rolling.get("_GLOBAL_DT"), dict) else {}
                g["dt_metrics"] = metrics_snap
                rolling["_GLOBAL_DT"] = g
                save_rolling(rolling)
            except Exception:
                pass
            try:
                if update_bandit_from_trades is not None:
                    update_bandit_from_trades()
            except Exception:
                pass
        except Exception:
            pass

        exec_summary: Dict[str, Any] | None = None
        if execute:
            exec_summary = execute_from_policy(execution_cfg)
            
            # Send cycle completion alert
            if alert_dt is not None and exec_summary:
                try:
                    orders = exec_summary.get("orders", 0)
                    considered = exec_summary.get("considered", 0)
                    blocked = exec_summary.get("blocked", 0)
                    exits_summary = exec_summary.get("exits", {})
                    exits_sent = exits_summary.get("exits_sent", 0) if isinstance(exits_summary, dict) else 0
                    
                    if orders > 0 or exits_sent > 0:
                        alert_dt(
                            f"DT Cycle Complete: {cycle_id[:8]}",
                            f"Lane: {lane_label} | Seq: {cycle_seq}",
                            level="info",
                            context={
                                "Orders Sent": f"{orders}",
                                "Exits Sent": f"{exits_sent}",
                                "Considered": f"{considered}",
                                "Blocked": f"{blocked}",
                                "Symbols": f"{len(symbols_lane) if isinstance(symbols_lane, list) else 0}",
                            }
                        )
                except Exception:
                    pass

        log(f"[daytrading_job] ✅ intraday cycle complete cycle_id={cycle_id} lane={lane_label}")
        
        # SSE Broadcast Note:
        # Data changes are picked up by SSE polling in events_router.py (/events/bots, /events/intraday)
        # which polls every 5 seconds. When this cycle completes, file writes update:
        # - rolling.json.gz (intraday signals)
        # - sim_logs/*.json (bot activity)
        # - sim_summary.json (PnL summary)
        # The next SSE poll will fetch fresh data and push to connected clients.
        # Client-side cache auto-invalidates on SSE push for instant UI updates.
        
        return {
            "cycle_id": cycle_id,
            "cycle_seq": cycle_seq,
            "lane": lane_label,
            "lane_symbols": len(symbols_lane) if isinstance(symbols_lane, list) else None,
            "context": ctx_summary,
            "features": feat_summary,
            "scoring": score_summary,
            "regime": regime_summary,
            "policy": policy_summary,
            "shadow": shadow_summary,
            "execution_dt": exec_dt_summary,
            "signals": signals_summary,
            "execution": exec_summary,
        }
    finally:
        try:
            if lk is not None:
                lk.release()
        except Exception:
            pass


def main() -> None:
    run_daytrading_cycle(execute=False)


if __name__ == "__main__":
    main()
