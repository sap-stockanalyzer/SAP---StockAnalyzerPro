# dt_backend/jobs/daytrading_job.py — v3.1 (SINGLE-ROLLING + SAFE DEFAULT LOCK)
"""Main intraday trading loop for AION dt_backend.

This job wires together the intraday pipeline:

    rolling → context_dt → features_dt → predictions_dt
            → regime → policy_dt → execution_dt → (optional) broker execution

Architecture (Linux server)
---------------------------
We use a **single** rolling cache file (DT_PATHS['rolling_intraday_file']).

* Live bars are a bounded sliding window and are expected to overwrite.
* Policy/execution/learning state is written into the same rolling.
* Long-lived learning should live in a separate dt_brain artifact (see
  dt_backend/core/dt_brain.py), not in rolling.

Safety notes
------------
- We do NOT embed model-training code in this job file.
- Rolling writes are atomic, but atomic replace does not prevent "lost updates"
  when multiple processes read-modify-write at the same time. So we default the
  rolling lock ON for this process unless the environment explicitly disables it.

    DT_USE_LOCK=1   (default here)
    DT_USE_LOCK=0   (disable if you truly know it's single-writer)
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional

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
    from dt_backend.ab.shadow_ab_dt import run_shadow_ab
except Exception:  # pragma: no cover
    run_shadow_ab = None  # type: ignore

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

def _append_cycle_decisions(cycle_id: str, rolling: Dict[str, Any]) -> None:
    """Write durable decision logs (Phase 0).

    We don't want to spam dt_trades.jsonl with *every* symbol, every minute.
    So we log:
      • any symbol with trade_gate=True
      • top N by |score|
      • any symbol in STAND_DOWN
      • any symbol where execution_dt cooled down an otherwise valid trade
    """
    if not isinstance(rolling, dict) or not rolling:
        return

    # Gather candidates
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

    # Cycle-level marker
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

        # Cooldown case: policy wanted BUY/SELL but execution became FLAT.
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
    """Run one full intraday cycle."""
    cycle_id = uuid.uuid4().hex[:12]

    # Per-cycle lock to prevent overlapping cycles across processes.
    # (You can disable by setting DT_CYCLE_LOCK=0, but don't unless you mean it.)
    if str(os.getenv("DT_CYCLE_LOCK", "1")).strip().lower() not in {"0", "false", "no", "off"}:
        lk = acquire_lock(CYCLE_LOCK_PATH, timeout_s=float(os.getenv("DT_CYCLE_LOCK_TIMEOUT", "5")))
        if not lk.acquired:
            warn(f"[daytrading_job] ⚠️ cycle overlap detected; skipping cycle_id={cycle_id}")
            append_trade_event({
                "type": "cycle_skipped",
                "cycle_id": cycle_id,
                "reason": "cycle_lock_busy",
            })
            return {"status": "skipped", "reason": "cycle_lock_busy", "cycle_id": cycle_id}
    else:
        lk = None

    log(f"[daytrading_job] 🚀 starting intraday cycle cycle_id={cycle_id}")

    # Single rolling file: disallow per-process split unless explicitly set elsewhere.
    os.environ.pop("DT_ROLLING_PATH", None)

    # Default rolling lock ON for this job (prevents read-modify-write stomps across processes).
    os.environ.setdefault("DT_USE_LOCK", (os.getenv("DT_USE_LOCK") or "1"))

    try:
        update_dt_state({
            "component": "daytrading_cycle",
            "cycle_id": cycle_id,
            "execute": bool(execute),
            "max_symbols": max_symbols,
        })

        ctx_summary = build_intraday_context()
        feat_summary = build_intraday_features(max_symbols=max_symbols)
        score_summary = score_intraday_tickers(max_symbols=max_symbols)
        regime_summary = classify_intraday_regime()

        # Phase 4: choose today's bots/risk/universe once per day.
        plan_force = str(os.getenv("DT_FORCE_NEW_PLAN", "0")).strip().lower() in {"1","true","yes","y"}
        plan = ensure_daily_plan(force=plan_force)

        # Persist daily plan into dt_state.json once per date
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

        # ------------------------------------------------------------
        # Phase 0 — Hard risk rails (kill-switch/cooldown)
        # ------------------------------------------------------------
        risk_rails_summary = None
        try:
            if assess_and_update_risk_rails is not None:
                risk_rails_summary = assess_and_update_risk_rails()
                # also mirror into rolling so policy can stand down instantly
                rolling_now = _read_rolling() or {}
                gdt = rolling_now.get("_GLOBAL_DT") if isinstance(rolling_now.get("_GLOBAL_DT"), dict) else {}
                gdt["risk_rails_dt"] = risk_rails_summary
                gdt["stand_down"] = bool(risk_rails_summary.get("stand_down")) if isinstance(risk_rails_summary, dict) else False
                gdt["stand_down_reason"] = str(risk_rails_summary.get("reason")) if isinstance(risk_rails_summary, dict) else ""
                rolling_now["_GLOBAL_DT"] = gdt
                save_rolling(rolling_now)
                try:
                    append_trade_event({
                        "type": "risk_rail",
                        "cycle_id": cycle_id,
                        "event": "assessed",
                        "stand_down": gdt.get("stand_down"),
                        "reason": gdt.get("stand_down_reason"),
                    })
                except Exception:
                    pass
        except Exception:
            risk_rails_summary = None

        policy_summary = apply_intraday_policy(max_positions=max_positions)

        # ------------------------------------------------------------
        # Phase 6.5 — Shadow mode (A/B discipline)
        # ------------------------------------------------------------
        shadow_summary = None
        try:
            shadow_on = str(os.getenv("DT_SHADOW_ENABLED", "0") or "0").strip().lower() in {"1", "true", "yes", "y", "on"}
            if shadow_on and run_shadow_cycle is not None:
                every_n = int(os.getenv("DT_SHADOW_EVERY_N", "1") or "1")
                # cheap rate limiter: run shadow every N cycles
                # cycle_id is a UUID; don't assume hex suffix is parseable.
                h = sum(bytearray(str(cycle_id).encode("utf-8", errors="ignore")))
                if every_n <= 1 or (h % max(1, every_n) == 0):
                    live_path = DT_PATHS.get("rolling_intraday_file")
                    if live_path is not None:
                        shadow_summary = run_shadow_cycle(
                            cycle_id=cycle_id,
                            live_rolling_path=live_path,
                            max_symbols=max_symbols,
                            max_positions=max_positions,
                        )
                        try:
                            append_trade_event({
                                "type": "shadow_compare",
                                "cycle_id": cycle_id,
                                "shadow_version": shadow_summary.get("shadow_version"),
                                "agreement_rate": shadow_summary.get("agreement_rate"),
                                "symbols_compared": shadow_summary.get("symbols_compared"),
                                "divergences": shadow_summary.get("divergences"),
                            })
                        except Exception:
                            pass
        except Exception:
            shadow_summary = None

        # Persist the latest view of the world (regime + micro + strategy weights)
        try:
            update_dt_state({
                "regime_dt": regime_summary,
                "micro_regime_dt": ( (_read_rolling() or {}).get("_GLOBAL_DT") or {} ).get("micro_regime_dt"),
                "daily_plan_dt": ( (_read_rolling() or {}).get("_GLOBAL_DT") or {} ).get("daily_plan_dt"),
                "policy_summary": policy_summary,
            })
        except Exception:
            pass

        exec_dt_summary = run_execution_intraday()
        signals_summary = build_intraday_signals()

        # Snapshot decisions (top candidates + any trade gates) for durable debugging.
        try:
            rolling = _read_rolling() or {}
            _append_cycle_decisions(cycle_id, rolling)
            metrics_snap = write_metrics_snapshot(rolling=rolling)
            # mirror metrics into rolling global for other modules (heat/risk)
            try:
                g = rolling.get("_GLOBAL_DT") if isinstance(rolling.get("_GLOBAL_DT"), dict) else {}
                g["dt_metrics"] = metrics_snap
                rolling["_GLOBAL_DT"] = g
                save_rolling(rolling)
            except Exception:
                pass
            # Phase 4.5: update bandit state from realized exits (best-effort)
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

        log(f"[daytrading_job] ✅ intraday cycle complete cycle_id={cycle_id}")
        return {
            "cycle_id": cycle_id,
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
