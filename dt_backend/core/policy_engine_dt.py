# v2.1.2 (LANE-AWARE, POSITION-AWARE, PROFILE-AWARE)
"""
Intraday policy engine for dt_backend.

Writes:
    rolling[sym]["policy_dt"] = {
        "action": "BUY"|"SELL"|"HOLD"|"STAND_DOWN",
        "intent":  same as action (legacy compatibility),
        "confidence": 0.0..0.99,
        "p_hit": optional calibrated probability-of-hit,
        "score": signed strength (positive favors BUY, negative favors SELL),
        "trade_gate": bool,
        "reason": short human-readable summary,
        "ts": ISO8601 UTC,
        "_state": { ... hysteresis memory ... }   # internal / debug
    }

Design goals
------------
• Stable (anti-flip hysteresis)
• Uses probabilities + context + global regime
• Safe defaults
• Pure Python

v2.1 additions
--------------
• Lane-aware: accepts optional `symbols=[...]` and respects `max_symbols`
• Avoids touching symbols outside the requested lane universe (except global safety stand-down)
• Fixes a bug where `feats` could be referenced before assignment in strategy path
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

from dt_backend.tuning.dt_profile_loader import load_dt_profile, strategy_weight
from .data_pipeline_dt import _read_rolling, save_rolling, ensure_symbol_node, log
from dt_backend.core.constants_dt import (
    CONFIDENCE_MIN,
    CONFIDENCE_MAX,
    EDGE_MIN_TO_FLIP,
    EDGE_HOLD_BIAS,
    CONFIRMATIONS_TO_FLIP,
    VOL_PENALTY_HIGH,
    VOL_PENALTY_MEDIUM,
    TREND_BOOST_STRONG,
    TREND_BOOST_MILD,
    REGIME_PENALTY_CHOP,
    REGIME_PENALTY_BEAR_BUY,
    REGIME_PENALTY_BULL_SELL,
    BUY_THRESHOLD,
    SELL_THRESHOLD,
)

# --- OPTIONAL IMPORTS (UNCHANGED) ---
try:
    from dt_backend.calibration.phit_calibrator_dt import get_phit
except Exception:
    get_phit = None  # type: ignore

try:
    from dt_backend.risk.news_event_risk_dt import assess_symbol_risk
except Exception:
    assess_symbol_risk = None  # type: ignore

try:
    from dt_backend.researcher.rules_runtime_dt import bot_allowed
except Exception:
    bot_allowed = None  # type: ignore

try:
    from dt_backend.risk.news_event_risk_dt import risk_adjust_policy
except Exception:
    risk_adjust_policy = None  # type: ignore

try:
    from dt_backend.risk.portfolio_heat_dt import apply_portfolio_heat_gates
except Exception:
    apply_portfolio_heat_gates = None  # type: ignore

try:
    from dt_backend.researcher.rules_dt import rules_adjust_setup
except Exception:
    rules_adjust_setup = None  # type: ignore

try:
    from dt_backend.strategies import select_best_setup
except Exception:
    select_best_setup = None  # type: ignore

try:
    from dt_backend.ml.feature_importance_tracker import get_tracker as get_feature_tracker
except Exception:
    get_feature_tracker = None  # type: ignore


# ============================================================
# 🔐 POSITION DETECTION (NEW, SAFE, BACKWARD-COMPATIBLE)
# ============================================================

def _has_position(node: Dict[str, Any]) -> bool:
    """
    Best-effort detection of an open long position.
    Compatible with multiple execution backends.
    """
    try:
        pos = node.get("position_dt") or node.get("position") or {}
        if isinstance(pos, dict):
            return float(pos.get("qty") or 0.0) > 0.0
        return bool(node.get("holding") is True)
    except Exception:
        return False

@dataclass
class PolicyConfig:
    # Signal edge thresholds (p_buy - p_sell)
    buy_threshold: float = BUY_THRESHOLD
    sell_threshold: float = SELL_THRESHOLD

    # Minimum confidence for acting (after adjustments)
    min_confidence: float = CONFIDENCE_MIN

    # Volatility penalties
    vol_penalty_high: float = VOL_PENALTY_HIGH
    vol_penalty_medium: float = VOL_PENALTY_MEDIUM

    # Trend boosts when aligned
    trend_boost_strong: float = TREND_BOOST_STRONG
    trend_boost_mild: float = TREND_BOOST_MILD

    # Regime effects
    chop_penalty: float = REGIME_PENALTY_CHOP
    bear_buy_penalty: float = REGIME_PENALTY_BEAR_BUY
    bull_sell_penalty: float = REGIME_PENALTY_BULL_SELL

    # Stability / anti-flip
    hysteresis_hold_bias: float = 0.03     # make HOLD "sticky" by requiring extra edge to flip
    min_edge_to_flip: float = 0.12         # require at least this edge magnitude to flip direction (raised from 0.06)
    confirmations_to_flip: int = 2         # require N consecutive signals before switching BUY<->SELL
    max_confidence: float = 0.99

    # Safety gate: in crash/stress regimes, optionally stand down
    stand_down_in_unknown_regime: bool = False
    stand_down_in_crash: bool = True

    # -----------------------------
    # Phase 4: Human-like action tiers
    # -----------------------------
    # PROBE: allow micro-sized entries even when we wouldn't pass the full trade gate.
    probe_enabled: bool = True
    probe_min_signal_conf: float = 0.18
    probe_min_abs_score: float = 12.0  # setup.score is typically 0..100

    # PRESS: scale up only when P(hit) + R are strong.
    press_min_phit: float = 0.62
    press_min_expected_r: float = 1.20
    press_conf_extra: float = 0.05


def _env_float(name: str, default: float) -> float:
    try:
        import os
        raw = (os.getenv(name, "") or "").strip()
        if raw == "":
            return float(default)
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        import os
        raw = (os.getenv(name, "") or "").strip()
        if raw == "":
            return int(default)
        return int(float(raw))
    except Exception:
        return int(default)


def _apply_env_overrides(cfg: PolicyConfig) -> PolicyConfig:
    """Optional per-bot tuning via env vars.

    Supported:
      DT_BUY_THRESHOLD, DT_SELL_THRESHOLD
      DT_MIN_CONFIDENCE
      DT_CONFIRMATIONS_TO_FLIP
      DT_MIN_EDGE_TO_FLIP
      DT_HOLD_STICKY_BIAS
    """
    cfg.buy_threshold = _env_float("DT_BUY_THRESHOLD", cfg.buy_threshold)
    cfg.sell_threshold = _env_float("DT_SELL_THRESHOLD", cfg.sell_threshold)
    cfg.min_confidence = _env_float("DT_MIN_CONFIDENCE", cfg.min_confidence)
    cfg.confirmations_to_flip = _env_int("DT_CONFIRMATIONS_TO_FLIP", cfg.confirmations_to_flip)
    cfg.min_edge_to_flip = _env_float("DT_MIN_EDGE_TO_FLIP", cfg.min_edge_to_flip)
    cfg.hysteresis_hold_bias = _env_float("DT_HOLD_STICKY_BIAS", cfg.hysteresis_hold_bias)

    # Phase 4: tiers
    try:
        import os
        cfg.probe_enabled = (os.getenv("DT_PROBE_ENABLED", "1").strip().lower() in {"1", "true", "yes", "y", "on"})
    except Exception:
        cfg.probe_enabled = True
    cfg.probe_min_signal_conf = _env_float("DT_PROBE_MIN_CONF", cfg.probe_min_signal_conf)
    cfg.probe_min_abs_score = _env_float("DT_PROBE_MIN_SCORE", cfg.probe_min_abs_score)
    cfg.press_min_phit = _env_float("DT_PRESS_MIN_PHIT", cfg.press_min_phit)
    cfg.press_min_expected_r = _env_float("DT_PRESS_MIN_R", cfg.press_min_expected_r)
    cfg.press_conf_extra = _env_float("DT_PRESS_CONF_EXTRA", cfg.press_conf_extra)

    # sane clamps
    cfg.buy_threshold = max(0.0, min(1.0, cfg.buy_threshold))
    cfg.sell_threshold = min(0.0, max(-1.0, cfg.sell_threshold))
    cfg.min_confidence = max(0.0, min(1.0, cfg.min_confidence))
    cfg.confirmations_to_flip = max(1, int(cfg.confirmations_to_flip))
    cfg.min_edge_to_flip = max(0.0, min(1.0, cfg.min_edge_to_flip))
    cfg.hysteresis_hold_bias = max(0.0, min(0.25, cfg.hysteresis_hold_bias))

    cfg.probe_min_signal_conf = max(0.0, min(0.60, float(cfg.probe_min_signal_conf)))
    cfg.probe_min_abs_score = max(0.0, min(100.0, float(cfg.probe_min_abs_score)))
    cfg.press_min_phit = max(0.50, min(0.90, float(cfg.press_min_phit)))
    cfg.press_min_expected_r = max(0.5, min(5.0, float(cfg.press_min_expected_r)))
    cfg.press_conf_extra = max(0.0, min(0.25, float(cfg.press_conf_extra)))
    return cfg



def _apply_profile_overrides(cfg: PolicyConfig, profile: Dict[str, Any]) -> PolicyConfig:
    """Apply playbook profile overrides (Phase 5). Soft gates only.

    NOTE: env overrides already applied earlier; we only clamp here.
    """
    try:
        soft = profile.get("soft_thresholds") or {}
        if "dt_min_confidence" in soft:
            cfg.min_confidence = float(soft["dt_min_confidence"])
    except Exception:
        pass

    try:
        p = profile.get("probe") or {}
        if "enabled" in p:
            cfg.probe_enabled = bool(p.get("enabled"))
        if "min_conf" in p:
            cfg.probe_min_signal_conf = float(p.get("min_conf"))
        if "min_score" in p:
            cfg.probe_min_abs_score = float(p.get("min_score"))
    except Exception:
        pass

    try:
        pr = profile.get("press") or {}
        if "min_phit" in pr:
            cfg.press_min_phit = float(pr.get("min_phit"))
        if "min_r" in pr:
            cfg.press_min_expected_r = float(pr.get("min_r"))
    except Exception:
        pass

    # Clamp (match env clamp behavior)
    cfg.buy_threshold = max(0.0, min(1.0, cfg.buy_threshold))
    cfg.sell_threshold = min(0.0, max(-1.0, cfg.sell_threshold))
    cfg.min_confidence = max(0.0, min(1.0, cfg.min_confidence))
    cfg.confirmations_to_flip = max(1, int(cfg.confirmations_to_flip))
    cfg.min_edge_to_flip = max(0.0, min(1.0, cfg.min_edge_to_flip))
    cfg.hysteresis_hold_bias = max(0.0, min(0.25, cfg.hysteresis_hold_bias))

    cfg.probe_min_signal_conf = max(0.0, min(0.60, float(cfg.probe_min_signal_conf)))
    cfg.probe_min_abs_score = max(0.0, min(100.0, float(cfg.probe_min_abs_score)))
    cfg.press_min_phit = max(0.50, min(0.90, float(cfg.press_min_phit)))
    cfg.press_min_expected_r = max(0.5, min(5.0, float(cfg.press_min_expected_r)))
    cfg.press_conf_extra = max(0.0, min(0.25, float(cfg.press_conf_extra)))
    return cfg

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _get_global_regime(rolling: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer new regime_dt (Phase 2) but support legacy regime."""
    g = rolling.get("_GLOBAL_DT") or {}
    if not isinstance(g, dict):
        return {"label": "unknown"}

    reg_dt = g.get("regime_dt")
    if isinstance(reg_dt, dict) and reg_dt.get("label"):
        return reg_dt

    reg = g.get("regime") or {}
    if isinstance(reg, dict) and reg.get("label"):
        return reg
    return {"label": "unknown"}


def _get_micro_regime(rolling: Dict[str, Any]) -> Dict[str, Any]:
    g = rolling.get("_GLOBAL_DT") or {}
    if not isinstance(g, dict):
        return {}
    m = g.get("micro_regime_dt") or {}
    return m if isinstance(m, dict) else {}


def _get_daily_plan(rolling: Dict[str, Any]) -> Dict[str, Any]:
    """Phase 4: daily plan selected by meta-controller."""
    g = rolling.get("_GLOBAL_DT") or {}
    if not isinstance(g, dict):
        return {}
    p = g.get("daily_plan_dt") or {}
    return p if isinstance(p, dict) else {}


def _regime_for_policy(label: str) -> str:
    """Map Phase 2 labels into the older bull/bear/chop/stress bucket set."""
    r = (label or "").strip().upper()
    if r in {"BULL", "BEAR", "CHOP", "CRASH", "STRESS", "UNKNOWN"}:
        return r.lower()
    if r == "TREND_UP":
        return "bull"
    if r == "TREND_DOWN":
        return "bear"
    if r in {"RANGE", "LOW_VOL"}:
        return "chop"
    if r == "HIGH_VOL":
        return "stress"
    return "unknown"


def _extract_prediction(node: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, float]]:
    """Best-effort extraction of dt predictions."""
    pred = node.get("predictions_dt") or node.get("predictions") or {}
    if not isinstance(pred, dict):
        return None, {}

    label = pred.get("label") or pred.get("class")
    proba_raw = pred.get("proba") or pred.get("probs") or {}
    if not isinstance(proba_raw, dict):
        proba_raw = {}

    proba: Dict[str, float] = {}
    for k, v in proba_raw.items():
        if not isinstance(k, str):
            continue
        proba[k.upper()] = _safe_float(v, 0.0)

    total = sum(proba.values())
    if total > 0:
        proba = {k: v / total for k, v in proba.items()}

    return (label.upper() if isinstance(label, str) else None), proba


def _trend_and_vol(ctx: Dict[str, Any]) -> Tuple[str, str]:
    trend = str(ctx.get("intraday_trend") or "").strip() or "flat"
    vol_bkt = str(ctx.get("vol_bucket") or "").strip() or "low"
    return trend, vol_bkt


def _raw_intent_from_edge(p_buy: float, p_hold: float, p_sell: float, cfg: PolicyConfig) -> Tuple[str, float, float]:
    """Returns (intent, base_conf, edge)."""
    edge = p_buy - p_sell
    base_conf = max(p_buy, p_sell)  # HOLD shouldn't trigger action by itself

    if edge >= cfg.buy_threshold:
        return "BUY", base_conf, edge
    if edge <= cfg.sell_threshold:
        return "SELL", base_conf, edge
    return "HOLD", base_conf, edge


def _adjust_conf(
    base_conf: float,
    intent: str,
    trend: str,
    vol_bkt: str,
    regime_label: str,
    cfg: PolicyConfig,
) -> Tuple[float, str]:
    conf = float(base_conf)
    detail: List[str] = []

    if conf <= 0:
        return 0.0, "no_base_conf"

    # Vol penalty
    if vol_bkt == "high":
        conf *= cfg.vol_penalty_high
        detail.append("vol=high")
    elif vol_bkt == "medium":
        conf *= cfg.vol_penalty_medium
        detail.append("vol=medium")
    else:
        detail.append("vol=low")

    # Trend alignment boosts
    if intent == "BUY":
        if trend == "strong_bull":
            conf *= cfg.trend_boost_strong
            detail.append("trend=strong_bull")
        elif trend == "bull":
            conf *= cfg.trend_boost_mild
            detail.append("trend=bull")
    elif intent == "SELL":
        if trend == "strong_bear":
            conf *= cfg.trend_boost_strong
            detail.append("trend=strong_bear")
        elif trend == "bear":
            conf *= cfg.trend_boost_mild
            detail.append("trend=bear")

    # Regime effects (global)
    r = (regime_label or "unknown").lower()
    if r in {"chop"}:
        conf *= cfg.chop_penalty
        detail.append("regime=chop")
    elif r in {"bear"} and intent == "BUY":
        conf *= cfg.bear_buy_penalty
        detail.append("regime=bear_buy_penalty")
    elif r in {"bull"} and intent == "SELL":
        conf *= cfg.bull_sell_penalty
        detail.append("regime=bull_sell_penalty")
    elif r in {"crash", "stress"}:
        conf *= 0.80
        detail.append(f"regime={r}")
    else:
        detail.append(f"regime={r}")

    conf = max(0.0, min(cfg.max_confidence, conf))
    return float(conf), "; ".join(detail)


def _stabilize_with_hysteresis(
    node: Dict[str, Any],
    proposed: str,
    edge: float,
    conf: float,
    cfg: PolicyConfig,
    *,
    policy_key: str = "policy_dt",
) -> Tuple[str, Dict[str, Any], str]:
    """Use per-symbol memory to prevent flip-flopping."""
    policy_prev = node.get(policy_key) or {}
    prev_action = str((policy_prev.get("action") or policy_prev.get("intent") or "HOLD")).upper()
    state = policy_prev.get("_state") or {}
    if not isinstance(state, dict):
        state = {}

    pending = str(state.get("pending_action") or "").upper()
    pending_count = int(state.get("pending_count") or 0)

    final_action = proposed
    note = ""

    # HOLD stickiness
    if prev_action == "HOLD" and proposed in {"BUY", "SELL"}:
        if abs(edge) < (cfg.min_edge_to_flip + cfg.hysteresis_hold_bias):
            final_action = "HOLD"
            note = "hysteresis_hold_sticky"
        else:
            note = "hysteresis_hold_released"

    # Direction flip rules
    if prev_action in {"BUY", "SELL"} and proposed in {"BUY", "SELL"} and proposed != prev_action:
        if abs(edge) < cfg.min_edge_to_flip:
            final_action = prev_action
            note = "flip_blocked_edge_too_small"
        else:
            # confirmations
            if pending != proposed:
                pending = proposed
                pending_count = 1
            else:
                pending_count += 1

            if pending_count < max(1, cfg.confirmations_to_flip):
                final_action = prev_action
                note = f"flip_wait_confirmations({pending_count}/{cfg.confirmations_to_flip})"
            else:
                final_action = proposed
                note = "flip_confirmed"

    # update state
    if final_action == proposed:
        state["pending_action"] = ""
        state["pending_count"] = 0
    else:
        state["pending_action"] = pending
        state["pending_count"] = pending_count

    state["prev_action"] = prev_action
    state["last_edge"] = float(edge)
    state["last_conf"] = float(conf)

    return final_action, state, note


def _lane_symbols(rolling: Dict[str, Any], symbols: Optional[List[str]], max_symbols: Optional[int]) -> List[str]:
    """Determine which symbols this call should touch (lane-aware)."""
    keys = [s for s in rolling.keys() if isinstance(s, str) and not s.startswith("_")]
    if isinstance(symbols, list) and symbols:
        wanted = {str(s).strip().upper() for s in symbols if str(s).strip()}
        keys = [s for s in keys if str(s).upper() in wanted]
    keys.sort()
    if max_symbols is not None:
        try:
            keys = keys[: max(0, int(max_symbols))]
        except Exception:
            pass
    return keys


def apply_intraday_policy(
    cfg: PolicyConfig | None = None,
    *,
    symbols: Optional[List[str]] = None,
    max_symbols: int | None = None,
    max_positions: int | None = None,
    rolling_override: Dict[str, Any] | None = None,
    save: bool = True,
    out_key: str = "policy_dt",
    **_kwargs: Any,
) -> Dict[str, Any]:
    cfg = _apply_env_overrides(cfg or PolicyConfig())
    rolling = rolling_override if isinstance(rolling_override, dict) else _read_rolling()
    if not rolling:
        log("[policy_dt] ⚠️ rolling empty.")
        return {"symbols": 0, "updated": 0}

    # normalize max_positions
    try:
        max_positions_n = int(max_positions) if max_positions is not None else None
        if max_positions_n is not None and max_positions_n <= 0:
            max_positions_n = None
    except Exception:
        max_positions_n = None

    global_regime = _get_global_regime(rolling)
    regime_label_raw = str(global_regime.get("label") or "unknown")
    regime_label = _regime_for_policy(regime_label_raw)

    # Phase 5: load playbook profile (soft gate knobs + strategy weights)
    dt_profile = {}
    try:
        dt_profile = load_dt_profile(regime_label_raw)
        cfg = _apply_profile_overrides(cfg, dt_profile)
        # Drop breadcrumb for debugging/UI
        gdt = rolling.get("_GLOBAL_DT") if isinstance(rolling.get("_GLOBAL_DT"), dict) else {}
        gdt["dt_playbook_profile"] = {"name": str(dt_profile.get("label") or ""), "ts": _utc_now_iso()}
        rolling["_GLOBAL_DT"] = gdt
    except Exception:
        dt_profile = {}

    # Choose which symbols to touch (lane-aware)
    lane_syms = _lane_symbols(rolling, symbols, max_symbols)

    # Phase 0: hard risk rails can force a global stand_down for the session.
    g = rolling.get("_GLOBAL_DT") if isinstance(rolling.get("_GLOBAL_DT"), dict) else {}
    if bool(g.get("stand_down")):
        reason = str(g.get("stand_down_reason") or (g.get("risk_rails_dt") or {}).get("reason") or "stand_down")
        updated = 0
        # Safety: global stand-down updates ALL symbols (not just lane)
        for sym, node_raw in list(rolling.items()):
            if str(sym).startswith("_"):
                continue
            if not isinstance(node_raw, dict):
                continue
            node = ensure_symbol_node(rolling, sym)
            node[out_key] = {
                "action": "STAND_DOWN",
                "intent": "STAND_DOWN",
                "confidence": 0.0,
                "p_hit": 0.0,
                "score": 0.0,
                "trade_gate": False,
                "reason": f"risk_rails={reason}",
                "ts": _utc_now_iso(),
                "_state": (node.get(out_key, {}) or {}).get("_state") if isinstance(node.get(out_key), dict) else {},
            }
            rolling[sym] = node
            updated += 1
        if save:
            save_rolling(rolling)
        log(f"[policy_dt] 🛑 global stand_down (risk rails): updated {updated} symbols.")
        return {"symbols": len(rolling), "updated": updated, "stand_down": True, "reason": reason}

    # Phase 2.5: time-of-day micro-regime gating (lunch, closed, etc.)
    micro = _get_micro_regime(rolling)
    ignore_micro = str((__import__("os").getenv("DT_IGNORE_MICRO_REGIME", "") or "")).strip().lower() in {"1", "true", "yes", "y"}
    if micro and not ignore_micro:
        allow = bool(micro.get("allow_trading"))
        label = str(micro.get("label") or "")
        if not allow:
            updated = 0
            # Safety: micro-regime stand-down updates ALL symbols (not just lane)
            for sym, node_raw in list(rolling.items()):
                if str(sym).startswith("_"):
                    continue
                if not isinstance(node_raw, dict):
                    continue
                node = ensure_symbol_node(rolling, sym)
                node[out_key] = {
                    "action": "STAND_DOWN",
                    "intent": "STAND_DOWN",
                    "confidence": 0.0,
                    "p_hit": 0.0,
                    "score": 0.0,
                    "trade_gate": False,
                    "reason": f"micro_regime={label} stand_down",
                    "ts": _utc_now_iso(),
                    "_state": {
                        "prev_action": str((node.get(out_key) or {}).get("action") or "HOLD").upper(),
                        "pending_action": "",
                        "pending_count": 0,
                        "last_edge": 0.0,
                        "last_conf": 0.0,
                    },
                }
                rolling[sym] = node
                updated += 1

            if save:
                save_rolling(rolling)
            log(f"[policy_dt] 💤 micro-regime stand_down ({label}); updated {updated} symbols.")
            return {"symbols": len(rolling), "updated": updated, "micro_regime": label}

    # Phase 4: meta-controller daily plan
    plan = _get_daily_plan(rolling)
    allowed_bots = plan.get("enabled_bots") if isinstance(plan, dict) else None
    bot_weights = plan.get("bot_weights") if isinstance(plan, dict) else None

    # Phase 5: profile strategy weights bias bot selection (multiply if daily plan already provided weights)
    try:
        prof_w = (dt_profile or {}).get("strategy_weights")
        if isinstance(prof_w, dict) and prof_w:
            if isinstance(bot_weights, dict) and bot_weights:
                merged = dict(bot_weights)
                for k, v in prof_w.items():
                    try:
                        merged[str(k).upper()] = float(merged.get(str(k).upper(), 1.0)) * float(v)
                    except Exception:
                        pass
                bot_weights = merged
            else:
                bot_weights = {str(k).upper(): float(v) for k, v in prof_w.items()}
    except Exception:
        pass
    universe = plan.get("universe") if isinstance(plan, dict) else None
    universe_set = {str(s).upper() for s in universe} if isinstance(universe, list) and universe else None
    allow_model_fallback = bool(plan.get("allow_model_fallback")) if isinstance(plan, dict) else False
    risk_mode = str(plan.get("risk_mode") or "").upper() if isinstance(plan, dict) else ""

    # Risk mode adjusts aggressiveness (without touching the model itself)
    if risk_mode == "CONSERVATIVE":
        cfg.min_confidence = max(cfg.min_confidence, 0.38)
        cfg.buy_threshold = max(cfg.buy_threshold, 0.15)
        cfg.sell_threshold = min(cfg.sell_threshold, -0.15)
    elif risk_mode == "AGGRESSIVE":
        cfg.min_confidence = min(cfg.min_confidence, 0.26)
        cfg.buy_threshold = min(cfg.buy_threshold, 0.10)
        cfg.sell_threshold = max(cfg.sell_threshold, -0.10)

    updated = 0
    for sym in lane_syms:
        node_raw = rolling.get(sym)
        if not isinstance(node_raw, dict):
            continue

        # Phase 4: universe focus (skip low-quality symbols)
        sym_u = str(sym).upper()
        if universe_set is not None and sym_u not in universe_set:
            node = ensure_symbol_node(rolling, sym)
            node[out_key] = {
                "action": "HOLD",
                "intent": "HOLD",
                "confidence": 0.0,
                "p_hit": 0.0,
                "score": 0.0,
                "trade_gate": False,
                "reason": "out_of_universe",
                "ts": _utc_now_iso(),
                "_state": (node.get(out_key, {}) or {}).get("_state") if isinstance(node.get(out_key), dict) else {},
            }
            rolling[sym] = node
            updated += 1
            continue

        node = ensure_symbol_node(rolling, sym)
        ctx = node.get("context_dt") or {}
        if not isinstance(ctx, dict):
            ctx = {}

        feats = node.get("features_dt") if isinstance(node.get("features_dt"), dict) else {}

        # ------------------------------------------------------------
        # Phase 3: Strategy-first policy (uses features_dt + levels_dt)
        # Fallback to model-based policy if no viable setup.
        # ------------------------------------------------------------
        setup = None
        if select_best_setup is not None:
            try:
                micro_label = str((micro or {}).get("label") or "").upper() if isinstance(micro, dict) else ""
                setup = select_best_setup(
                    str(sym).upper(),
                    node,
                    rolling=rolling,
                    micro=micro_label,
                    allowed_bots=allowed_bots if isinstance(allowed_bots, list) else None,
                    bot_weights=bot_weights if isinstance(bot_weights, dict) else None,
                )
            except Exception:
                setup = None

        # Maintain squeeze memory for Phase 3 (release detection).
        try:
            sq_on = bool(float(feats.get("squeeze_on") or 0.0) >= 0.5) if isinstance(feats, dict) else False
            st = node.get("_squeeze_state")
            st = st if isinstance(st, dict) else {}
            st["prev_squeeze_on"] = bool(sq_on)
            st["ts"] = _utc_now_iso()
            node["_squeeze_state"] = st
        except Exception:
            pass

        if isinstance(setup, dict) and str(setup.get("side") or "").upper() in {"BUY", "SELL"}:

            # Phase 5: apply strategy weights from the active playbook
            try:
                bname = str(setup.get("bot") or setup.get("strategy") or "").upper()
                w = strategy_weight(dt_profile or {}, bname) if bname else 1.0
                if w != 1.0:
                    setup = dict(setup)
                    setup["score"] = float(setup.get("score") or 0.0) * float(w)
                    # tiny confidence nudge (bounded)
                    setup["confidence"] = max(0.0, min(1.0, float(setup.get("confidence") or 0.0) * (0.90 + 0.10 * float(w))))
                    setup["reason"] = f"{setup.get('reason') or ''} (w={w:.2f})".strip()
            except Exception:
                pass
            proposed_intent = str(setup.get("side") or "HOLD").upper()
            has_position = _has_position(node)
            if proposed_intent == "BUY" and has_position:
                proposed_intent = "HOLD"
            elif proposed_intent == "SELL" and not has_position:
                proposed_intent = "HOLD"
            base_conf = float(setup.get("confidence") or 0.0)
            edge = float(setup.get("score") or 0.0) / 100.0  # score -> pseudo-edge scale

            trend, vol_bkt = _trend_and_vol(ctx)

            conf_adj, adj_detail = _adjust_conf(base_conf, proposed_intent, trend, vol_bkt, regime_label, cfg)

            stabilized_action, new_state, hyst_note = _stabilize_with_hysteresis(
                node=node,
                proposed=proposed_intent,
                edge=edge,
                conf=conf_adj,
                cfg=cfg,
                policy_key=out_key,
            )

            action = stabilized_action
            conf_final = float(conf_adj) if action in {"BUY", "SELL"} else 0.0
            trade_gate = bool(action in {"BUY", "SELL"} and conf_final >= cfg.min_confidence)

            # Phase 8: news/event risk seatbelt
            risk_penalty = 1.0
            risk_note = ""
            bot_name = str(setup.get("bot") or "").upper() or "UNKNOWN"
            micro_label = str((micro or {}).get("label") or "").upper() if isinstance(micro, dict) else ""

            try:
                if assess_symbol_risk is not None:
                    risk = assess_symbol_risk(symbol=str(sym).upper(), features_dt=feats, now_utc=None)
                    if isinstance(risk, dict):
                        node["risk_dt"] = risk
                        if bool(risk.get("stand_down")):
                            action = "STAND_DOWN"
                        risk_penalty = float(risk.get("penalty") or 1.0)
                        risk_note = ",".join([str(x) for x in (risk.get("reasons") or [])][:3])
            except Exception:
                risk_penalty = 1.0

            # Phase 9: researcher rules
            try:
                if bot_allowed is not None:
                    ok, why = bot_allowed(bot=bot_name, regime=regime_label, micro=micro_label, features_dt=feats)
                    if not ok:
                        node["execution_plan_dt"] = {}
                        node[out_key] = {
                            "action": "HOLD",
                            "intent": "HOLD",
                            "confidence": 0.0,
                            "p_hit": 0.0,
                            "score": 0.0,
                            "trade_gate": False,
                            "reason": f"filtered_{why}",
                            "ts": _utc_now_iso(),
                            "bot": bot_name,
                            "_state": new_state,
                        }
                        rolling[sym] = node
                        updated += 1
                        continue
            except Exception:
                pass

            # Phase 7: calibrated probability-of-hit
            p_hit = conf_final
            try:
                if get_phit is not None:
                    ph = get_phit(bot=bot_name, regime_label=regime_label, base_conf=conf_final)
                    if ph is not None:
                        p_hit = float(ph)
            except Exception:
                p_hit = conf_final

            # Apply risk penalty to BOTH confidence and p_hit (seatbelt, not steering wheel)
            # IMPORTANT: we keep a separate "signal" confidence even when action stabilizes to HOLD.
            signal_conf = float(max(0.0, min(cfg.max_confidence, conf_final * risk_penalty)))
            signal_phit = float(max(0.0, min(1.0, p_hit * risk_penalty)))

            if action == "STAND_DOWN":
                conf_final = 0.0
                p_hit = 0.0
                signal_conf = 0.0
                signal_phit = 0.0
                trade_gate = False
            else:
                # Policy confidence is only "live" when we actually take BUY/SELL.
                conf_final = float(signal_conf) if action in {"BUY", "SELL"} else 0.0
                p_hit = float(signal_phit) if action in {"BUY", "SELL"} else 0.0
                trade_gate = bool(action in {"BUY", "SELL"} and conf_final >= cfg.min_confidence)

            # Compute expected R using last price, stop, tp if present.
            expected_r = 1.0
            try:
                last_px = float(feats.get("last_price") or 0.0)
                risk_d = setup.get("risk") if isinstance(setup.get("risk"), dict) else {}
                stop = risk_d.get("stop")
                tp = risk_d.get("take_profit")
                if last_px > 0 and stop is not None and tp is not None:
                    denom = abs(float(last_px) - float(stop))
                    num = abs(float(tp) - float(last_px))
                    if denom > 1e-9:
                        expected_r = float(max(0.2, min(5.0, num / denom)))
            except Exception:
                expected_r = 1.0

            # Phase 4: NO_TRADE / PROBE / ENTER / PRESS
            tier = "NO_TRADE"
            try:
                setup_score = float(setup.get("score") or 0.0)
            except Exception:
                setup_score = 0.0

            if action == "STAND_DOWN":
                tier = "NO_TRADE"
            elif bool(trade_gate):
                tier = "ENTER"
                if (
                    float(signal_phit) >= float(cfg.press_min_phit)
                    and float(expected_r) >= float(cfg.press_min_expected_r)
                    and float(signal_conf) >= float(cfg.min_confidence + cfg.press_conf_extra)
                ):
                    tier = "PRESS"
            else:
                if (
                    bool(cfg.probe_enabled)
                    and str(proposed_intent).upper() in {"BUY", "SELL"}
                    and float(signal_conf) >= float(cfg.probe_min_signal_conf)
                    and abs(float(setup_score)) >= float(cfg.probe_min_abs_score)
                ):
                    tier = "PROBE"

            # Add derived safety-friendly scalars if missing
            try:
                last_px = float(feats.get("last_price") or 0.0)
                atr = float(feats.get("atr_14") or 0.0)
                if last_px > 0 and "atr_pct" not in feats:
                    feats = dict(feats)
                    feats["atr_pct"] = float(atr / last_px) if atr > 0 else 0.0
            except Exception:
                pass

            node["execution_plan_dt"] = {
                "bot": str(setup.get("bot") or "").upper(),
                "side": str(setup.get("side") or "").upper(),
                "tier": str(tier),
                "confidence": float(setup.get("confidence") or 0.0),
                "base_conf": float(base_conf),
                "p_hit": float(signal_phit),
                "expected_r": float(expected_r),
                "entry_features": {
                    "rel_volume": float((feats or {}).get("rel_volume") or 0.0) if isinstance(feats, dict) else 0.0,
                    "vwap_dist": float((feats or {}).get("vwap_dist") or 0.0) if isinstance(feats, dict) else 0.0,
                    "squeeze_on": float((feats or {}).get("squeeze_on") or 0.0) if isinstance(feats, dict) else 0.0,
                    "atr_pct": float((feats or {}).get("atr_pct") or 0.0) if isinstance(feats, dict) else 0.0,
                    "spread_pct": float((feats or {}).get("spread_pct") or 0.0) if isinstance(feats, dict) else 0.0,
                    "news_shock_score": float((feats or {}).get("news_shock_score") or 0.0) if isinstance(feats, dict) else 0.0,
                },
                "score": float(setup.get("score") or 0.0),
                "entry": setup.get("entry") or {"type": "MKT"},
                "risk": setup.get("risk") or {},
                "reason": str(setup.get("reason") or ""),
                "ts": setup.get("ts") or _utc_now_iso(),
            }

            reason = (
                f"bot={node['execution_plan_dt'].get('bot')} {node['execution_plan_dt'].get('reason')}; "
                f"trend={trend} vol={vol_bkt} regime={regime_label}; adj={adj_detail}; hyst={hyst_note}; "
                f"risk={(risk_note or 'ok')}"
            )

            node[out_key] = {
                "action": action,
                "intent": action,
                "confidence": float(conf_final),
                "p_hit": float(p_hit),
                "tier": str(tier),
                "signal_intent": str(proposed_intent),
                "signal_confidence": float(signal_conf),
                "signal_p_hit": float(signal_phit),
                "score": float(setup.get("score") or 0.0),
                "trade_gate": bool(trade_gate),
                "reason": reason,
                "ts": _utc_now_iso(),
                "bot": node["execution_plan_dt"].get("bot"),
                "_state": new_state,
            }
            
            # Log feature importance for this policy decision
            try:
                if get_feature_tracker is not None and isinstance(feats, dict) and feats:
                    tracker = get_feature_tracker()
                    tracker.log_prediction(
                        symbol=sym,
                        features_dict=feats,
                        prediction=action,
                        confidence=float(conf_final),
                        metadata={"cycle": "policy", "bot": node["execution_plan_dt"].get("bot")}
                    )
            except Exception:
                pass
            
            rolling[sym] = node
            updated += 1
            continue

        # Phase 4: optionally disable model fallback entirely
        if not allow_model_fallback:
            node[out_key] = {
                "action": "HOLD",
                "intent": "HOLD",
                "confidence": 0.0,
                "p_hit": 0.0,
                "score": 0.0,
                "trade_gate": False,
                "reason": "no_strategy_setup",
                "ts": _utc_now_iso(),
                "_state": (node.get(out_key, {}) or {}).get("_state") if isinstance(node.get(out_key), dict) else {},
            }
            rolling[sym] = node
            updated += 1
            continue

        # Require model output
        _, proba = _extract_prediction(node)
        if not proba:
            # still write a safe HOLD so downstream doesn't see stale data
            node[out_key] = {
                "action": "HOLD",
                "intent": "HOLD",
                "confidence": 0.0,
                "p_hit": 0.0,
                "score": 0.0,
                "trade_gate": False,
                "reason": "missing_model_proba",
                "ts": _utc_now_iso(),
                "_state": (node.get(out_key, {}) or {}).get("_state") if isinstance(node.get(out_key), dict) else {},
            }
            rolling[sym] = node
            updated += 1
            continue

        p_buy = float(proba.get("BUY", 0.0))
        p_hold = float(proba.get("HOLD", 0.0))
        p_sell = float(proba.get("SELL", 0.0))

        proposed_intent, base_conf, edge = _raw_intent_from_edge(p_buy, p_hold, p_sell, cfg)
        has_position = _has_position(node)
        if proposed_intent == "BUY" and has_position:
            proposed_intent = "HOLD"
        elif proposed_intent == "SELL" and not has_position:
            proposed_intent = "HOLD"
        trend, vol_bkt = _trend_and_vol(ctx)

        conf_adj, adj_detail = _adjust_conf(base_conf, proposed_intent, trend, vol_bkt, regime_label, cfg)

        r = regime_label.lower()
        if cfg.stand_down_in_crash and r in {"crash", "stress"}:
            action = "STAND_DOWN"
            conf_final = 0.0
            score = 0.0
            trade_gate = False
            reason = f"regime={regime_label} stand_down"
            new_state = (node.get(out_key) or {}).get("_state") if isinstance(node.get(out_key), dict) else {}
            p_hit = 0.0
        elif cfg.stand_down_in_unknown_regime and r in {"unknown"}:
            action = "STAND_DOWN"
            conf_final = 0.0
            score = 0.0
            trade_gate = False
            reason = f"regime={regime_label} stand_down_unknown"
            new_state = (node.get(out_key) or {}).get("_state") if isinstance(node.get(out_key), dict) else {}
            p_hit = 0.0
        else:
            score = float(edge) * float(conf_adj)

            if proposed_intent in {"BUY", "SELL"} and conf_adj < cfg.min_confidence:
                proposed_intent = "HOLD"

            stabilized_action, new_state, hyst_note = _stabilize_with_hysteresis(
                node=node,
                proposed=proposed_intent,
                edge=edge,
                conf=conf_adj,
                cfg=cfg,
                policy_key=out_key,
            )

            action = stabilized_action
            conf_final = float(conf_adj) if action in {"BUY", "SELL"} else 0.0
            trade_gate = bool(action in {"BUY", "SELL"} and conf_final >= cfg.min_confidence)

            # Phase 7: calibrated probability-of-hit
            p_hit = conf_final
            try:
                if get_phit is not None:
                    ph = get_phit(bot="MODEL", regime_label=regime_label, base_conf=conf_final)
                    if ph is not None:
                        p_hit = float(ph)
            except Exception:
                p_hit = conf_final

            reason = (
                f"edge={edge:.3f} p_buy={p_buy:.3f} p_sell={p_sell:.3f} p_hold={p_hold:.3f}; "
                f"trend={trend} vol={vol_bkt} regime={regime_label}; "
                f"adj={adj_detail}; hyst={hyst_note}"
            )

        node[out_key] = {
            "action": action,
            "intent": action,
            "confidence": float(conf_final),
            "p_hit": float(p_hit),
            "score": float(score),
            "trade_gate": bool(trade_gate),
            "reason": reason,
            "ts": _utc_now_iso(),
            "_state": new_state if isinstance(new_state, dict) else {},
        }
        
        # Log feature importance for model-based policy decision
        try:
            if get_feature_tracker is not None and isinstance(feats, dict) and feats:
                tracker = get_feature_tracker()
                tracker.log_prediction(
                    symbol=sym,
                    features_dict=feats,
                    prediction=action,
                    confidence=float(conf_final),
                    metadata={"cycle": "policy", "model": "ensemble"}
                )
        except Exception:
            pass
        
        rolling[sym] = node
        updated += 1

    # ------------------------------------------------------------
    # Hard cap: allow only top N trade candidates by |score|
    # Note: applied only within the lane symbols touched by this call.
    # ------------------------------------------------------------
    capped = 0
    if max_positions_n is not None:
        candidates: List[Tuple[str, float]] = []
        for sym in lane_syms:
            node = rolling.get(sym)
            if not isinstance(node, dict):
                continue
            p = node.get(out_key)
            if not isinstance(p, dict):
                continue
            if p.get("trade_gate") is True and str(p.get("action")).upper() in {"BUY", "SELL"}:
                candidates.append((sym, abs(float(p.get("score") or 0.0))))

        candidates.sort(key=lambda t: t[1], reverse=True)
        for sym, _ in candidates[max_positions_n:]:
            node = rolling.get(sym)
            if not isinstance(node, dict):
                continue
            p = node.get(out_key)
            if not isinstance(p, dict):
                continue

            st = p.get("_state")
            if not isinstance(st, dict):
                st = {}
            st["capped_action"] = str(p.get("action") or "").upper()
            st["capped_score"] = float(p.get("score") or 0.0)
            p["_state"] = st

            p["action"] = "HOLD"
            p["intent"] = "HOLD"
            p["trade_gate"] = False
            p["confidence"] = 0.0
            p["p_hit"] = 0.0
            p["score"] = 0.0
            p["reason"] = (str(p.get("reason") or "") + f"; cap=max_positions({max_positions_n})").strip()
            p["ts"] = _utc_now_iso()

            node[out_key] = p
            rolling[sym] = node
            capped += 1

    # ------------------------------------------------------------
    # Phase 5.5: Portfolio heat manager (sector/exposure caps)
    # ------------------------------------------------------------
    heat_summary = None
    try:
        if apply_portfolio_heat_gates is not None:
            rolling, heat_summary = apply_portfolio_heat_gates(rolling, out_key=out_key)
            gdt = rolling.get("_GLOBAL_DT") if isinstance(rolling.get("_GLOBAL_DT"), dict) else {}
            if isinstance(heat_summary, dict):
                gdt["heat_dt"] = heat_summary
            rolling["_GLOBAL_DT"] = gdt
    except Exception:
        heat_summary = None

    if save:
        save_rolling(rolling)
    
    # Validate rolling structure before returning (PR #4)
    try:
        from dt_backend.core.schema_validator_dt import validate_rolling, ValidationError
        validate_rolling(rolling)
    except ValidationError as e:
        log(f"[policy_dt] ❌ Schema validation failed: {e}")
        # Log but don't raise - allow graceful degradation
    except Exception as e:
        log(f"[policy_dt] ⚠️ Validation error: {e}")

    extra = f", capped={capped}, max_positions={max_positions_n}" if max_positions_n is not None else ""
    lane_note = f", lane_symbols={len(lane_syms)}" if isinstance(lane_syms, list) else ""
    log(f"[policy_dt] ✅ updated policy_dt for {updated} symbols (regime={regime_label}){extra}{lane_note}.")
    return {
        "symbols": len(rolling),
        "updated": updated,
        "regime": regime_label,
        "capped": capped,
        "max_positions": max_positions_n,
        "lane_symbols": len(lane_syms),
    }
