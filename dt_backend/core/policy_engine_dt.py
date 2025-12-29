# dt_backend/core/policy_engine_dt.py — v2.0
"""
Intraday policy engine for dt_backend.

Writes:
    rolling[sym]["policy_dt"] = {
        "action": "BUY"|"SELL"|"HOLD"|"STAND_DOWN",
        "intent":  same as action (legacy compatibility),
        "confidence": 0.0..0.99,
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
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, Optional

from .data_pipeline_dt import _read_rolling, save_rolling, ensure_symbol_node, log


@dataclass
class PolicyConfig:
    # Signal edge thresholds (p_buy - p_sell)
    buy_threshold: float = 0.12
    sell_threshold: float = -0.12

    # Minimum confidence for acting (after adjustments)
    min_confidence: float = 0.30

    # Volatility penalties
    vol_penalty_high: float = 0.65
    vol_penalty_medium: float = 0.85

    # Trend boosts when aligned
    trend_boost_strong: float = 1.25
    trend_boost_mild: float = 1.10

    # Regime effects
    chop_penalty: float = 0.90
    bear_buy_penalty: float = 0.85
    bull_sell_penalty: float = 0.90

    # Stability / anti-flip
    hysteresis_hold_bias: float = 0.03     # make HOLD "sticky" by requiring extra edge to flip
    min_edge_to_flip: float = 0.06         # require at least this edge magnitude to flip direction
    confirmations_to_flip: int = 2         # require N consecutive signals before switching BUY<->SELL
    max_confidence: float = 0.99

    # Safety gate: in crash/stress regimes, optionally stand down
    stand_down_in_unknown_regime: bool = False
    stand_down_in_crash: bool = True


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
    g = rolling.get("_GLOBAL_DT") or {}
    if not isinstance(g, dict):
        return {"label": "unknown", "breadth_up": 0.5}
    reg = g.get("regime") or {}
    if not isinstance(reg, dict):
        return {"label": "unknown", "breadth_up": 0.5}
    return reg


def _extract_prediction(node: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Best-effort extraction of dt predictions.

    Expected:
      node["predictions_dt"] = {"label": "...", "proba": {"BUY":0.5,"HOLD":0.3,"SELL":0.2}}
    Also tolerates:
      node["predictions"] = ...
      "probs" instead of "proba"
    """
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

    # normalize
    total = sum(proba.values())
    if total > 0:
        proba = {k: v / total for k, v in proba.items()}

    return (label.upper() if isinstance(label, str) else None), proba


def _trend_and_vol(ctx: Dict[str, Any]) -> Tuple[str, str]:
    trend = str(ctx.get("intraday_trend") or "").strip() or "flat"
    vol_bkt = str(ctx.get("vol_bucket") or "").strip() or "low"
    return trend, vol_bkt


def _raw_intent_from_edge(p_buy: float, p_hold: float, p_sell: float, cfg: PolicyConfig) -> Tuple[str, float, float]:
    """
    Returns (intent, base_conf, edge).
    base_conf is max(p_buy, p_sell) because HOLD shouldn't trigger action by itself.
    """
    edge = p_buy - p_sell
    base_conf = max(p_buy, p_sell)

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
    detail = []

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
) -> Tuple[str, Dict[str, Any], str]:
    """
    Use per-symbol memory to prevent flip-flopping.
    Stores a small internal state under policy_dt["_state"].

    Rules:
      - HOLD is sticky: needs extra edge to exit HOLD
      - BUY<->SELL flips require:
          * abs(edge) >= min_edge_to_flip
          * confirmations_to_flip consecutive proposals
    """
    policy_prev = node.get("policy_dt") or {}
    prev_action = str((policy_prev.get("action") or policy_prev.get("intent") or "HOLD")).upper()
    state = policy_prev.get("_state") or {}
    if not isinstance(state, dict):
        state = {}

    pending = str(state.get("pending_action") or "").upper()
    pending_count = int(state.get("pending_count") or 0)

    # default: accept
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
        # reset pending when we accept
        state["pending_action"] = ""
        state["pending_count"] = 0
    else:
        state["pending_action"] = pending
        state["pending_count"] = pending_count

    state["prev_action"] = prev_action
    state["last_edge"] = float(edge)
    state["last_conf"] = float(conf)

    return final_action, state, note


def apply_intraday_policy(
    cfg: PolicyConfig | None = None,
    *,
    max_positions: int | None = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    cfg = cfg or PolicyConfig()
    rolling = _read_rolling()
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
    regime_label = str(global_regime.get("label") or "unknown")

    updated = 0
    for sym, node_raw in list(rolling.items()):
        if str(sym).startswith("_"):
            continue
        if not isinstance(node_raw, dict):
            continue

        node = ensure_symbol_node(rolling, sym)
        ctx = node.get("context_dt") or {}
        if not isinstance(ctx, dict):
            ctx = {}

        # Require model output
        _, proba = _extract_prediction(node)
        if not proba:
            continue

        p_buy = float(proba.get("BUY", 0.0))
        p_hold = float(proba.get("HOLD", 0.0))
        p_sell = float(proba.get("SELL", 0.0))

        proposed_intent, base_conf, edge = _raw_intent_from_edge(p_buy, p_hold, p_sell, cfg)
        trend, vol_bkt = _trend_and_vol(ctx)

        conf_adj, adj_detail = _adjust_conf(base_conf, proposed_intent, trend, vol_bkt, regime_label, cfg)

        r = regime_label.lower()
        if cfg.stand_down_in_crash and r in {"crash", "stress"}:
            action = "STAND_DOWN"
            conf_final = 0.0
            score = 0.0
            trade_gate = False
            reason = f"regime={regime_label} stand_down"
        elif cfg.stand_down_in_unknown_regime and r in {"unknown"}:
            action = "STAND_DOWN"
            conf_final = 0.0
            score = 0.0
            trade_gate = False
            reason = f"regime={regime_label} stand_down_unknown"
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
            )

            action = stabilized_action
            conf_final = float(conf_adj) if action in {"BUY", "SELL"} else 0.0
            trade_gate = bool(action in {"BUY", "SELL"} and conf_final >= cfg.min_confidence)

            reason = (
                f"edge={edge:.3f} p_buy={p_buy:.3f} p_sell={p_sell:.3f} p_hold={p_hold:.3f}; "
                f"trend={trend} vol={vol_bkt} regime={regime_label}; "
                f"adj={adj_detail}; hyst={hyst_note}"
            )

            node["policy_dt"] = {
                "action": action,
                "intent": action,
                "confidence": float(conf_final),
                "score": float(score),
                "trade_gate": bool(trade_gate),
                "reason": reason,
                "ts": _utc_now_iso(),
                "_state": new_state,
            }
            rolling[sym] = node
            updated += 1
            continue

        # STAND_DOWN path / safety
        node["policy_dt"] = {
            "action": action,
            "intent": action,
            "confidence": float(conf_final),
            "score": float(score),
            "trade_gate": bool(trade_gate),
            "reason": reason,
            "ts": _utc_now_iso(),
            "_state": {
                "prev_action": str((node.get("policy_dt") or {}).get("action") or "HOLD").upper(),
                "pending_action": "",
                "pending_count": 0,
                "last_edge": 0.0,
                "last_conf": 0.0,
            },
        }
        rolling[sym] = node
        updated += 1

    # ------------------------------------------------------------
    # Hard cap: allow only top N trade candidates by |score|
    # (Safety: if capped out, force HOLD + trade_gate False)
    # ------------------------------------------------------------
    capped = 0
    if max_positions_n is not None:
        candidates = []
        for sym, node in rolling.items():
            if not isinstance(sym, str) or sym.startswith("_"):
                continue
            if not isinstance(node, dict):
                continue
            p = node.get("policy_dt")
            if not isinstance(p, dict):
                continue
            if p.get("trade_gate") is True and str(p.get("action")).upper() in {"BUY", "SELL"}:
                candidates.append((sym, abs(float(p.get("score") or 0.0))))

        candidates.sort(key=lambda t: t[1], reverse=True)
        keep = set(sym for sym, _ in candidates[:max_positions_n])

        for sym, _ in candidates[max_positions_n:]:
            node = rolling.get(sym)
            if not isinstance(node, dict):
                continue
            p = node.get("policy_dt")
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
            p["score"] = 0.0
            p["reason"] = (str(p.get("reason") or "") + f"; cap=max_positions({max_positions_n})").strip()
            p["ts"] = _utc_now_iso()

            node["policy_dt"] = p
            rolling[sym] = node
            capped += 1

    save_rolling(rolling)
    extra = f", capped={capped}, max_positions={max_positions_n}" if max_positions_n is not None else ""
    log(f"[policy_dt] ✅ updated policy_dt for {updated} symbols (regime={regime_label}){extra}.")
    return {"symbols": len(rolling), "updated": updated, "regime": regime_label, "capped": capped, "max_positions": max_positions_n}