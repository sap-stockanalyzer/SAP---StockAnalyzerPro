# backend/core/policy_engine.py — v1.2 (Regression Edition + AION Brain)
"""
AION Unified Policy Engine — Regression Edition

Moves AION from classification-era heuristics to a clean,
return-driven, confidence-aware decision architecture.

Key drivers:
    • predicted_return (primary signal)
    • score (secondary normalization)
    • confidence (calibrated probability-ish)
    • regime (bull / bear / panic / chop)
    • news + social sentiment (from context_state)
    • volatility
    • drift (continuous learning feedback; horizon-aware)
    • sector performance health (from continuous_learning meta)

UPDATED (v1.2):
    ✅ Adds AION brain (global behavioral memory):
         - reads PATHS["brain"] (aion_brain.json.gz)
         - modulates confidence/risk/exposure in a traceable way
         - supports optional regime modifiers in AION brain meta
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Dict, Any, Optional

from backend.core.data_pipeline import (
    _read_rolling,
    _read_brain,
    _read_aion_brain,     # ✅ NEW
    save_rolling,
    safe_float,
    log,
)
from backend.core.regime_detector import detect_regime


# ======================================================================
# Horizon weights (more weight to predicted_return)
# ======================================================================
HORIZON_WEIGHTS = {
    "1d": 0.25,
    "3d": 0.25,
    "1w": 0.20,
    "2w": 0.15,
    "4w": 0.10,
    "13w": 0.03,
    "26w": 0.015,
    "52w": 0.01,
}

_PRIMARY_DRIFT_H = "1w"


# ======================================================================
# AION BRAIN DEFAULTS (behavioral knobs)
# ======================================================================
_DEFAULT_AION_META = {
    # Multiplies final policy confidence (bounded downstream)
    "confidence_bias": 1.0,
    # Multiplies computed risk sizing (bounded downstream)
    "risk_bias": 1.0,
    # Multiplies exposure_scale (aggressiveness)
    "aggressiveness": 1.0,
    # Optional regime overrides: {"bull": {...}, "bear": {...}, "panic": {...}, "chop": {...}}
    "regime_mods": {},
}


def _load_aion_brain_meta() -> Dict[str, Any]:
    """
    AION brain schema:
      {
        "_meta": {
          "confidence_bias": 1.0,
          "risk_bias": 1.0,
          "aggressiveness": 1.0,
          "regime_mods": { ... },
          "updated_at": "...",
          ...
        }
      }
    """
    ab = _read_aion_brain() or {}
    meta = ab.get("_meta", {}) if isinstance(ab, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    out = dict(_DEFAULT_AION_META)
    out.update(meta)

    # sanitize
    out["confidence_bias"] = float(max(0.70, min(1.30, safe_float(out.get("confidence_bias", 1.0)) or 1.0)))
    out["risk_bias"] = float(max(0.60, min(1.40, safe_float(out.get("risk_bias", 1.0)) or 1.0)))
    out["aggressiveness"] = float(max(0.60, min(1.50, safe_float(out.get("aggressiveness", 1.0)) or 1.0)))
    if not isinstance(out.get("regime_mods"), dict):
        out["regime_mods"] = {}

    return out


def _aion_regime_mod(aion_meta: Dict[str, Any], regime: Optional[dict]) -> Dict[str, float]:
    lab = (regime or {}).get("label", "chop").lower()
    mods = aion_meta.get("regime_mods") or {}
    if not isinstance(mods, dict):
        return {"confidence_bias": 1.0, "risk_bias": 1.0, "aggressiveness": 1.0}

    rm = mods.get(lab) or mods.get(lab.upper()) or {}
    if not isinstance(rm, dict):
        return {"confidence_bias": 1.0, "risk_bias": 1.0, "aggressiveness": 1.0}

    return {
        "confidence_bias": float(max(0.70, min(1.30, safe_float(rm.get("confidence_bias", 1.0)) or 1.0))),
        "risk_bias": float(max(0.60, min(1.40, safe_float(rm.get("risk_bias", 1.0)) or 1.0))),
        "aggressiveness": float(max(0.60, min(1.50, safe_float(rm.get("aggressiveness", 1.0)) or 1.0))),
    }


# ======================================================================
# MULTI-HORIZON FUSION (REGRESSION)
# ======================================================================
def _fuse(preds: Dict[str, Any]) -> Dict[str, float]:
    if not preds or not isinstance(preds, dict):
        return {
            "ret": 0.0,
            "score": 0.0,
            "conf": 0.5,
            "short": 0.0,
            "mid": 0.0,
            "long": 0.0,
        }

    w_ret = w_score = w_conf = 0.0
    total_w = 0.0

    short_s, mid_s, long_s = [], [], []

    for h, w in HORIZON_WEIGHTS.items():
        block = preds.get(h)
        if not isinstance(block, dict):
            continue

        ret = safe_float(block.get("predicted_return", 0.0))
        score = safe_float(block.get("score", 0.0))
        conf = safe_float(block.get("confidence", 0.5))

        w_ret += ret * w
        w_score += score * w
        w_conf += conf * w
        total_w += w

        if h in ("1d", "3d"):
            short_s.append(score)
        elif h in ("1w", "2w"):
            mid_s.append(score)
        else:
            long_s.append(score)

    if total_w == 0.0:
        return {
            "ret": 0.0,
            "score": 0.0,
            "conf": 0.5,
            "short": 0.0,
            "mid": 0.0,
            "long": 0.0,
        }

    return {
        "ret": w_ret / total_w,
        "score": w_score / total_w,
        "conf": w_conf / total_w,
        "short": mean(short_s) if short_s else 0.0,
        "mid": mean(mid_s) if mid_s else 0.0,
        "long": mean(long_s) if long_s else 0.0,
    }


# ======================================================================
# CONTEXT EFFECTS (ALIGNED TO context_state.py v1.1)
# ======================================================================
def _ctx(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    context_state v1.1 writes:
      node["context"] = {
        "sentiment": <combined>,
        "volatility": <macro_vol alias>,
        "trend": "...",
        "news": {sentiment, impact_score, shock_score, ...},
        "social": {sentiment, heat_score, ...},
        ...
      }

    We treat ctx["sentiment"] as the blended base, then ALSO blend in
    explicit news/social blocks for richer modulation.
    """
    ctx = node.get("context") or {}
    if not isinstance(ctx, dict):
        ctx = {}

    news = ctx.get("news") or {}
    if not isinstance(news, dict):
        news = {}

    social = ctx.get("social") or node.get("social") or {}
    if not isinstance(social, dict):
        social = {}

    sentiment = safe_float(ctx.get("sentiment", 0.0))
    volatility = safe_float(ctx.get("volatility", ctx.get("macro_vol", 0.0)))
    trend = str(ctx.get("trend") or "neutral").lower()

    news_sent = safe_float(news.get("sentiment", 0.0))
    news_imp = safe_float(news.get("impact_score", 0.0))

    soc_sent = safe_float(social.get("sentiment", 0.0))
    soc_heat = safe_float(social.get("heat_score", 0.0))

    # volatility damping (up to -40%)
    vol_mult = 1.0 - min(max(volatility * 1.2, 0.0), 0.4)

    # sentiment boost (±20%)
    # sentiment already blends news/social, but we still include explicit components
    sent_raw = sentiment + 0.4 * news_sent + 0.4 * soc_sent
    sent_mult = 1.0 + max(min(sent_raw * 0.08, 0.20), -0.20)

    # news impact acceleration + social heat assist
    impact_mult = 1.0 + 0.08 * news_imp + 0.04 * math.tanh(soc_heat)

    # trend influence
    trend_mult = 1.10 if trend == "bullish" else 0.90 if trend == "bearish" else 1.0

    return {
        "vol_mult": float(vol_mult),
        "sent_mult": float(sent_mult),
        "impact_mult": float(impact_mult),
        "trend_mult": float(trend_mult),
        "volatility": float(volatility),
        "sentiment": float(sentiment),
        "news_sentiment": float(news_sent),
        "news_impact": float(news_imp),
        "social_sentiment": float(soc_sent),
        "social_heat": float(soc_heat),
        "trend": trend,
    }


# ======================================================================
# DRIFT, SECTOR, REGIME ADJUSTMENTS
# ======================================================================
def _regime_mult(regime: Optional[dict]) -> float:
    lab = (regime or {}).get("label", "chop").lower()
    if lab == "bull":
        return 1.15
    if lab == "bear":
        return 0.85
    if lab == "panic":
        return 0.75
    return 1.0


def _drift_mult(brain_meta: dict, brain_node: Dict[str, Any], horizon: str = _PRIMARY_DRIFT_H) -> float:
    """
    continuous_learning stores:
      - brain["_meta"]["horizon_drift"][h]["avg_drift"]  (global)
      - brain[sym]["horizon_perf"][h]["drift"]           (per-symbol)

    drift meaning:
      drift = mae_long - mae_short
        > 0 => recent improved
        < 0 => recent degraded

    We penalize degradation (negative drift).
    """
    hperf = brain_node.get("horizon_perf", {})
    if isinstance(hperf, dict):
        hp = hperf.get(horizon)
        if isinstance(hp, dict):
            d = safe_float(hp.get("drift", 0.0))
            if d < -0.010:
                return 0.75
            if d < -0.005:
                return 0.85
            return 1.0

    hd = (brain_meta.get("horizon_drift") or {}).get(horizon)
    if isinstance(hd, dict):
        d = safe_float(hd.get("avg_drift", 0.0))
        if d < -0.010:
            return 0.80
        if d < -0.005:
            return 0.90
        return 1.0

    return 1.0


def _sector_mult(sector: str, brain_meta: dict, horizon: str = _PRIMARY_DRIFT_H) -> float:
    """
    continuous_learning meta sector_perf[h] fields:
      n, mae, rmse, bias, avg_conf
    """
    if not sector:
        return 1.0

    sec_map = brain_meta.get("sector_perf") or {}
    sec = sec_map.get(str(sector).upper())
    if not isinstance(sec, dict):
        return 1.0

    hsec = sec.get(horizon)
    if not isinstance(hsec, dict):
        return 1.0

    mae = safe_float(hsec.get("mae", 0.0))
    avg_conf = safe_float(hsec.get("avg_conf", 0.0))

    mult = 1.0

    if avg_conf > 0.70:
        mult *= 1.05
    elif avg_conf < 0.55:
        mult *= 0.92

    if mae > 0.12:
        mult *= 0.85
    elif mae > 0.08:
        mult *= 0.92

    return float(max(0.80, min(1.20, mult)))


# ======================================================================
# INTENT DECISION
# ======================================================================
def _intent(pred_ret: float, final_score: float, regime: dict, ctx: dict) -> str:
    reg = (regime or {}).get("label", "chop").lower()
    trend = ctx.get("trend")

    if pred_ret >= 0.015:
        if reg != "bear":
            return "BUY"
        if reg == "bear" and pred_ret >= 0.025 and final_score > 0.06 and trend == "bullish":
            return "BUY"

    if pred_ret <= -0.012:
        return "SELL"
    if final_score <= -0.06:
        return "SELL"

    return "HOLD"


# ======================================================================
# POLICY FOR ONE SYMBOL
# ======================================================================
def _build(sym, node, preds, brain_node, regime, brain_meta):
    fused = _fuse(preds)
    ctx = _ctx(node)

    sector = node.get("sector") or (node.get("fundamentals") or {}).get("sector") or ""

    mult = (
        ctx["vol_mult"]
        * ctx["sent_mult"]
        * ctx["impact_mult"]
        * ctx["trend_mult"]
        * _drift_mult(brain_meta, brain_node, _PRIMARY_DRIFT_H)
        * _regime_mult(regime)
        * _sector_mult(sector, brain_meta, _PRIMARY_DRIFT_H)
    )

    mult = float(max(0.5, min(1.7, mult)))

    final_score = float(fused["score"] * mult)
    final_conf = float(max(0.5, min(0.97, fused["conf"] * (1.0 + 0.15 * (mult - 1.0)))))


    intent = _intent(fused["ret"], final_score, regime, ctx)
    exposure = float(abs(final_score) * final_conf)

    risk = 0.02
    if ctx["volatility"] > 0.04:
        risk *= 0.6
    elif ctx["volatility"] < 0.015:
        risk *= 1.1

    lab = (regime or {}).get("label", "chop").lower()
    if lab == "bear":
        risk *= 0.75
    elif lab == "panic":
        risk *= 0.6
    elif lab == "bull":
        risk *= 1.05

    risk = float(max(0.005, min(0.05, risk)))

    # Phase S0 (Swing): lightweight reason codes for decision auditing.
    reason_codes = []
    if intent == "HOLD":
        reason_codes.append("NO_EDGE")
    if lab in ("bear", "panic") and intent == "BUY":
        reason_codes.append("REGIME_BLOCK_BUY")
    if ctx.get("volatility", 0.0) > 0.04:
        reason_codes.append("HIGH_VOL_PENALTY")
    if ctx.get("impact_mult", 1.0) < 1.0:
        reason_codes.append("NEWS_IMPACT_PENALTY")

    trade_gate = not (lab in ("bear", "panic") and intent == "BUY")

    return {
        "intent": intent,
        "score": round(final_score, 4),
        "confidence": round(final_conf, 4),
        "exposure_scale": round(exposure, 4),
        "risk": round(risk, 4),
        "trade_gate": bool(trade_gate),
        "reason_codes": reason_codes,
        "reason": ";".join(reason_codes) if reason_codes else "ok",
        "reasons": {
            "predicted_return": round(float(fused["ret"]), 4),
            "confidence_raw": round(float(fused["conf"]), 4),
            "short_term": round(float(fused["short"]), 4),
            "mid_term": round(float(fused["mid"]), 4),
            "long_term": round(float(fused["long"]), 4),
            "volatility": float(ctx["volatility"]),
            "sentiment": float(ctx["sentiment"]),
            "trend": ctx["trend"],
            "drift_mult": round(float(_drift_mult(brain_meta, brain_node, _PRIMARY_DRIFT_H)), 4),
            "sector_mult": round(float(_sector_mult(sector, brain_meta, _PRIMARY_DRIFT_H)), 4),
            "sector": sector,
            "regime": lab,
            "multiplier": round(mult, 4),
        },
    }


# ======================================================================
# PUBLIC ENTRY
# ======================================================================
def apply_policy():
    rolling = _read_rolling()
    if not rolling:
        log("[policy_engine] No rolling.json.gz — exiting.")
        return {}

    brain = _read_brain() or {}
    brain_meta = brain.get("_meta", {}) if isinstance(brain, dict) else {}

    # ✅ NEW: Load AION brain meta
    aion_meta = _load_aion_brain_meta()
    regime = detect_regime(rolling)

    # ✅ NEW: Optional regime-level AION modifiers
    aion_reg_mod = _aion_regime_mod(aion_meta, regime)

    updated = 0

    for sym, node in rolling.items():
        if str(sym).startswith("_"):
            continue
        if not isinstance(node, dict):
            continue

        preds = node.get("predictions")
        if not isinstance(preds, dict) or not preds:
            continue

        brain_node = {}
        if isinstance(brain, dict):
            brain_node = brain.get(sym, {}) or brain.get(str(sym).upper(), {}) or {}

        policy = _build(sym, node, preds, brain_node if isinstance(brain_node, dict) else {}, regime, brain_meta)

        # ==================================================================
        # ✅ AION BRAIN MODULATION (behavioral policy overlay)
        # ==================================================================
        # Combine global + regime mods
        cb = float(aion_meta.get("confidence_bias", 1.0)) * float(aion_reg_mod.get("confidence_bias", 1.0))
        rb = float(aion_meta.get("risk_bias", 1.0)) * float(aion_reg_mod.get("risk_bias", 1.0))
        ag = float(aion_meta.get("aggressiveness", 1.0)) * float(aion_reg_mod.get("aggressiveness", 1.0))

        # Apply with bounds (prevent runaway)
        policy_conf = float(policy.get("confidence", 0.0)) * cb
        policy_risk = float(policy.get("risk", 0.0)) * rb
        policy_expo = float(policy.get("exposure_scale", 0.0)) * ag

        policy["confidence"] = round(float(max(0.25, min(0.99, policy_conf))), 4)
        policy["risk"] = round(float(max(0.003, min(0.06, policy_risk))), 4)
        policy["exposure_scale"] = round(float(max(0.0, min(2.0, policy_expo))), 4)

        # Traceability: what did AION brain do?
        reasons = policy.get("reasons") or {}
        if isinstance(reasons, dict):
            reasons["aion_confidence_bias"] = round(cb, 4)
            reasons["aion_risk_bias"] = round(rb, 4)
            reasons["aion_aggressiveness"] = round(ag, 4)
            reasons["aion_regime_mod"] = (regime or {}).get("label", "chop")
            policy["reasons"] = reasons

        node["policy"] = policy
        rolling[sym] = node
        updated += 1

    save_rolling(rolling)
    log(f"[policy_engine] Updated policy for {updated} symbols (v1.2 regression + AION brain).")
    return {"updated": updated, "regime": regime, "aion_meta": aion_meta}  # include meta for debugging