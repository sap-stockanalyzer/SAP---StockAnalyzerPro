# backend/bots/brains/swing_brain_v2.py
from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any

# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _tanh(x: float) -> float:
    return math.tanh(x)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ------------------------------------------------------------
# Regime → exposure mapping
# ------------------------------------------------------------

REGIME_EXPOSURE = {
    "bull": 1.00,
    "chop": 0.70,
    "bear": 0.45,
    "panic": 0.20,
}

DEFAULT_VOL_REF = 0.03  # ~3% daily vol reference


# ------------------------------------------------------------
# Core brain
# ------------------------------------------------------------

def rank_universe_v2(
    *,
    rolling: Dict[str, dict],
    insights: List[dict],
    horizon: str,
    conf_threshold: float,
) -> List[Tuple[str, float]]:
    """
    Rank symbols using explicit utility:

        U = 0.70 * Profit
          + 0.10 * Smoothness
          + 0.15 * Truthfulness
          + 0.05 * Drawdown

    Returns:
        List[(symbol, score)] sorted desc
    """

    # ---------------- Insight rank ---------------- #
    insight_rank: Dict[str, int] = {}
    for i, row in enumerate(insights):
        sym = str(row.get("symbol") or row.get("ticker") or "").upper()
        if sym:
            insight_rank[sym] = i

    ranked: List[Tuple[str, float]] = []

    for sym, node in rolling.items():
        node = node or {}

        # ---------------- Price ---------------- #
        price = (
            node.get("price")
            or node.get("last")
            or node.get("close")
            or node.get("c")
        )
        price = _safe_float(price, 0.0)
        if price <= 0:
            continue

        # ---------------- Policy ---------------- #
        pol = node.get("policy") or {}
        intent = str(pol.get("intent") or "").upper()
        confidence = _safe_float(pol.get("confidence"), 0.0)
        pol_score = _safe_float(pol.get("score"), 0.0)

        if intent != "BUY":
            continue
        if confidence < conf_threshold:
            continue

        # ---------------- Prediction ---------------- #
        preds = node.get("predictions") or {}
        hblk = preds.get(horizon) or {}
        exp_ret = _safe_float(hblk.get("predicted_return"), 0.0)
        if exp_ret <= 0:
            continue

        # ---------------- Volatility proxy ---------------- #
        # Try multiple fallbacks — must never explode
        vol = (
            node.get("volatility")
            or node.get("atr_pct")
            or node.get("atr")
            or node.get("vol")
        )
        vol = _safe_float(vol, DEFAULT_VOL_REF)
        if vol <= 0:
            vol = DEFAULT_VOL_REF

        # ---------------- Regime ---------------- #
        ctx = node.get("context") or {}
        regime = str(
            ctx.get("regime")
            or ctx.get("market_regime")
            or "chop"
        ).lower()

        risk_off = _safe_float(ctx.get("risk_off"), 0.0)
        exposure = REGIME_EXPOSURE.get(regime, 0.70)

        # ------------------------------------------------
        # Component scores
        # ------------------------------------------------

        # 1) Profit (risk-adjusted)
        profit_score = _tanh(exp_ret / (vol + 1e-6))

        # 2) Smoothness (vol penalty)
        smooth_score = 1.0 - _clamp(vol / DEFAULT_VOL_REF, 0.0, 1.0)

        # 3) Truthfulness (calibrated confidence)
        truth_score = _clamp(confidence, 0.0, 1.0)

        # 4) Drawdown (regime-aware)
        drawdown_score = 1.0 - _clamp(risk_off, 0.0, 1.0)

        # ------------------------------------------------
        # Utility
        # ------------------------------------------------

        U = (
            0.70 * profit_score
            + 0.10 * smooth_score
            + 0.15 * truth_score
            + 0.05 * drawdown_score
        )

        # small directional + insight nudges (non-dominant)
        U += 0.10 * pol_score

        if sym in insight_rank:
            rank = insight_rank[sym]
            n = max(1, len(insights))
            U += 0.05 * (1.0 - rank / n)

        # penny stock penalty
        if price < 3.0:
            U *= 0.7

        # final exposure scaling
        U *= exposure

        if U > 0:
            ranked.append((sym, float(U)))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
