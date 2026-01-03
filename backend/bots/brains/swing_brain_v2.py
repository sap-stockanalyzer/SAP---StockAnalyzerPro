# backend/bots/brains/swing_brain_v2.py
from __future__ import annotations

import math
import os
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
# Phase 1 — Human-like swing entry gating (tiered)
# ------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name, "") or "").strip()
    try:
        return float(raw) if raw else float(default)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name, "") or "").strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _tiered_entry_gate(
    *,
    conf: float,
    exp_ret: float,
    vol: float,
    regime_label: str,
    regime_conf: float,
    allow_countertrend: bool = False,
) -> Tuple[bool, str, str]:
    """Tiered entry gate (Phase 1).

    Goal: trade more like a good human.
    - Don't demand "perfect" signals; allow decent ones with smaller size later.
    - Require trend/regime alignment OR explicit opt-in to countertrend.
    - Avoid hostile volatility (high-vol chop) unless signal is very strong.

    Returns: (ok, tier, reason_code)
      tier in {"A","B","C"}

    Env knobs (optional):
      SWING_TIER_A_CONF, SWING_TIER_B_CONF, SWING_TIER_C_CONF
      SWING_TIER_A_RET,  SWING_TIER_B_RET,  SWING_TIER_C_RET
      SWING_MAX_VOL      (annualized-ish proxy from your feature; default 0.06)
      SWING_BEAR_BUY_BLOCK (1=block buys in bear; default 0)
      SWING_BEAR_CONF_BONUS (extra conf required for BUY in bear; default 0.05)
    """

    # --- default tiers (loosened vs strict v0) ---
    a_conf = _env_float("SWING_TIER_A_CONF", 0.52)
    b_conf = _env_float("SWING_TIER_B_CONF", 0.38)
    c_conf = _env_float("SWING_TIER_C_CONF", 0.26)

    a_ret = _env_float("SWING_TIER_A_RET", 0.040)
    b_ret = _env_float("SWING_TIER_B_RET", 0.018)
    c_ret = _env_float("SWING_TIER_C_RET", 0.010)

    # Volatility seatbelt: if vol is high, require stronger confidence.
    max_vol = _env_float("SWING_MAX_VOL", 0.060)
    vol = float(vol or 0.0)

    rl = (regime_label or "").strip().lower()

    # Legacy knob: previously this could hard-block buys in bear/risk_off regimes.
    # Phase 3 upgrade: do NOT forbid trading outright — instead we downgrade the tier
    # (which shrinks position sizing downstream) and tag the reason.
    bear_block = _env_bool("SWING_BEAR_BUY_BLOCK", False)
    if bear_block and rl in {"bear", "risk_off"} and not allow_countertrend:
        # Keep it tradeable, but treat as low-conviction.
        return True, "C", "regime_risk_bear"

    # Soft regime adjustment: in bear, demand a bit more confidence for BUY signals.
    bear_conf_bonus = _env_float("SWING_BEAR_CONF_BONUS", 0.05)

    # Treat low regime confidence as "fog" => tighten slightly.
    fog_pen = 0.00
    if float(regime_conf or 0.0) < 0.35:
        fog_pen = 0.03

    # Vol penalty: above max_vol, only Tier A survives unless explicitly allowed.
    vol_pen = 0.0
    if vol > max_vol:
        vol_pen = 0.06

    # Apply adjustments
    conf_adj = float(conf)
    if rl in {"bear", "risk_off"}:
        conf_adj -= (bear_conf_bonus if not allow_countertrend else 0.0)
    conf_req_bump = fog_pen + vol_pen

    # Choose tier by adjusted confidence + expected return.
    if conf_adj >= (a_conf + conf_req_bump) and exp_ret >= a_ret:
        return True, "A", "ok"
    if conf_adj >= (b_conf + conf_req_bump) and exp_ret >= b_ret:
        return True, "B", "ok"
    if conf_adj >= (c_conf + conf_req_bump) and exp_ret >= c_ret:
        return True, "C", "ok"

    # Rejection reason (for missed-opportunity tracking)
    if vol_pen > 0.0 and conf_adj < (a_conf + conf_req_bump):
        return False, "C", "vol_hostile"
    if fog_pen > 0.0 and conf_adj < (c_conf + conf_req_bump):
        return False, "C", "regime_fog"
    return False, "C", "below_tier_thresholds"

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

    # Legacy hard gates (very strict): keep as an opt-in escape hatch.
    # Phase 1 default is tiered gating.
    use_legacy = _env_bool("SWING_USE_LEGACY_ENTRY_GATES", False)

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

        # Phase 1: allow "early" entries even when policy says HOLD.
        # The policy engine is intentionally conservative; swing positions
        # can be built earlier with smaller sizing (Phase 2). Here we only
        # allow it when there is a real edge (exp_ret/vol/regime checks below).
        allow_early = _env_bool("SWING_ALLOW_EARLY_ENTRIES", True)
        is_buy_like = (intent == "BUY") or (allow_early and intent in {"HOLD", ""})
        if not is_buy_like:
            continue

        # ---------------- Prediction ---------------- #
        preds = node.get("predictions") or {}
        hblk = preds.get(horizon) or {}
        exp_ret = _safe_float(hblk.get("predicted_return"), 0.0)

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

        # ---------------- Entry gating ---------------- #
        # Phase 1 default: tiered gates. Legacy mode keeps the old strict rules.
        if use_legacy:
            if confidence < conf_threshold:
                continue
            if exp_ret <= 0:
                continue
            tier = "A"
        else:
            ok, tier, _why = _tiered_entry_gate(regime=regime, risk_off=risk_off, conf=confidence, exp_ret=exp_ret, vol=vol)
            if not ok:
                continue
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

        # Phase 1: tiny bonus for stronger entry tiers (A > B > C).
        # Keeps ordering sane without making tiers hard permissions.
        U += {"A": 0.04, "B": 0.02, "C": 0.01}.get(str(tier).upper(), 0.0)

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