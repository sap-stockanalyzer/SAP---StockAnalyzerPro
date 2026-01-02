# dt_backend/strategies/strategy_engine_dt.py — v1.0 (Phase 3)
"""Strategy engine for dt_backend (Phase 3).

This module evaluates a small set of hand-crafted intraday strategies
using the shared feature spine (features_dt) + levels (levels_dt) +
global regime weights (regime_dt).

It outputs a *trade plan* that higher layers can convert into execution.

Design constraints
------------------
* Best-effort and safe: never raise in the main loop.
* Deterministic: same inputs -> same plan.
* Conservative defaults: it is better to skip than to spam trades.

Returned setup schema
---------------------
    {
      "bot": "VWAP_MR"|"ORB"|"TREND_PULLBACK"|"SQUEEZE",
      "side": "BUY"|"SELL"|"FLAT",
      "confidence": 0.0..0.99,
      "score": float,                  # higher is better
      "entry": {"type": "MKT"},
      "risk": {
        "stop": float|None,
        "take_profit": float|None,
        "time_stop_min": int|None,
        "r_target": float|None,
        "partials": bool,
        "trail": bool,
      },
      "reason": str,
    }
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _f(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return max(float(lo), min(float(hi), float(x)))
    except Exception:
        return float(lo)


def _env_float(name: str, default: float) -> float:
    try:
        raw = (os.getenv(name, "") or "").strip()
        return float(raw) if raw != "" else float(default)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name, "") or "").strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _get_global_regime_dt(rolling: Dict[str, Any]) -> Dict[str, Any]:
    g = rolling.get("_GLOBAL_DT") or {}
    return g.get("regime_dt") if isinstance(g, dict) and isinstance(g.get("regime_dt"), dict) else {}


def _get_strategy_weights(rolling: Dict[str, Any]) -> Dict[str, float]:
    reg = _get_global_regime_dt(rolling)
    w = reg.get("strategy_weights")
    if isinstance(w, dict):
        out: Dict[str, float] = {}
        for k, v in w.items():
            if isinstance(k, str):
                out[k.upper()] = _clamp(_f(v, 0.0), 0.0, 2.0)
        return out
    return {}


def _get_micro_label(rolling: Dict[str, Any]) -> str:
    g = rolling.get("_GLOBAL_DT") or {}
    if not isinstance(g, dict):
        return ""
    m = g.get("micro_regime_dt") or {}
    if not isinstance(m, dict):
        return ""
    return str(m.get("label") or "").upper()


def _allowed_in_micro(bot: str, micro: str) -> bool:
    """Basic time-of-day gating.

    You can override with env flags per bot.
    """
    bot = bot.upper()
    micro = (micro or "").upper()

    # Closed/premarket are off by default.
    if micro in {"CLOSED", "PREMARKET"}:
        return _env_bool("DT_ALLOW_PREMARKET_TRADES", False)

    # Lunch is off by default.
    if micro == "LUNCH":
        if _env_bool("DT_ALLOW_LUNCH_TRADES", False):
            return True
        # Allow only VWAP MR by default if you insist.
        return bot == "VWAP_MR" and _env_bool("DT_ALLOW_LUNCH_VWAP_MR", False)

    return True


def _base_risk(atr: float, *, stop_mult: float, tp_mult: float, last: float, side: str) -> Tuple[Optional[float], Optional[float]]:
    if atr <= 0 or last <= 0:
        return None, None
    side = side.upper()
    if side == "BUY":
        stop = last - stop_mult * atr
        tp = last + tp_mult * atr
    else:
        stop = last + stop_mult * atr
        tp = last - tp_mult * atr
    # guard against negative prices
    stop = max(0.01, float(stop))
    tp = max(0.01, float(tp))
    return float(stop), float(tp)


def _mk_setup(
    *,
    bot: str,
    side: str,
    confidence: float,
    score: float,
    reason: str,
    stop: Optional[float] = None,
    take_profit: Optional[float] = None,
    time_stop_min: Optional[int] = None,
    r_target: Optional[float] = None,
    partials: bool = True,
    trail: bool = True,
) -> Dict[str, Any]:
    return {
        "bot": str(bot).upper(),
        "side": str(side).upper(),
        "confidence": float(_clamp(confidence, 0.0, 0.99)),
        "score": float(score),
        "entry": {"type": "MKT"},
        "risk": {
            "stop": float(stop) if stop is not None else None,
            "take_profit": float(take_profit) if take_profit is not None else None,
            "time_stop_min": int(time_stop_min) if time_stop_min is not None else None,
            "r_target": float(r_target) if r_target is not None else None,
            "partials": bool(partials),
            "trail": bool(trail),
        },
        "reason": str(reason)[:280],
        "ts": _utc_iso(),
    }


# ---------------------------------------------------------------------------
# Bot logic
# ---------------------------------------------------------------------------


def bot_vwap_mean_reversion(sym: str, feat: Dict[str, Any], levels: Dict[str, Any], *, micro: str) -> Optional[Dict[str, Any]]:
    """VWAP mean reversion.

    Signal idea:
      - price deviates from VWAP by a volatility/ATR scaled amount
      - trend is not strongly directional
      - enter back toward VWAP
    """
    if not _allowed_in_micro("VWAP_MR", micro):
        return None

    last = _f(feat.get("last_price"), 0.0)
    vwap = _f(feat.get("vwap"), 0.0)
    vwap_dist = _f(feat.get("vwap_dist"), 0.0)  # pct
    atr = _f(feat.get("atr_14"), 0.0)
    rv = _f(feat.get("realized_vol"), 0.0)
    trend_score = _f(feat.get("trend_score"), 0.0)
    rel_vol = _f(feat.get("rel_volume"), 0.0)

    if last <= 0 or vwap <= 0 or atr <= 0:
        return None

    # Require meaningful deviation.
    # Use a hybrid threshold: pct deviation vs ATR/price.
    dev = abs(vwap_dist)
    atr_pct = atr / last if last > 0 else 0.0

    dev_th = _env_float("DT_VWAP_MR_DEV_TH", 0.004)  # 0.4%
    atr_th = _env_float("DT_VWAP_MR_ATR_PCT_TH", 0.45)  # 0.45x ATR as pct-of-price
    # Deviation must exceed either fixed pct or ATR-scaled
    if not (dev >= dev_th or dev >= (atr_th * atr_pct)):
        return None

    # Avoid fighting strong trends.
    if abs(trend_score) > _env_float("DT_VWAP_MR_MAX_TREND", 0.75):
        return None

    # Optional: require some liquidity pop (avoid dead tape).
    if rel_vol < _env_float("DT_VWAP_MR_MIN_RELVOL", 0.9):
        return None

    side = "BUY" if vwap_dist < 0 else "SELL"  # below vwap -> buy; above -> sell

    # Confidence increases with deviation and decreases with volatility.
    conf = _clamp((dev / max(dev_th, 1e-6)) * 0.35, 0.10, 0.80)
    conf *= _clamp(1.2 - (rv * 25.0), 0.55, 1.10)

    # Target VWAP (take profit) and stop beyond extension.
    stop_mult = _env_float("DT_VWAP_MR_STOP_ATR", 1.25)
    stop, _ = _base_risk(atr, stop_mult=stop_mult, tp_mult=1.0, last=last, side=side)
    tp = vwap  # mean reversion target

    # Score: deviation * (1 - |trend|)
    score = float(dev) * (1.0 - min(1.0, abs(trend_score)))
    score *= 100.0

    reason = f"VWAP_MR dev={dev:.4f} vwap_dist={vwap_dist:.4f} trend={trend_score:.2f} relvol={rel_vol:.2f}"
    return _mk_setup(
        bot="VWAP_MR",
        side=side,
        confidence=conf,
        score=score,
        reason=reason,
        stop=stop,
        take_profit=float(tp),
        time_stop_min=int(_env_float("DT_VWAP_MR_TIME_STOP_MIN", 45)),
        r_target=1.0,
        partials=True,
        trail=False,
    )


def bot_opening_range_breakout(sym: str, feat: Dict[str, Any], levels: Dict[str, Any], *, micro: str) -> Optional[Dict[str, Any]]:
    """Opening range breakout.

    Signal idea:
      - break OR5/OR15 with volume confirmation
      - only early-session micro regimes
    """
    if not _allowed_in_micro("ORB", micro):
        return None
    if micro not in {"OPEN", "MID"} and not _env_bool("DT_ALLOW_ORB_ALL_DAY", False):
        return None

    last = _f(feat.get("last_price"), 0.0)
    atr = _f(feat.get("atr_14"), 0.0)
    trend_score = _f(feat.get("trend_score"), 0.0)
    rel_vol = _f(feat.get("rel_volume"), 0.0)

    or5_h = _f(feat.get("or5_high"), 0.0)
    or5_l = _f(feat.get("or5_low"), 0.0)
    or15_h = _f(feat.get("or15_high"), 0.0)
    or15_l = _f(feat.get("or15_low"), 0.0)

    if last <= 0 or atr <= 0:
        return None
    if (or5_h <= 0 and or5_l <= 0 and or15_h <= 0 and or15_l <= 0):
        return None

    # Prefer 15m range if available.
    hi = or15_h if or15_h > 0 else or5_h
    lo = or15_l if or15_l > 0 else or5_l

    # Must actually break.
    side = ""
    if hi > 0 and last > hi:
        side = "BUY"
    elif lo > 0 and last < lo:
        side = "SELL"
    else:
        return None

    # Volume confirmation.
    if rel_vol < _env_float("DT_ORB_MIN_RELVOL", 1.4):
        return None

    # Optional: require trend alignment.
    if side == "BUY" and trend_score < _env_float("DT_ORB_MIN_TREND", 0.15):
        return None
    if side == "SELL" and trend_score > -_env_float("DT_ORB_MIN_TREND", 0.15):
        return None

    stop_mult = _env_float("DT_ORB_STOP_ATR", 1.2)
    tp_mult = _env_float("DT_ORB_TP_ATR", 2.2)
    stop, tp = _base_risk(atr, stop_mult=stop_mult, tp_mult=tp_mult, last=last, side=side)

    # Confidence grows with rel volume + trend alignment.
    conf = 0.35 + 0.10 * min(3.0, rel_vol) + 0.20 * min(1.0, abs(trend_score))
    conf = _clamp(conf, 0.20, 0.90)

    score = (abs(last - (hi if side == "BUY" else lo)) / max(atr, 1e-6))
    score *= (min(3.0, rel_vol))
    score *= 25.0

    reason = f"ORB {side} break={'OR15' if (or15_h>0 and or15_l>0) else 'OR5'} relvol={rel_vol:.2f} trend={trend_score:.2f}"
    return _mk_setup(
        bot="ORB",
        side=side,
        confidence=conf,
        score=score,
        reason=reason,
        stop=stop,
        take_profit=tp,
        time_stop_min=int(_env_float("DT_ORB_TIME_STOP_MIN", 75)),
        r_target=2.0,
        partials=True,
        trail=True,
    )


def bot_trend_pullback(sym: str, feat: Dict[str, Any], levels: Dict[str, Any], *, micro: str) -> Optional[Dict[str, Any]]:
    """Trend continuation pullback.

    Signal idea:
      - strong trend_score in one direction
      - price pulls back toward VWAP/SMA20 then resumes
    """
    if not _allowed_in_micro("TREND_PULLBACK", micro):
        return None

    last = _f(feat.get("last_price"), 0.0)
    atr = _f(feat.get("atr_14"), 0.0)
    trend_score = _f(feat.get("trend_score"), 0.0)
    vwap_dist = _f(feat.get("vwap_dist"), 0.0)
    sma20_dist = _f(feat.get("sma20_dist"), 0.0)
    rsi_14 = _f(feat.get("rsi_14"), 50.0)
    rel_vol = _f(feat.get("rel_volume"), 0.0)

    if last <= 0 or atr <= 0:
        return None

    min_trend = _env_float("DT_TP_MIN_TREND", 0.45)
    if abs(trend_score) < min_trend:
        return None

    # Determine direction.
    side = "BUY" if trend_score > 0 else "SELL"

    # Pullback condition: against-the-trend distance to VWAP/SMA20.
    # For longs: small negative vwap_dist/sma20_dist (pulled back)
    # For shorts: small positive
    pb_max = _env_float("DT_TP_PULLBACK_MAX", 0.006)
    pb_min = _env_float("DT_TP_PULLBACK_MIN", 0.001)

    if side == "BUY":
        pb = max(0.0, -(vwap_dist)) + max(0.0, -(sma20_dist))
        ok_pb = (pb_min <= pb <= pb_max)
        ok_rsi = rsi_14 >= _env_float("DT_TP_LONG_MIN_RSI", 48.0)
    else:
        pb = max(0.0, (vwap_dist)) + max(0.0, (sma20_dist))
        ok_pb = (pb_min <= pb <= pb_max)
        ok_rsi = rsi_14 <= _env_float("DT_TP_SHORT_MAX_RSI", 52.0)

    if not ok_pb:
        return None

    # Require at least normal liquidity.
    if rel_vol < _env_float("DT_TP_MIN_RELVOL", 1.0):
        return None

    stop_mult = _env_float("DT_TP_STOP_ATR", 1.35)
    tp_mult = _env_float("DT_TP_TP_ATR", 2.6)
    stop, tp = _base_risk(atr, stop_mult=stop_mult, tp_mult=tp_mult, last=last, side=side)

    conf = 0.35 + 0.25 * min(1.0, abs(trend_score)) + 0.10 * min(2.5, rel_vol)
    conf = _clamp(conf, 0.25, 0.92)

    score = abs(trend_score) * 40.0 + min(3.0, rel_vol) * 10.0 + (pb / max(pb_min, 1e-6)) * 5.0

    reason = f"TREND_PB {side} trend={trend_score:.2f} pb={pb:.4f} rsi={rsi_14:.1f} relvol={rel_vol:.2f}"
    return _mk_setup(
        bot="TREND_PULLBACK",
        side=side,
        confidence=conf,
        score=score,
        reason=reason,
        stop=stop,
        take_profit=tp,
        time_stop_min=int(_env_float("DT_TP_TIME_STOP_MIN", 120)),
        r_target=2.0,
        partials=True,
        trail=True,
    )


def bot_squeeze_breakout(sym: str, feat: Dict[str, Any], levels: Dict[str, Any], *, micro: str, node_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Volatility squeeze breakout.

    Signal idea:
      - squeeze_on was true recently and now releases (or strongly breaks during squeeze)
      - volume expands
      - direction inferred from OR breaks / trend_score
    """
    if not _allowed_in_micro("SQUEEZE", micro):
        return None

    last = _f(feat.get("last_price"), 0.0)
    atr = _f(feat.get("atr_14"), 0.0)
    squeeze_on = _f(feat.get("squeeze_on"), 0.0) >= 0.5
    squeeze_ratio = _f(feat.get("squeeze_ratio"), 0.0)
    rel_vol = _f(feat.get("rel_volume"), 0.0)
    trend_score = _f(feat.get("trend_score"), 0.0)
    or15_break = _f(feat.get("or15_break"), 0.0)
    vwap_slope = _f(feat.get("vwap_slope"), 0.0)

    if last <= 0 or atr <= 0:
        return None

    # State memory: detect release.
    st = node_state if isinstance(node_state, dict) else {}
    prev_on = bool(st.get("prev_squeeze_on"))
    # release means: was on, now off
    released = (prev_on and not squeeze_on)

    # Must be (recently) squeezed.
    min_ratio = _env_float("DT_SQ_MAX_RATIO", 1.0)  # bb/kc below 1 means true squeeze
    if not (squeeze_on or released):
        return None
    if squeeze_ratio > (min_ratio * 1.35) and not released:
        return None

    # Volume expansion.
    if rel_vol < _env_float("DT_SQ_MIN_RELVOL", 1.25):
        return None

    # Direction.
    if or15_break > 0:
        side = "BUY"
    elif or15_break < 0:
        side = "SELL"
    else:
        # Fall back to trend_score / vwap slope.
        if trend_score > 0.2 or vwap_slope > 0:
            side = "BUY"
        elif trend_score < -0.2 or vwap_slope < 0:
            side = "SELL"
        else:
            return None

    stop_mult = _env_float("DT_SQ_STOP_ATR", 1.35)
    tp_mult = _env_float("DT_SQ_TP_ATR", 2.8)
    stop, tp = _base_risk(atr, stop_mult=stop_mult, tp_mult=tp_mult, last=last, side=side)

    conf = 0.35 + 0.15 * (1.0 if released else 0.0) + 0.15 * min(3.0, rel_vol) + 0.15 * min(1.0, abs(trend_score))
    conf = _clamp(conf, 0.25, 0.92)

    score = (min(3.0, rel_vol) * 20.0) + (abs(trend_score) * 18.0) + (1.0 if released else 0.0) * 10.0
    reason = f"SQUEEZE {side} released={int(released)} sq_on={int(squeeze_on)} ratio={squeeze_ratio:.2f} relvol={rel_vol:.2f}"
    return _mk_setup(
        bot="SQUEEZE",
        side=side,
        confidence=conf,
        score=score,
        reason=reason,
        stop=stop,
        take_profit=tp,
        time_stop_min=int(_env_float("DT_SQ_TIME_STOP_MIN", 150)),
        r_target=2.0,
        partials=True,
        trail=True,
    )


def build_setups_for_symbol(
    sym: str,
    node: Dict[str, Any],
    *,
    rolling: Dict[str, Any],
    micro: str,
) -> List[Dict[str, Any]]:
    feat = node.get("features_dt") if isinstance(node.get("features_dt"), dict) else {}
    levels = node.get("levels_dt") if isinstance(node.get("levels_dt"), dict) else {}

    if not feat:
        return []

    setups: List[Dict[str, Any]] = []
    try:
        s = bot_opening_range_breakout(sym, feat, levels, micro=micro)
        if s:
            setups.append(s)
    except Exception:
        pass
    try:
        s = bot_trend_pullback(sym, feat, levels, micro=micro)
        if s:
            setups.append(s)
    except Exception:
        pass
    try:
        s = bot_vwap_mean_reversion(sym, feat, levels, micro=micro)
        if s:
            setups.append(s)
    except Exception:
        pass
    try:
        st = node.get("_squeeze_state") if isinstance(node.get("_squeeze_state"), dict) else {}
        s = bot_squeeze_breakout(sym, feat, levels, micro=micro, node_state=st)
        if s:
            setups.append(s)
    except Exception:
        pass

    return setups


def select_best_setup(
    sym: str,
    node: Dict[str, Any],
    *,
    rolling: Dict[str, Any],
    micro: str,
    allowed_bots: Optional[List[str]] = None,
    bot_weights: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    setups = build_setups_for_symbol(sym, node, rolling=rolling, micro=micro)
    if not setups:
        return None

    # Weights from regime detector (Phase 2), optionally overridden/augmented
    # by a Phase 4 meta-controller.
    w = _get_strategy_weights(rolling)
    if isinstance(bot_weights, dict) and bot_weights:
        # Multiply so the meta-controller can downweight/disable styles.
        w2: Dict[str, float] = {}
        for k, v in w.items():
            w2[str(k).upper()] = _f(v, 1.0) * 1.0
        for k, v in bot_weights.items():
            if isinstance(k, str):
                kk = k.upper()
                w2[kk] = _f(w2.get(kk), 1.0) * _f(v, 1.0)
        w = w2

    # Enable/disable bots.
    enabled = {
        "VWAP_MR": _env_bool("DT_ENABLE_VWAP_MR", True),
        "ORB": _env_bool("DT_ENABLE_ORB", True),
        "TREND_PULLBACK": _env_bool("DT_ENABLE_TREND_PULLBACK", True),
        "SQUEEZE": _env_bool("DT_ENABLE_SQUEEZE", True),
    }

    allowed_set = None
    if isinstance(allowed_bots, list) and allowed_bots:
        allowed_set = {str(b).upper() for b in allowed_bots if isinstance(b, str)}

    best = None
    best_score = -1e9
    for s in setups:
        bot = str(s.get("bot") or "").upper()
        if allowed_set is not None and bot not in allowed_set:
            continue
        if not enabled.get(bot, True):
            continue
        base = _f(s.get("score"), 0.0)
        weight = _f(w.get(bot), 1.0) if w else 1.0
        score = base * weight
        if score > best_score:
            best = s
            best_score = score

    if best is None:
        return None

    # Final gating thresholds.
    min_conf = _env_float("DT_STRAT_MIN_CONF", 0.32)
    min_score = _env_float("DT_STRAT_MIN_SCORE", 7.0)
    if _f(best.get("confidence"), 0.0) < min_conf:
        return None
    if _f(best.get("score"), 0.0) < min_score:
        return None

    return best
