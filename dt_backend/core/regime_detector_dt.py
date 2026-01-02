# dt_backend/core/regime_detector_dt.py — v2.0 (Phase 2 + 2.5)
"""Intraday regime detector for dt_backend.

Phase 2 — Regime v2 with hysteresis
----------------------------------
Goal: stop running the wrong strategy on the wrong day.

Writes:
    rolling["_GLOBAL_DT"]["regime_dt"] = {
        "label": "TREND_UP"|"TREND_DOWN"|"RANGE"|"HIGH_VOL"|"LOW_VOL"|"UNKNOWN",
        "day_type": "TREND_DAY"|"MEAN_REVERT_DAY"|"CHOP_DAY"|"UNKNOWN",
        "confidence": 0.0..0.99,
        "mkt_trend": float,   # from market proxies
        "mkt_vol": float,
        "strategy_weights": {"VWAP_MR":..., "ORB":..., "TREND_PULLBACK":..., "SQUEEZE":...},
        "ts": "...Z",
        "_state": {"streak": int, "stable_label": str}
    }

For backward compatibility, we also maintain:
    rolling["_GLOBAL_DT"]["regime"] = {"label": "bull"|"bear"|"chop"|"unknown", ...}

Phase 2.5 — Micro-regimes + levels
---------------------------------
Also writes:
    rolling["_GLOBAL_DT"]["micro_regime_dt"] = {...}
and (best-effort):
    rolling[sym]["levels_dt"] = {...}
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from .data_pipeline_dt import _read_rolling, save_rolling, log, ensure_symbol_node
from .micro_regime_dt import compute_micro_regime
from .levels_engine_dt import update_levels_in_rolling


def _utc_iso(now_utc: Optional[datetime] = None) -> str:
    dt = now_utc or datetime.now(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        raw = (os.getenv(name, "") or "").strip()
        return float(raw) if raw else float(default)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        raw = (os.getenv(name, "") or "").strip()
        return int(float(raw)) if raw else int(default)
    except Exception:
        return int(default)


def _extract_market_proxy(rolling: Dict[str, Any]) -> Tuple[float, float]:
    """Return (mkt_trend, mkt_vol) from any features_dt snapshot."""
    for sym, node in rolling.items():
        if not isinstance(sym, str) or sym.startswith("_"):
            continue
        if not isinstance(node, dict):
            continue
        feat = node.get("features_dt")
        if not isinstance(feat, dict):
            continue
        if "mkt_trend" in feat and "mkt_vol" in feat:
            return _safe_float(feat.get("mkt_trend"), 0.0), _safe_float(feat.get("mkt_vol"), 0.0)
    return 0.0, 0.0


def _classify_candidate(mkt_trend: float, mkt_vol: float) -> Tuple[str, str, float]:
    """Return (label, day_type, confidence)."""
    # Knobs (env-overridable)
    trend_th = _env_float("DT_REGIME_TREND_TH", 0.35)
    trend_strong = _env_float("DT_REGIME_TREND_STRONG", 0.70)
    vol_low = _env_float("DT_REGIME_VOL_LOW", 0.006)
    vol_high = _env_float("DT_REGIME_VOL_HIGH", 0.018)

    t = float(mkt_trend)
    v = float(mkt_vol)

    # Primary label
    if abs(t) >= trend_th:
        label = "TREND_UP" if t > 0 else "TREND_DOWN"
    else:
        # In flat-ish markets, volatility dominates the "day feel"
        if v >= vol_high:
            label = "HIGH_VOL"
        elif v <= vol_low:
            label = "LOW_VOL"
        else:
            label = "RANGE"

    # Day type (strategy routing hint)
    if label in {"TREND_UP", "TREND_DOWN"} and (abs(t) >= trend_strong or v >= (0.5 * vol_high)):
        day_type = "TREND_DAY"
    elif label in {"RANGE", "LOW_VOL"} and v <= vol_high:
        day_type = "MEAN_REVERT_DAY"
    elif label in {"HIGH_VOL"}:
        day_type = "CHOP_DAY"
    else:
        day_type = "CHOP_DAY" if v >= (0.75 * vol_high) else "UNKNOWN"

    # Confidence: combine clarity of trend + clarity of vol
    t_score = min(1.0, abs(t) / max(trend_th, 1e-6))
    v_score = 0.0
    if v >= vol_high:
        v_score = 1.0
    elif v <= vol_low:
        v_score = 1.0
    else:
        # middling vol is less informative
        v_score = 0.35
    conf = 0.45 + 0.35 * t_score + 0.20 * v_score
    conf = max(0.0, min(0.99, conf))
    return label, day_type, float(conf)


def _weights_for(day_type: str, label: str) -> Dict[str, float]:
    # Four strategy families (your Phase 3 bots)
    # VWAP_MR: VWAP mean reversion
    # ORB: opening range breakout
    # TREND_PULLBACK: continuation pullback
    # SQUEEZE: volatility squeeze breakout

    if day_type == "TREND_DAY":
        w = {"VWAP_MR": 0.10, "ORB": 0.35, "TREND_PULLBACK": 0.35, "SQUEEZE": 0.20}
    elif day_type == "MEAN_REVERT_DAY":
        w = {"VWAP_MR": 0.45, "ORB": 0.15, "TREND_PULLBACK": 0.15, "SQUEEZE": 0.25}
    elif day_type == "CHOP_DAY":
        w = {"VWAP_MR": 0.20, "ORB": 0.10, "TREND_PULLBACK": 0.10, "SQUEEZE": 0.20}
    else:
        w = {"VWAP_MR": 0.25, "ORB": 0.15, "TREND_PULLBACK": 0.15, "SQUEEZE": 0.20}

    # If extreme vol, bias toward "wait for expansion" (squeeze) and away from mean reversion.
    if label == "HIGH_VOL":
        w["SQUEEZE"] += 0.15
        w["VWAP_MR"] = max(0.05, w["VWAP_MR"] - 0.10)

    # Normalize to sum<=1 (we allow leftover "stand_down" weight implicitly)
    s = sum(max(0.0, float(x)) for x in w.values())
    if s > 0:
        w = {k: float(v) / s for k, v in w.items()}
    return w


def _apply_hysteresis(global_node: Dict[str, Any], candidate: str) -> Tuple[str, Dict[str, Any]]:
    """Return (stable_label, state)."""
    state = global_node.get("_regime_dt_state")
    if not isinstance(state, dict):
        state = {}

    stable = str(state.get("stable_label") or "UNKNOWN")
    last = str(state.get("last_candidate") or "")
    streak = int(state.get("streak") or 0)
    last_switch_t = float(state.get("last_switch_t") or 0.0)

    persist_n = _env_int("DT_REGIME_PERSIST_N", 3)
    min_switch_s = _env_float("DT_REGIME_MIN_SWITCH_SEC", 60.0)

    now_t = time.time()

    if candidate == last:
        streak += 1
    else:
        last = candidate
        streak = 1

    # initialize
    if stable in {"", "UNKNOWN"}:
        stable = candidate
        last_switch_t = now_t
    else:
        if candidate != stable and streak >= max(1, persist_n) and (now_t - last_switch_t) >= min_switch_s:
            stable = candidate
            last_switch_t = now_t

    state.update({
        "stable_label": stable,
        "last_candidate": last,
        "streak": int(streak),
        "last_switch_t": float(last_switch_t),
    })
    return stable, state


def _legacy_map(label: str) -> str:
    # Keep older policy logic alive (bull/bear/chop)
    if label == "TREND_UP":
        return "bull"
    if label == "TREND_DOWN":
        return "bear"
    if label in {"RANGE", "LOW_VOL"}:
        return "chop"
    if label == "HIGH_VOL":
        return "stress"
    return "unknown"


def classify_intraday_regime(*, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
    rolling = _read_rolling() or {}
    if not isinstance(rolling, dict) or not rolling:
        log("[regime_dt] ⚠️ rolling empty.")
        return {"label": "UNKNOWN", "confidence": 0.0}

    g = ensure_symbol_node(rolling, "_GLOBAL_DT")

    # Phase 2.5 micro-regime
    micro = compute_micro_regime(now_utc=now_utc)
    g["micro_regime_dt"] = micro

    # Phase 2 main regime from market proxies
    mkt_trend, mkt_vol = _extract_market_proxy(rolling)
    cand, day_type, conf = _classify_candidate(mkt_trend, mkt_vol)
    stable, state = _apply_hysteresis(g, cand)
    weights = _weights_for(day_type, stable)

    regime_dt = {
        "label": stable,
        "day_type": day_type,
        "confidence": float(conf),
        "mkt_trend": float(mkt_trend),
        "mkt_vol": float(mkt_vol),
        "strategy_weights": weights,
        "ts": _utc_iso(now_utc),
        "_state": {"streak": int(state.get("streak") or 0), "stable_label": stable},
    }
    g["regime_dt"] = regime_dt
    g["_regime_dt_state"] = state

    # Legacy schema
    legacy = {
        "label": _legacy_map(stable),
        "breadth_up": 0.5,
        "n": 0,
        "ts": regime_dt["ts"],
    }
    g["regime"] = legacy

    # Phase 2.5 levels (best-effort + throttled)
    levels_meta = g.get("_levels_meta")
    if not isinstance(levels_meta, dict):
        levels_meta = {}
    min_interval = _env_float("DT_LEVELS_MIN_INTERVAL", 60.0)
    last_t = _safe_float(levels_meta.get("last_t"), 0.0)
    if (time.time() - last_t) >= max(0.0, min_interval):
        max_syms = _env_int("DT_LEVELS_MAX_SYMBOLS", 300)
        try:
            stats = update_levels_in_rolling(rolling, max_symbols=max_syms)
            levels_meta["last_t"] = time.time()
            levels_meta["last_updated"] = _utc_iso(now_utc)
            levels_meta["stats"] = stats
        except Exception:
            pass
    g["_levels_meta"] = levels_meta

    rolling["_GLOBAL_DT"] = g
    save_rolling(rolling)

    log(f"[regime_dt] ✅ {stable} (day={day_type}, conf={conf:.2f}, micro={micro.get('label')})")
    return regime_dt
