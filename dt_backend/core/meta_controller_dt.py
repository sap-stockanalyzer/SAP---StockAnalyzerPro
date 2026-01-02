# dt_backend/core/meta_controller_dt.py — v1.0 (Phase 4)
"""Meta-controller for dt_backend (Phase 4).

This module decides, *once per trading day*, which strategy bots are enabled,
their weights, a risk mode, and a symbol universe.

It is deliberately conservative: if the inputs are missing or ambiguous, it
prefers fewer bots, higher thresholds, and a smaller universe.

Outputs
-------
Writes into rolling["_GLOBAL_DT"]["daily_plan_dt"] and returns the plan dict.

The plan is also intended to be persisted into dt_state.json via
dt_backend.services.dt_truth_store.update_dt_state.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dt_backend.core.config_dt import TIMEZONE
from dt_backend.core.data_pipeline_dt import _read_rolling, save_rolling, log

# Phase 4.5: contextual bandit (shadow-first, optional)
try:
    from dt_backend.bandit.contextual_bandit_dt import suggest_bot_weights
except Exception:  # pragma: no cover
    suggest_bot_weights = None  # type: ignore


SUPPORTED_BOTS = ["VWAP_MR", "ORB", "TREND_PULLBACK", "SQUEEZE"]


def _now_local_date_str() -> str:
    try:
        return datetime.now(TIMEZONE).date().isoformat()
    except Exception:
        return datetime.utcnow().date().isoformat()


def _utc_iso() -> str:
    # keep UTC strings for artifacts
    from datetime import timezone

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


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name, "") or "").strip()
    try:
        return int(float(raw)) if raw else int(default)
    except Exception:
        return int(default)


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


def _get_global_dt(rolling: Dict[str, Any]) -> Dict[str, Any]:
    g = rolling.get("_GLOBAL_DT")
    return g if isinstance(g, dict) else {}


def _get_regime_dt(rolling: Dict[str, Any]) -> Dict[str, Any]:
    g = _get_global_dt(rolling)
    r = g.get("regime_dt")
    return r if isinstance(r, dict) else {}


def _symbol_liquidity_score(sym: str, node: Dict[str, Any]) -> float:
    """Approximate dollar-volume score using latest bar volume * last_price."""
    feat = node.get("features_dt") if isinstance(node.get("features_dt"), dict) else {}
    last_px = _f(feat.get("last_price"), 0.0)
    if last_px <= 0:
        return 0.0

    # Prefer 5m bars if present; fallback to 1m
    bars = node.get("bars_intraday_5m")
    if not isinstance(bars, list) or not bars:
        bars = node.get("bars_intraday")
    vol = 0.0
    try:
        if isinstance(bars, list) and bars:
            b = bars[-1]
            if isinstance(b, dict):
                vol = _f(b.get("v") or b.get("volume"), 0.0)
    except Exception:
        vol = 0.0

    # rel_volume helps prioritize "alive" tape
    rel_vol = _f(feat.get("rel_volume"), 1.0)
    return float(last_px * vol * max(0.25, rel_vol))


def _build_universe(rolling: Dict[str, Any]) -> List[str]:
    """Select a tradable universe for today.

    The goal is to focus the engine on symbols with enough movement + activity.
    """
    max_n = _env_int("DT_UNIVERSE_SIZE", 60)
    min_price = _env_float("DT_UNIVERSE_MIN_PRICE", 2.0)
    min_atr_pct = _env_float("DT_UNIVERSE_MIN_ATR_PCT", 0.0015)  # 0.15%
    max_atr_pct = _env_float("DT_UNIVERSE_MAX_ATR_PCT", 0.0800)  # 8%

    candidates: List[Tuple[str, float]] = []
    for sym, node in rolling.items():
        if not isinstance(sym, str) or sym.startswith("_"):
            continue
        if not isinstance(node, dict):
            continue
        feat = node.get("features_dt") if isinstance(node.get("features_dt"), dict) else {}
        last_px = _f(feat.get("last_price"), 0.0)
        atr = _f(feat.get("atr_14"), 0.0)
        if last_px <= 0 or atr <= 0:
            continue
        if last_px < min_price:
            continue
        atr_pct = atr / last_px if last_px > 0 else 0.0
        if atr_pct < min_atr_pct or atr_pct > max_atr_pct:
            continue

        score = _symbol_liquidity_score(sym, node)
        if score <= 0:
            continue
        candidates.append((sym.upper(), score))

    # Always include market proxies if present
    proxies = [s.strip().upper() for s in (os.getenv("DT_MARKET_PROXIES", "SPY,QQQ").split(",")) if s.strip()]
    proxy_set = set(proxies)

    candidates.sort(key=lambda t: t[1], reverse=True)
    universe = [sym for sym, _ in candidates[: max(0, max_n)]]
    for p in proxies:
        if p and p not in universe and p in proxy_set:
            universe.append(p)

    return universe


def _risk_mode_from_regime(label: str, conf: float) -> str:
    label = (label or "").strip().upper()

    if conf <= 0.25:
        return "CONSERVATIVE"

    if label == "HIGH_VOL":
        return "CONSERVATIVE"

    if label in {"TREND_UP", "TREND_DOWN"}:
        return "AGGRESSIVE" if conf >= 0.60 else "NORMAL"

    if label in {"RANGE", "LOW_VOL"}:
        return "NORMAL" if conf >= 0.45 else "CONSERVATIVE"

    return "CONSERVATIVE" if conf < 0.40 else "NORMAL"


def _bots_from_regime(label: str) -> List[str]:
    label = (label or "").strip().upper()
    if label in {"TREND_UP", "TREND_DOWN"}:
        return ["ORB", "TREND_PULLBACK"]
    if label in {"RANGE", "LOW_VOL"}:
        return ["VWAP_MR", "SQUEEZE"]
    if label == "HIGH_VOL":
        # high vol can support breakouts, but keep it limited
        return ["ORB", "SQUEEZE"]
    # unknown/other
    return ["VWAP_MR"]


def _merge_weights(base: Dict[str, float], extra: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for b in SUPPORTED_BOTS:
        w = _f(base.get(b), 1.0) * _f(extra.get(b), 1.0)
        out[b] = _clamp(w, 0.0, 3.0)
    return out


def build_daily_plan(*, rolling: Dict[str, Any], date_override: Optional[str] = None) -> Dict[str, Any]:
    """Compute today's bot plan using regime outputs + internal heuristics."""
    reg = _get_regime_dt(rolling)
    label = str(reg.get("label") or "").upper() or "UNKNOWN"
    conf = _clamp(_f(reg.get("confidence"), 0.0), 0.0, 1.0)

    regime_weights = {}
    try:
        w = reg.get("strategy_weights")
        if isinstance(w, dict):
            regime_weights = {str(k).upper(): _clamp(_f(v, 1.0), 0.0, 3.0) for k, v in w.items() if isinstance(k, str)}
    except Exception:
        regime_weights = {}

    enabled = _bots_from_regime(label)

    # Optional user overrides
    if _env_bool("DT_FORCE_ALL_BOTS", False):
        enabled = SUPPORTED_BOTS[:]
    else:
        # individual enables
        enabled = [
            b
            for b in enabled
            if _env_bool(f"DT_ENABLE_{b}", True)
        ]

    # Ensure at least one bot.
    if not enabled:
        enabled = ["VWAP_MR"]

    risk_mode = _risk_mode_from_regime(label, conf)

    # Base weights start at 1.0, then multiply by regime weights, then by risk-mode scaling.
    base = {b: 1.0 for b in SUPPORTED_BOTS}
    weights = _merge_weights(base, regime_weights)

    if risk_mode == "CONSERVATIVE":
        # prefer mean reversion / avoid overtrading
        weights["VWAP_MR"] *= 1.15
        weights["ORB"] *= 0.85
        weights["SQUEEZE"] *= 0.90
    elif risk_mode == "AGGRESSIVE":
        weights["ORB"] *= 1.10
        weights["TREND_PULLBACK"] *= 1.10

    # Normalize weights for enabled bots only (for interpretability)
    enabled_w = {b: float(_clamp(weights.get(b, 1.0), 0.0, 3.0)) for b in enabled}
    s = sum(enabled_w.values())
    if s > 0:
        enabled_w = {b: v / s for b, v in enabled_w.items()}

    # Phase 4.5: optional bandit overlay (OFF by default).
    # Safe defaults: only applies when DT_BANDIT_ENABLED=1.
    # For live safety, DT_BANDIT_SHADOW_ONLY=1 keeps it confined to shadow mode.
    try:
        bandit_on = _env_bool('DT_BANDIT_ENABLED', False)
        shadow_only = _env_bool('DT_BANDIT_SHADOW_ONLY', True)
        shadow_enabled = _env_bool('DT_SHADOW_ENABLED', False)
        if bandit_on and suggest_bot_weights is not None and (not shadow_only or shadow_enabled):
            context = {'regime': label, 'regime_conf': conf, 'risk_mode': risk_mode}
            bw = suggest_bot_weights(context=context)
            if isinstance(bw, dict) and bw:
                mixed = {b: float(enabled_w.get(b, 0.0)) * float(bw.get(b, 0.0) or 0.0) for b in enabled_w.keys()}
                s2 = sum(mixed.values())
                if s2 > 0:
                    enabled_w = {b: v / s2 for b, v in mixed.items()}
    except Exception:
        pass

    universe = _build_universe(rolling)

    allow_model_fallback = _env_bool("DT_ALLOW_MODEL_FALLBACK", False)

    return {
        "date": (str(date_override) if date_override else _now_local_date_str()),
        "ts": _utc_iso(),
        "regime": {"label": label, "confidence": conf},
        "risk_mode": risk_mode,
        "enabled_bots": enabled,
        "bot_weights": enabled_w,
        "universe": universe,
        "allow_model_fallback": bool(allow_model_fallback),
        "reason": f"meta: regime={label} conf={conf:.2f} risk={risk_mode} enabled={','.join(enabled)}",
        "version": "phase4_v1",
    }


def ensure_daily_plan(*, force: bool = False, date_override: Optional[str] = None) -> Dict[str, Any]:
    """Ensure rolling contains today's daily_plan_dt.

    If it's a new date or force=True, recompute.
    """
    rolling = _read_rolling() or {}
    if not isinstance(rolling, dict) or not rolling:
        return {}

    g = rolling.get("_GLOBAL_DT")
    if not isinstance(g, dict):
        g = {}

    today = (str(date_override) if date_override else _now_local_date_str())
    existing = g.get("daily_plan_dt") if isinstance(g.get("daily_plan_dt"), dict) else None
    if not force and isinstance(existing, dict) and str(existing.get("date") or "") == today:
        return existing

    plan = build_daily_plan(rolling=rolling, date_override=today)
    g["daily_plan_dt"] = plan
    rolling["_GLOBAL_DT"] = g
    save_rolling(rolling)
    log(f"[meta_dt] 🧠 daily plan set: {plan.get('reason')}")
    return plan
