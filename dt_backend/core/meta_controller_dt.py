# dt_backend/core/meta_controller_dt.py — v1.1 (Phase 4 + Candidate Universe)
"""Meta-controller for dt_backend (Phase 4).

This module decides, *once per trading day*, which strategy bots are enabled,
their weights, a risk mode, and a symbol universe.

It is deliberately conservative: if the inputs are missing or ambiguous, it
prefers fewer bots, higher thresholds, and a smaller universe.

Outputs
-------
Writes into rolling["_GLOBAL_DT"]["daily_plan_dt"] and returns the plan dict.

The plan is also intended to be persisted into dt_state.json via
  dt_backend.services.dt_truth_store.update_dt_state

v1.1 additions
--------------
Candidate universe support:
  - You can constrain the daily universe to a pre-approved list.
  - Sources (highest priority first):
      1) DT_CANDIDATE_UNIVERSE (comma-separated)
      2) DT_CANDIDATE_UNIVERSE_FILE / DT_UNIVERSE_FILE (txt/csv/json)
      3) None -> use all symbols present in rolling

File formats:
  - .txt / .csv: one symbol per line (or comma-separated)
  - .json: either a list ["AAPL", ...] or {"symbols": [...]} or {"universe": [...]}.

Safety:
  - Best-effort; never crashes the loop.
  - If candidate filtering produces too few symbols, we fall back to rolling.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path
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


# ----------------------
# Candidate universe I/O
# ----------------------

def _split_syms(text: str) -> List[str]:
    parts: List[str] = []
    for line in (text or "").replace("\r", "\n").split("\n"):
        for tok in line.replace(";", ",").split(","):
            s = (tok or "").strip().upper()
            if not s:
                continue
            # basic sanity: tickers usually alnum + . -
            ok = all(ch.isalnum() or ch in {".", "-"} for ch in s)
            if ok:
                parts.append(s)
    return parts


def _load_candidate_universe() -> Tuple[Optional[List[str]], str]:
    """Return (symbols or None, source_label)."""

    # 1) env list
    env_list = (os.getenv("DT_CANDIDATE_UNIVERSE", "") or "").strip()
    if env_list:
        syms = sorted(set(_split_syms(env_list)))
        return (syms if syms else None), "env:DT_CANDIDATE_UNIVERSE"

    # 2) file
    path_raw = (
        (os.getenv("DT_CANDIDATE_UNIVERSE_FILE", "") or "").strip()
        or (os.getenv("DT_UNIVERSE_FILE", "") or "").strip()
    )
    if not path_raw:
        return None, "rolling"

    try:
        p = Path(path_raw)
        if not p.exists() or not p.is_file():
            return None, f"missing:{p}"

        if p.suffix.lower() == ".json":
            import json

            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                syms = [str(x).strip().upper() for x in raw]
            elif isinstance(raw, dict):
                cand = raw.get("symbols") or raw.get("universe") or raw.get("tickers")
                if isinstance(cand, list):
                    syms = [str(x).strip().upper() for x in cand]
                else:
                    syms = []
            else:
                syms = []
            syms = sorted(set([s for s in syms if s]))
            return (syms if syms else None), f"file:{p.name}"

        # txt/csv/etc
        text = p.read_text(encoding="utf-8", errors="ignore")
        syms = sorted(set(_split_syms(text)))
        return (syms if syms else None), f"file:{p.name}"

    except Exception as e:
        return None, f"error:{e}"


# ----------------------
# Universe scoring
# ----------------------

def _latest_bar_volume(node: Dict[str, Any]) -> float:
    # Prefer 5m bars if present; fallback to 1m
    bars = node.get("bars_intraday_5m")
    if not isinstance(bars, list) or not bars:
        bars = node.get("bars_intraday")
    if not isinstance(bars, list) or not bars:
        return 0.0
    b = bars[-1]
    if not isinstance(b, dict):
        return 0.0
    return _f(b.get("v") or b.get("volume"), 0.0)


def _symbol_liquidity_score(sym: str, node: Dict[str, Any]) -> float:
    """Approximate dollar-volume score using last_price * latest volume * rel_volume."""
    feat = node.get("features_dt") if isinstance(node.get("features_dt"), dict) else {}
    last_px = _f(feat.get("last_price"), 0.0)

    # Fallback: infer last_px from bars if features are missing
    if last_px <= 0:
        bars = node.get("bars_intraday_5m")
        if not isinstance(bars, list) or not bars:
            bars = node.get("bars_intraday")
        if isinstance(bars, list) and bars:
            b = bars[-1]
            if isinstance(b, dict):
                last_px = _f(b.get("c") or b.get("close") or b.get("price"), 0.0)

    if last_px <= 0:
        return 0.0

    vol = _latest_bar_volume(node)
    rel_vol = _f(feat.get("rel_volume"), 1.0)
    return float(last_px * vol * max(0.25, rel_vol))


def _build_universe(rolling: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """Select a tradable universe for today.

    Returns (universe, meta).
    """
    max_n = _env_int("DT_UNIVERSE_SIZE", 60)
    min_price = _env_float("DT_UNIVERSE_MIN_PRICE", 2.0)
    min_atr_pct = _env_float("DT_UNIVERSE_MIN_ATR_PCT", 0.0015)  # 0.15%
    max_atr_pct = _env_float("DT_UNIVERSE_MAX_ATR_PCT", 0.0800)  # 8%

    candidates_src, source_label = _load_candidate_universe()
    candidate_set = set(candidates_src or []) if candidates_src else None

    # Always include market proxies if present
    proxies = [s.strip().upper() for s in (os.getenv("DT_MARKET_PROXIES", "SPY,QQQ").split(",")) if s.strip()]
    proxy_set = set(proxies)

    # Filter to symbols present in rolling (prevents requesting data we don't have)
    rolling_syms = [s for s in rolling.keys() if isinstance(s, str) and not s.startswith("_")]
    rolling_set = set([s.upper() for s in rolling_syms])

    if candidate_set is not None:
        filtered = sorted(candidate_set & rolling_set)
        # If candidate set is too restrictive, fall back to rolling.
        min_keep = _env_int("DT_UNIVERSE_MIN_CANDIDATES", 10)
        if len(filtered) < min_keep:
            candidate_set = None
            source_label = f"fallback:rolling(min_candidates={len(filtered)})"
        else:
            candidate_set = set(filtered)

    pool = candidate_set if candidate_set is not None else rolling_set

    scored: List[Tuple[str, float]] = []

    for sym in pool:
        node = rolling.get(sym)
        if not isinstance(node, dict):
            continue

        feat = node.get("features_dt") if isinstance(node.get("features_dt"), dict) else {}
        last_px = _f(feat.get("last_price"), 0.0)
        atr_v = _f(feat.get("atr_14"), 0.0)

        # Fallbacks if features aren't ready yet
        if last_px <= 0:
            bars = node.get("bars_intraday_5m")
            if not isinstance(bars, list) or not bars:
                bars = node.get("bars_intraday")
            if isinstance(bars, list) and bars:
                b = bars[-1]
                if isinstance(b, dict):
                    last_px = _f(b.get("c") or b.get("close") or b.get("price"), 0.0)

        if last_px < min_price:
            continue

        # If ATR isn't available yet, don't over-filter; just treat atr_pct as unknown.
        atr_pct = (atr_v / last_px) if (last_px > 0 and atr_v > 0) else None
        if atr_pct is not None:
            if atr_pct < min_atr_pct or atr_pct > max_atr_pct:
                continue

        score = _symbol_liquidity_score(sym, node)
        if score <= 0:
            continue
        scored.append((sym, score))

    scored.sort(key=lambda t: t[1], reverse=True)
    universe = [sym for sym, _ in scored[: max(0, max_n)]]

    # Ensure proxies appended (useful for market context)
    for p in proxies:
        if p and p in rolling_set and p not in universe and p in proxy_set:
            universe.append(p)

    meta = {
        "source": source_label,
        "candidate_count": (len(candidates_src) if candidates_src else None),
        "rolling_count": len(rolling_set),
        "pool_count": len(pool),
        "selected": len(universe),
    }
    return universe, meta


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
        enabled = [b for b in enabled if _env_bool(f"DT_ENABLE_{b}", True)]

    if not enabled:
        enabled = ["VWAP_MR"]

    risk_mode = _risk_mode_from_regime(label, conf)

    base = {b: 1.0 for b in SUPPORTED_BOTS}
    weights = _merge_weights(base, regime_weights)

    if risk_mode == "CONSERVATIVE":
        weights["VWAP_MR"] *= 1.15
        weights["ORB"] *= 0.85
        weights["SQUEEZE"] *= 0.90
    elif risk_mode == "AGGRESSIVE":
        weights["ORB"] *= 1.10
        weights["TREND_PULLBACK"] *= 1.10

    enabled_w = {b: float(_clamp(weights.get(b, 1.0), 0.0, 3.0)) for b in enabled}
    s = sum(enabled_w.values())
    if s > 0:
        enabled_w = {b: v / s for b, v in enabled_w.items()}

    # Phase 4.5: optional bandit overlay
    try:
        bandit_on = _env_bool("DT_BANDIT_ENABLED", False)
        shadow_only = _env_bool("DT_BANDIT_SHADOW_ONLY", True)
        shadow_enabled = _env_bool("DT_SHADOW_ENABLED", False)
        if bandit_on and suggest_bot_weights is not None and (not shadow_only or shadow_enabled):
            context = {"regime": label, "regime_conf": conf, "risk_mode": risk_mode}
            bw = suggest_bot_weights(context=context)
            if isinstance(bw, dict) and bw:
                mixed = {b: float(enabled_w.get(b, 0.0)) * float(bw.get(b, 0.0) or 0.0) for b in enabled_w.keys()}
                s2 = sum(mixed.values())
                if s2 > 0:
                    enabled_w = {b: v / s2 for b, v in mixed.items()}
    except Exception:
        pass

    universe, uni_meta = _build_universe(rolling)

    allow_model_fallback = _env_bool("DT_ALLOW_MODEL_FALLBACK", False)

    return {
        "date": (str(date_override) if date_override else _now_local_date_str()),
        "ts": _utc_iso(),
        "regime": {"label": label, "confidence": conf},
        "risk_mode": risk_mode,
        "enabled_bots": enabled,
        "bot_weights": enabled_w,
        "universe": universe,
        "universe_meta": uni_meta,
        "allow_model_fallback": bool(allow_model_fallback),
        "reason": f"meta: regime={label} conf={conf:.2f} risk={risk_mode} enabled={','.join(enabled)}",
        "version": "phase4_v1.1",
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
