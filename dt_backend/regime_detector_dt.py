"""
regime_detector_dt.py — v1.1
Infers intraday regime (per-ticker + global) from volatility, breadth, and short-term trends.
Writes: node.context.regime, node.context.regime_conf
"""
from __future__ import annotations
import json, os
from typing import Dict, Any
from dt_backend.config_dt import DT_PATHS
from dt_backend.data_pipeline_dt import _read_dt_rolling as _read_rolling, save_dt_rolling as save_rolling, log

GLBL_PATH = DT_PATHS["dtml_data"] / "market_state.json"
GLBL_PATH.parent.mkdir(parents=True, exist_ok=True)

def _load_glbl() -> Dict[str, Any]:
    try:
        with open(GLBL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"market_state": "neutral", "macro_vol": 0.5}

def _assign_regime(node_ctx: Dict[str, Any], glbl: Dict[str, Any]):
    macro_vol = float(node_ctx.get("macro_vol", glbl.get("macro_vol", 0.5)) or 0.5)
    trend     = (node_ctx.get("trend") or "neutral").lower()
    news_st   = float(node_ctx.get("news_stance", 0.0) or 0.0)

    score = 0.0
    score += 0.7 if trend == "bullish" else (-0.7 if trend == "bearish" else 0.0)
    score += 0.5 * news_st
    score -= 0.8 * (macro_vol - 0.5) * 2

    if macro_vol > 0.75:
        regime = "high_vol"
    elif score > 0.6:
        regime = "trending"
    elif abs(score) < 0.25:
        regime = "choppy"
    elif score < -0.6:
        regime = "panic"
    else:
        regime = "neutral"

    conf = max(0.0, min(1.0, 0.5 + 0.5 * (abs(score) / 1.5)))
    return regime, round(conf, 3)

def run() -> Dict[str, Any]:
    glbl = _load_glbl()
    rolling = _read_rolling() or {}
    count = 0
    for sym, node in (rolling or {}).items():
        ctx = dict(node.get("context") or {})
        regime, conf = _assign_regime(ctx, glbl)
        ctx["regime"] = regime
        ctx["regime_conf"] = conf
        node["context"] = ctx
        rolling[sym] = node
        count += 1
    save_rolling(rolling)
    log(f"[regime_detector_dt] ✅ intraday regimes updated for {count:,} symbols (global={glbl.get('market_state','neutral')})")
    return {"symbols": count, "global": glbl}
