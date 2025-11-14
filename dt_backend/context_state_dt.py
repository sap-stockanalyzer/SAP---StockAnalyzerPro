"""
context_state_dt.py — v1.2
Builds per-ticker and global intraday context features.
Safe to run standalone or inside daytrading_job.
Outputs:
  • context.* dicts written into intraday rolling cache
  • ml_data_dt/market_state.json (intraday market regime snapshot)
"""
from __future__ import annotations
import os, json
from typing import Dict, Any
from datetime import datetime

from dt_backend.config_dt import DT_PATHS
from dt_backend.data_pipeline_dt import _read_dt_rolling as _read_rolling, save_dt_rolling as save_rolling, log

GLBL_PATH = DT_PATHS["dtml_data"] / "market_state.json"
NEWS_DIR  = DT_PATHS["dtnews"]
MACRO_DIR = DT_PATHS["dtmacro"]

NEWS_DIR.mkdir(parents=True, exist_ok=True)
MACRO_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Utility helpers ----------------
def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _latest_with_prefix(folder, prefix):
    try:
        cands = [p for p in os.listdir(folder) if p.startswith(prefix) and p.endswith(".json")]
        if not cands:
            return None
        cands.sort(key=lambda n: os.path.getmtime(os.path.join(folder, n)), reverse=True)
        return os.path.join(folder, cands[0])
    except Exception:
        return None

def _clamp(x, a=0.0, b=1.0):
    try:
        return max(a, min(b, float(x)))
    except Exception:
        return (a + b) / 2

# ---------------- Core logic ----------------
def _mk_global_state(macro_js: Dict[str, Any] | None) -> Dict[str, Any]:
    breadth = 0.0
    vol_z   = 0.0
    if isinstance(macro_js, dict):
        breadth = float(macro_js.get("breadth_pos", 0.0) or 0.0)
        vol_z   = float(macro_js.get("vol_z", 0.0) or macro_js.get("vix_z", 0.0) or 0.0)
    macro_vol = _clamp(0.5 + 0.1 * vol_z, 0.0, 1.0)
    if macro_vol > 0.65 or breadth < -10:
        state = "risk_off"
    elif abs(breadth) <= 5 and macro_vol <= 0.65:
        state = "neutral"
    else:
        state = "risk_on"
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "market_state": state,
        "macro_vol": round(macro_vol, 3),
        "breadth": breadth,
        "vol_z": vol_z,
    }

def _ticker_news_map(news_js: Dict[str, Any] | None) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not isinstance(news_js, dict):
        return out
    rows = news_js.get("per_ticker") or news_js.get("intel_per_ticker") or []
    for r in rows:
        t = r.get("ticker") or r.get("symbol")
        if not t:
            continue
        out.setdefault(t, {})
        out[t]["news_stance"] = float(r.get("stance", 0.0) or 0.0)
        out[t]["buzz"] = float(r.get("buzz", 0) or 0)
        for k in ("event_short_impulse", "event_mid_impulse", "event_long_impulse"):
            if k in r:
                out[t][k] = float(r.get(k) or 0.0)
    return out

# ---------------- Public update ----------------
def update() -> Dict[str, Any]:
    rolling = _read_rolling() or {}

    macro_path = _latest_with_prefix(MACRO_DIR, "macro_features_") or os.path.join(MACRO_DIR, "macro_features.json")
    macro_js = _load_json(macro_path) if macro_path and os.path.exists(macro_path) else None

    news_path = _latest_with_prefix(NEWS_DIR, "news_intel_") or os.path.join(NEWS_DIR, "news_intel_daily.json")
    news_js = _load_json(news_path) if news_path and os.path.exists(news_path) else None
    tnews = _ticker_news_map(news_js)

    # Merge social sentiment
    social_path = _latest_with_prefix(NEWS_DIR, "social_sentiment_")
    social_js = _load_json(social_path) if social_path and os.path.exists(social_path) else None
    if isinstance(social_js, dict):
        data_block = social_js.get("data") or social_js
        for sym, node in (data_block or {}).items():
            sym = sym.upper()
            if not isinstance(node, dict):
                continue
            sentiment = float(node.get("avg_sentiment", 0.0) or node.get("sentiment", 0.0))
            buzz = int(node.get("buzz", 0) or node.get("mentions", 0))
            if sym not in tnews:
                tnews[sym] = {}
            if "news_stance" in tnews[sym]:
                tnews[sym]["news_stance"] = round(
                    0.5 * float(tnews[sym]["news_stance"]) + 0.5 * sentiment, 4
                )
            else:
                tnews[sym]["news_stance"] = round(sentiment, 4)
            tnews[sym]["buzz"] = int(tnews[sym].get("buzz", 0) + buzz)

    glbl = _mk_global_state(
        macro_js if isinstance(macro_js, dict)
        else (macro_js[-1] if isinstance(macro_js, list) and macro_js else {})
    )

    for sym, node in (rolling or {}).items():
        ctx = dict(node.get("context") or {})
        if sym in tnews:
            ctx.update(tnews[sym])
        ctx.setdefault("market_state", glbl["market_state"])
        ctx.setdefault("macro_vol", glbl["macro_vol"])
        if "trend" in node:
            ctx.setdefault("trend", node.get("trend"))
        node["context"] = ctx
        rolling[sym] = node

    try:
        GLBL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GLBL_PATH, "w", encoding="utf-8") as f:
            json.dump(glbl, f, indent=2)
    except Exception as e:
        log(f"[context_state_dt] ⚠️ Failed to write global state: {e}")

    save_rolling(rolling)
    log(f"[context_state_dt] ✅ context updated for {len(rolling):,} symbols; global={glbl}")
    return {"symbols": len(rolling), "global": glbl}
