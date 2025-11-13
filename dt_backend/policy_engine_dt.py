# dt_backend/policy_engine_dt.py — v1.2 (Intraday Policy Engine, isolated for DT rolling)
from __future__ import annotations
from typing import Dict, Any
import math
from dt_backend.data_pipeline_dt import (
    _read_rolling as _read_rolling,
   save_dt_rolling as save_rolling,
    log,
)

# =====================================================================
# INTERNAL HELPERS
# =====================================================================

def _sigmoid(x: float) -> float:
    """Smooth scaling between 0..1, robust to overflow."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5


def _scale_from_context(row: dict, ctx: Any) -> float:
    """Compute exposure scaling from model score + context."""
    # ✅ guard against ctx being invalid
    if not isinstance(ctx, dict):
        ctx = {}

    # Base from model quality
    rs = float(row.get("rankingScore", row.get("score", 0.0)) or 0.0)
    conf = float(row.get("confidence", 0.5) or 0.5)
    base = 0.5 + 0.5 * _sigmoid((rs - 0.0) * 2.0)  # 0.5..1.0
    base *= 0.5 + 0.5 * conf                        # 0.25..1.0

    # Context nudges
    stance = float(ctx.get("news_stance", 0.0) or 0.0)  # -1..+1
    mstate = (ctx.get("market_state") or "neutral").lower()
    if mstate == "risk_on":
        risk_on = 1.0
    elif mstate == "neutral":
        risk_on = 0.85
    else:
        risk_on = 0.7

    macro_vol = float(ctx.get("macro_vol", 0.0) or 0.0)
    vol_pen = 1.0 - 0.4 * macro_vol
    stance_boost = 1.0 + 0.2 * stance

    return max(0.1, min(2.0, base * risk_on * vol_pen * stance_boost))


# =====================================================================
# MAIN POLICY APPLY FUNCTION
# =====================================================================

def apply(rolling: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Applies dynamic policy scaling for intraday context.
    Reads/writes exclusively to data_dt/rolling_intraday.json.gz.
    """
    if rolling is None:
        rolling = _read_rolling() or {}

    out = {}
    for sym, node in (rolling or {}).items():
        if not isinstance(node, dict):
            continue

        preds = node.get("predictions") or {}
        if not preds:
            continue

        ctx = node.get("context")
        if not isinstance(ctx, dict):
            ctx = {"market_state": str(ctx)} if ctx else {}

        # --- Pick best horizon prediction ---
        best = None
        for h in ("1w", "2w", "4w"):
            p = preds.get(h)
            if not p:
                continue
            if best is None or p.get("rankingScore", 0) > best.get("rankingScore", 0):
                best = p

        if not best:
            continue

        # --- Compute dynamic scaling ---
        scale = _scale_from_context(best, ctx)
        regime = (ctx.get("regime") or "neutral").lower()
        conf = float(best.get("confidence", 0.5) or 0.5)
        buzz = float(node.get("buzz", ctx.get("buzz", 0)) or 0)
        macro_vol = float(ctx.get("macro_vol", 0.0) or 0.0)

        # --- Safety gates ---
        trade_gate = not (regime in {"panic", "high_vol"} and conf < 0.55)
        if macro_vol > 0.7 or buzz > 50:
            scale *= 0.6

        pol = {
            "exposure_scale": round(scale, 3),
            "trade_gate": bool(trade_gate),
            "max_risk": 0.01 if macro_vol < 0.7 else 0.005,
            "kill_switch": False,
        }
        node["policy"] = {**(node.get("policy") or {}), **pol}
        out[sym] = node

    # --- Save and log ---
    save_rolling(rolling)
    log(f"[policy_engine_dt] ✅ policy computed for {len(out):,} symbols")
    return rolling


# =====================================================================
# MANUAL TEST
# =====================================================================
if __name__ == "__main__":
    data = _read_rolling()
    updated = apply(data)
    print(f"✅ Applied intraday policy scaling for {len(updated)} symbols.")
