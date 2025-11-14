"""
insights_router.py — v1.3 (FastAPI Insights Delivery)
Author: AION Analytics / StockAnalyzerPro

Purpose:
- Exposes REST endpoints for retrieving AI-generated stock insights.
- Delegates heavy computation to insights_builder.py.
- Automatically falls back to Rolling cache if builder fails.
"""

from __future__ import annotations
from datetime import datetime
from fastapi import APIRouter, Query
from backend.insights_builder import build_daily_insights
from backend.data_pipeline import _read_rolling, log

router = APIRouter(prefix="/insights", tags=["Insights"])


@router.get("/")
async def get_insights(
    limit: int = Query(50, description="Number of top insights to return."),
    sort_by: str = Query(
        "score",
        description="Sorting key: score | confidence | expectedReturnPct | predictedPrice",
    ),
    sector: str | None = Query(
        None, description="Optional sector filter (e.g., 'Technology' or 'Healthcare')."
    ),
):
    """
    Returns latest ranked AI insights (Top N) from the nightly pipeline.

    If the builder is unavailable or fails, falls back to reading the Rolling cache
    and returns a simplified structure.

    Returns:
        dict: {timestamp, count, insights, source}
    """
    try:
        # Primary: use the insights builder output (already cached)
        log(f"[insights_router] 🚀 Building insights (limit={limit}, sort_by={sort_by}, sector={sector})")
        res = build_daily_insights(limit=limit)
        insights = []

        # Flatten builder outputs for API use
        if isinstance(res, dict) and "outputs" in res:
            for h, meta in res["outputs"].items():
                insights.append({
                    "horizon": h,
                    "count": meta.get("count"),
                    "path": meta.get("path"),
                })

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "count": sum((i.get("count") or 0) for i in insights),
            "insights": insights,
            "source": "builder",
        }

    except Exception as e:
        # Fallback: pull directly from Rolling cache
        log(f"⚠️ insights_router fallback due to builder error: {e}")
        rolling = _read_rolling() or {}
        rows = []
        for sym, node in list(rolling.items())[:limit]:
            if not isinstance(node, dict):
                continue
            preds = node.get("predictions", {})
            if sector and node.get("sector") and node.get("sector") != sector:
                continue
            rows.append({
                "symbol": sym,
                "name": node.get("name"),
                "sector": node.get("sector"),
                "predictedPrice": preds.get("predictedPrice") or preds.get("price_target"),
                "expectedReturnPct": preds.get("expectedReturnPct"),
                "confidence": preds.get("confidence"),
                "score": preds.get("score"),
            })

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "count": len(rows),
            "insights": rows,
            "source": "rolling_fallback",
            "note": f"⚠️ Fallback used — build_daily_insights() failed with: {e}",
        }
