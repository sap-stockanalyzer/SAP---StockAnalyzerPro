"""
dashboard_router.py — StockAnalyzerPro Dashboard API
v1.6 (Unified Config + Safe JSON Read)
Provides cached dashboard metrics and top-performer data
for frontend display (accuracy badge + top score cards).
"""

from fastapi import APIRouter
import os, json, datetime as dt
from .config import PATHS  # ✅ unified config import

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _read_json(path: os.PathLike):
    """Safely load JSON file from unified path."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@router.get("/metrics")
def get_dashboard_metrics():
    """
    Returns weighted accuracy metrics for the past 30 days.
    Cached in PATHS["dashboard"]/metrics.json (nightly or manual recompute).
    """
    path = PATHS["dashboard"] / "metrics.json"
    data = _read_json(path)
    if not data:
        return {"error": f"metrics.json not found at {path}"}
    data["api_timestamp"] = dt.datetime.utcnow().isoformat() + "Z"
    return data


@router.get("/top/{horizon}")
def get_top_performers(horizon: str):
    """
    Returns top performer tickers for 1w or 1m horizon.
    Each entry contains frozen predicted price and live current gain%.
    """
    if horizon not in ("1w", "1m"):
        return {"error": "Invalid horizon. Use 1w or 1m."}

    path = PATHS["dashboard"] / f"top_{horizon}.json"
    data = _read_json(path)
    if not data:
        return {"error": f"top_{horizon}.json not found at {path}"}

    data_out = {
        "horizon": horizon,
        "tickers": data,
        "api_timestamp": dt.datetime.utcnow().isoformat() + "Z",
    }
    return data_out
