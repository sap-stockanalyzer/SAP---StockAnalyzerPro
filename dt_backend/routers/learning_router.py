"""
dt_backend/routers/learning_router.py

API endpoints for day trading learning metrics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/dt/learning", tags=["dt-learning"])

try:
    from dt_backend.core.config_dt import DT_PATHS
    from dt_backend.core.logger_dt import log
    from dt_backend.ml.trade_outcome_analyzer import TradeOutcomeAnalyzer
    from dt_backend.ml.missed_opportunity_tracker import MissedOpportunityTracker
    from dt_backend.core.dt_brain import read_dt_brain, KNOB_CONFIG
    from dt_backend.services.auto_retrain_dt import AutoRetrainSystem
except Exception as e:
    DT_PATHS = {}
    TradeOutcomeAnalyzer = None
    MissedOpportunityTracker = None
    KNOB_CONFIG = {}
    def log(msg: str) -> None:
        print(msg, flush=True)
    def read_dt_brain():
        return {}


def _get_learning_path() -> Path:
    """Get learning data path."""
    learning_path = DT_PATHS.get("learning")
    if learning_path:
        return Path(learning_path)
    da_brains = DT_PATHS.get("da_brains", Path("da_brains"))
    return Path(da_brains) / "dt_learning"


@router.get("/metrics")
def get_learning_metrics(
    days_7d: int = Query(7, description="Days for 7-day window"),
    days_30d: int = Query(30, description="Days for 30-day window"),
) -> Dict[str, Any]:
    """Get current learning metrics for dashboard.
    
    Returns:
    - Performance trends (7d, 30d, all-time)
    - Model health (confidence calibration, days since retrain)
    - Missed opportunities summary
    - DT Brain knob values vs defaults
    - Trade quality breakdown
    """
    try:
        if TradeOutcomeAnalyzer is None:
            raise HTTPException(status_code=503, detail="TradeOutcomeAnalyzer not available")
        
        analyzer = TradeOutcomeAnalyzer(_get_learning_path())
        
        # Get performance windows
        perf_7d = analyzer.get_performance_window(days=days_7d)
        perf_30d = analyzer.get_performance_window(days=days_30d)
        baseline = analyzer.get_baseline_performance()
        
        # Get days since retrain
        retrain_system = AutoRetrainSystem()
        days_since_retrain = retrain_system._days_since_last_retrain()
        
        # Get missed opportunities
        missed_tracker = MissedOpportunityTracker(_get_learning_path())
        missed_patterns = missed_tracker.analyze_missed_patterns()
        
        # Get DT Brain knobs
        brain = read_dt_brain()
        knobs = brain.get("knobs", {})
        
        knob_status = {}
        for knob_name, config in KNOB_CONFIG.items():
            current = knobs.get(knob_name, config["default"])
            default = config["default"]
            knob_status[knob_name] = {
                "current": current,
                "default": default,
                "range": config["range"],
                "diff": current - default,
                "diff_pct": ((current - default) / default * 100) if default != 0 else 0,
            }
        
        return {
            "performance_7d": perf_7d,
            "performance_30d": perf_30d,
            "baseline": baseline,
            "model_health": {
                "days_since_retrain": days_since_retrain,
                "confidence_calibration": perf_7d.get("accuracy", 0.0) / perf_7d.get("avg_confidence", 1.0) if perf_7d.get("avg_confidence", 0) > 0 else 0.0,
                "next_retrain_estimate": "7 days" if days_since_retrain < 7 else "Due now",
            },
            "missed_opportunities": missed_patterns,
            "dt_brain_knobs": knob_status,
            "trade_quality": {
                "total_trades_7d": perf_7d.get("total_trades", 0),
                "win_rate_7d": perf_7d.get("win_rate", 0.0),
                "profit_factor_7d": perf_7d.get("profit_factor", 0.0),
                "sharpe_ratio_7d": perf_7d.get("sharpe_ratio", 0.0),
            },
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log(f"[learning_router] Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trade-outcomes")
def get_trade_outcomes(days: int = Query(7, ge=1, le=90)) -> Dict[str, Any]:
    """Get recent trade outcomes for analysis."""
    try:
        if TradeOutcomeAnalyzer is None:
            raise HTTPException(status_code=503, detail="TradeOutcomeAnalyzer not available")
        
        analyzer = TradeOutcomeAnalyzer(_get_learning_path())
        trades = analyzer._read_recent_trades(days=days)
        
        return {
            "trades": trades,
            "count": len(trades),
            "days": days,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log(f"[learning_router] Error getting trade outcomes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/missed-opportunities")
def get_missed_opportunities(days: int = Query(7, ge=1, le=90)) -> Dict[str, Any]:
    """Get missed opportunities analysis."""
    try:
        if MissedOpportunityTracker is None:
            raise HTTPException(status_code=503, detail="MissedOpportunityTracker not available")
        
        tracker = MissedOpportunityTracker(_get_learning_path())
        
        # Get unevaluated signals
        unevaluated = tracker._read_unevaluated_signals(days=days)
        
        # Get analysis
        patterns = tracker.analyze_missed_patterns()
        
        return {
            "unevaluated_count": len(unevaluated),
            "patterns": patterns,
            "days": days,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log(f"[learning_router] Error getting missed opportunities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/brain-status")
def get_dt_brain_status() -> Dict[str, Any]:
    """Get current DT Brain knob values and recent adjustments."""
    try:
        brain = read_dt_brain()
        knobs = brain.get("knobs", {})
        
        # Read recent adjustments
        learning_path = _get_learning_path()
        adjustments_file = learning_path / "knob_adjustments.jsonl"
        
        recent_adjustments = []
        if adjustments_file.exists():
            try:
                with open(adjustments_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    # Get last 20 adjustments
                    for line in lines[-20:]:
                        if line.strip():
                            recent_adjustments.append(json.loads(line))
            except Exception:
                pass
        
        knob_status = {}
        for knob_name, config in KNOB_CONFIG.items():
            current = knobs.get(knob_name, config["default"])
            knob_status[knob_name] = {
                "current": current,
                "default": config["default"],
                "range": config["range"],
            }
        
        return {
            "knobs": knob_status,
            "recent_adjustments": recent_adjustments,
            "last_update": brain.get("_meta", {}).get("updated_at"),
        }
        
    except Exception as e:
        log(f"[learning_router] Error getting brain status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain")
def trigger_manual_retrain() -> Dict[str, Any]:
    """Manually trigger model retraining."""
    try:
        from dt_backend.services.auto_retrain_dt import AutoRetrainSystem
        
        system = AutoRetrainSystem()
        result = system.retrain_intraday_models()
        
        return result
        
    except Exception as e:
        log(f"[learning_router] Error triggering retrain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance-history")
def get_performance_history(days: int = Query(30, ge=1, le=365)) -> Dict[str, Any]:
    """Get historical performance metrics."""
    try:
        if TradeOutcomeAnalyzer is None:
            raise HTTPException(status_code=503, detail="TradeOutcomeAnalyzer not available")
        
        analyzer = TradeOutcomeAnalyzer(_get_learning_path())
        
        # Get metrics over time
        # For now, return aggregated data
        metrics = analyzer._load_metrics()
        
        return {
            "global_metrics": metrics.get("global", {}),
            "by_symbol": metrics.get("by_symbol", {}),
            "by_regime": metrics.get("by_regime", {}),
            "by_time_of_day": metrics.get("by_time_of_day", {}),
            "confidence_buckets": metrics.get("confidence_buckets", {}),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log(f"[learning_router] Error getting performance history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def learning_health_check() -> Dict[str, Any]:
    """Health check for learning system."""
    return {
        "status": "healthy",
        "analyzer_available": TradeOutcomeAnalyzer is not None,
        "tracker_available": MissedOpportunityTracker is not None,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
