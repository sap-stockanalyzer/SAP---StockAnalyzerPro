"""dt_backend/routers/replay_extended_router.py

REST API endpoints for intra-day replay and walk-forward validation.

Endpoints:
- POST /api/replay/cycle/{cycle_id} - Replay a cycle with modified knobs
- GET /api/replay/results - Get recent replay results
- POST /api/replay/walk-forward - Run walk-forward validation
- GET /api/replay/decisions/{cycle_id} - Get all decisions from a cycle
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime, timezone

router = APIRouter(prefix="/api/replay", tags=["replay"])


@router.post("/cycle/{cycle_id}")
def replay_cycle(
    cycle_id: str,
    modified_knobs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Replay a historical cycle with optional modified knobs.
    
    This endpoint allows you to test "what if" scenarios by replaying
    a past trading cycle with different parameter settings.
    
    Args:
        cycle_id: The ID of the cycle to replay
        modified_knobs: Optional dict of knobs to override
                       Example: {"stop_loss_pct": 0.015, "take_profit_pct": 0.08}
    
    Returns:
        Dict with comparison of original vs replay P&L and improvement metrics
        
    Example:
        POST /api/replay/cycle/abc123
        Body: {"stop_loss_pct": 0.02, "take_profit_pct": 0.05}
    """
    from dt_backend.replay.intraday_replay_engine import IntraDayReplayEngine
    
    try:
        engine = IntraDayReplayEngine()
        result = engine.replay_cycle(cycle_id, modified_knobs=modified_knobs)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Replay failed: {str(e)}")


@router.get("/results")
def get_replay_results(days: int = Query(7, ge=1, le=30)) -> Dict[str, Any]:
    """Get recent replay results.
    
    Retrieves all replay comparisons from the last N days.
    
    Args:
        days: Number of days to look back (1-30, default 7)
    
    Returns:
        Dict with count and list of replay results
        
    Example:
        GET /api/replay/results?days=14
    """
    from dt_backend.replay.intraday_replay_engine import IntraDayReplayEngine
    
    try:
        engine = IntraDayReplayEngine()
        results = engine.get_replay_results(days=days)
        
        return {
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get results: {str(e)}")


@router.post("/walk-forward")
def run_walk_forward_validation(
    days_back: int = Query(60, ge=7, le=365),
) -> Dict[str, Any]:
    """Run walk-forward validation on historical data.
    
    Evaluates strategy performance on rolling train/test windows
    to assess consistency and robustness.
    
    Args:
        days_back: Number of days of history to validate (7-365, default 60)
    
    Returns:
        Summary dict with aggregated metrics across all windows:
        - windows: Number of test windows
        - total_pnl: Aggregate P&L
        - avg_sharpe: Average Sharpe ratio
        - avg_win_rate: Average win rate
        - consistent: Whether all windows were profitable
        
    Example:
        POST /api/replay/walk-forward?days_back=90
    """
    from dt_backend.ml.walk_forward_validator import WalkForwardValidator
    
    try:
        validator = WalkForwardValidator()
        summary = validator.run_validation(days_back=days_back)
        
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/decisions/{cycle_id}")
def get_cycle_decisions(cycle_id: str) -> Dict[str, Any]:
    """Get all decisions recorded for a specific cycle.
    
    Returns the complete decision log for a trading cycle including
    symbol selection, entries, exits, and all associated metrics.
    
    Args:
        cycle_id: The cycle ID to retrieve decisions for
    
    Returns:
        Dict with cycle_id, decisions_count, and list of all decisions
        
    Example:
        GET /api/replay/decisions/abc123
    """
    from dt_backend.services.decision_recorder import DecisionRecorder
    
    try:
        recorder = DecisionRecorder()
        decisions = recorder.get_cycle_decisions(cycle_id)
        
        return {
            "cycle_id": cycle_id,
            "decisions_count": len(decisions),
            "decisions": decisions,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get decisions: {str(e)}")
