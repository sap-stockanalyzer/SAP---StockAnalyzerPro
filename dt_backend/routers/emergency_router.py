"""Emergency stop API endpoints for dt_backend."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class EmergencyStopRequest(BaseModel):
    """Request model for triggering emergency stop."""
    reason: str = "manual"


@router.post("/emergency/stop")
def emergency_stop(request: EmergencyStopRequest = EmergencyStopRequest()):
    """Trigger emergency stop to halt all trading.
    
    This creates a stop file that will prevent all trading operations
    until the stop is cleared. This is a safety mechanism for crisis situations.
    
    Args:
        request: Optional request body with reason
    
    Returns:
        dict: Status information
    """
    try:
        from dt_backend.risk.emergency_stop_dt import trigger_emergency_stop, check_emergency_stop
        
        reason = request.reason if request else "manual"
        trigger_emergency_stop(reason)
        
        # Verify it was created
        is_stopped, _ = check_emergency_stop()
        
        return {
            "status": "stopped",
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Emergency stop activated. All trading halted.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger emergency stop: {str(e)}")


@router.post("/emergency/resume")
def emergency_resume():
    """Clear emergency stop to resume trading.
    
    This removes the stop file, allowing trading operations to resume.
    Use with caution - ensure the crisis situation has been resolved.
    
    Returns:
        dict: Status information
    """
    try:
        from dt_backend.risk.emergency_stop_dt import clear_emergency_stop, check_emergency_stop
        
        # Check if stop is active
        is_stopped, reason = check_emergency_stop()
        
        if not is_stopped:
            return {
                "status": "not_stopped",
                "message": "Emergency stop was not active.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        
        clear_emergency_stop()
        
        return {
            "status": "resumed",
            "previous_reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Emergency stop cleared. Trading can resume.",
        }
    except FileNotFoundError:
        return {
            "status": "not_stopped",
            "message": "Emergency stop file does not exist.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear emergency stop: {str(e)}")


@router.get("/emergency/status")
def emergency_status():
    """Check emergency stop status.
    
    Returns current emergency stop status including whether it's active
    and the reason if it is.
    
    Returns:
        dict: Detailed status information
    """
    try:
        from dt_backend.risk.emergency_stop_dt import get_stop_status
        
        status = get_stop_status()
        status["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        if status["is_stopped"]:
            status["message"] = f"Emergency stop is ACTIVE. Reason: {status['reason']}"
        else:
            status["message"] = "Emergency stop is NOT active. Trading can proceed."
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check emergency stop status: {str(e)}")


@router.post("/emergency/test")
def emergency_test():
    """Test emergency stop mechanism (for verification only).
    
    This endpoint allows testing the emergency stop mechanism without
    actually halting trading. It checks if the stop file can be created
    and removed successfully.
    
    Returns:
        dict: Test results
    """
    try:
        from dt_backend.risk.emergency_stop_dt import (
            trigger_emergency_stop,
            check_emergency_stop,
            clear_emergency_stop,
        )
        
        results = {
            "test_create": False,
            "test_check": False,
            "test_clear": False,
            "test_passed": False,
        }
        
        # Test create
        try:
            trigger_emergency_stop("test")
            results["test_create"] = True
        except Exception as e:
            results["create_error"] = str(e)
            return results
        
        # Test check
        try:
            is_stopped, reason = check_emergency_stop()
            if is_stopped and reason == "test":
                results["test_check"] = True
        except Exception as e:
            results["check_error"] = str(e)
            return results
        
        # Test clear
        try:
            clear_emergency_stop()
            is_stopped, _ = check_emergency_stop()
            if not is_stopped:
                results["test_clear"] = True
        except Exception as e:
            results["clear_error"] = str(e)
            return results
        
        results["test_passed"] = all([
            results["test_create"],
            results["test_check"],
            results["test_clear"],
        ])
        
        results["message"] = "Emergency stop mechanism test passed" if results["test_passed"] else "Emergency stop mechanism test failed"
        results["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")
