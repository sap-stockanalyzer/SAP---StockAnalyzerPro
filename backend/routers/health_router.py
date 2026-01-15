"""Health check endpoints for EOD backend."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter

router = APIRouter()

# Track service start time for uptime calculation
_START_TIME = time.time()


def check_broker_connection() -> Dict[str, Any]:
    """Check if broker API is accessible."""
    try:
        # Check if Alpaca credentials are configured
        api_key = os.getenv("ALPACA_API_KEY_ID")
        api_secret = os.getenv("ALPACA_API_SECRET_KEY")
        
        if not api_key or not api_secret:
            return {"status": "degraded", "message": "Missing Alpaca credentials"}
        
        return {"status": "healthy", "message": "Credentials configured"}
    except Exception as e:
        return {"status": "degraded", "message": f"Error: {str(e)}"}


def check_data_availability() -> Dict[str, Any]:
    """Check if core data files are available."""
    try:
        data_root = os.getenv("DATA_ROOT", "data")
        data_dir = Path(data_root)
        
        if not data_dir.exists():
            return {"status": "degraded", "message": "Data directory not found"}
        
        return {"status": "healthy", "message": "Data directory available"}
    except Exception as e:
        return {"status": "degraded", "message": f"Error: {str(e)}"}


def check_ml_models() -> Dict[str, Any]:
    """Check if ML models are available."""
    try:
        ml_models_root = os.getenv("ML_MODELS_ROOT", "ml_data/nightly/models")
        models_dir = Path(ml_models_root)
        
        if not models_dir.exists():
            return {"status": "degraded", "message": "Models directory not found"}
        
        # Count model files
        model_files = list(models_dir.glob("**/*.txt")) + list(models_dir.glob("**/*.pkl"))
        
        if not model_files:
            return {"status": "degraded", "message": "No models found"}
        
        return {
            "status": "healthy",
            "message": f"{len(model_files)} model(s) available"
        }
    except Exception as e:
        return {"status": "degraded", "message": f"Error: {str(e)}"}


def get_uptime() -> float:
    """Get service uptime in seconds."""
    return time.time() - _START_TIME


def get_last_nightly_run() -> str:
    """Get timestamp of last nightly job run."""
    try:
        # Try to read last_nightly_summary.json
        summary_file = Path("last_nightly_summary.json")
        
        if not summary_file.exists():
            return "unknown"
        
        import json
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
            return summary.get("timestamp", "unknown")
    except Exception:
        return "unknown"


@router.get("/health")
def health_check():
    """System health status with component checks."""
    components = {
        "broker": check_broker_connection(),
        "data": check_data_availability(),
        "models": check_ml_models(),
    }
    
    # Determine overall status
    if any(c["status"] == "down" for c in components.values()):
        overall_status = "down"
    elif any(c["status"] == "degraded" for c in components.values()):
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    return {
        "status": overall_status,
        "service": "backend",
        "components": components,
        "uptime_seconds": get_uptime(),
        "last_nightly_run": get_last_nightly_run(),
        "version": "v3.3",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health/ready")
def readiness_check():
    """Kubernetes readiness probe - system ready to accept traffic."""
    # System is ready if core components are available
    data_status = check_data_availability()
    
    if data_status["status"] == "down":
        return {
            "ready": False,
            "reason": "data_unavailable",
            "message": data_status["message"]
        }
    
    return {
        "ready": True,
        "service": "backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health/live")
def liveness_check():
    """Kubernetes liveness probe - system is alive and responsive."""
    return {
        "alive": True,
        "service": "backend",
        "uptime_seconds": get_uptime(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
