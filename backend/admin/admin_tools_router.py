from __future__ import annotations

import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from backend.admin.auth import require_admin
from utils.live_log import tail_lines

router = APIRouter(prefix="/admin/tools", tags=["admin-tools"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Paths ---
SWING_REPLAY_STATE = PROJECT_ROOT / "data" / "replay" / "swing" / "replay_state.json"

LOCK_PATHS = [
    PROJECT_ROOT / "data" / "locks",
    PROJECT_ROOT / "data" / "replay" / "locks",
]


# --- Logs ---
@router.get("/logs")
def get_logs(_: str = Depends(require_admin)):
    return {
        "lines": tail_lines(300)
    }


# --- Clear locks + reset replay state ---
@router.post("/clear-locks")
def clear_locks(_: str = Depends(require_admin)):
    removed = []

    # Remove lock files
    for lock_dir in LOCK_PATHS:
        if lock_dir.exists():
            for p in lock_dir.glob("*"):
                try:
                    p.unlink()
                    removed.append(str(p))
                except Exception:
                    pass

    # Reset swing replay state
    if SWING_REPLAY_STATE.exists():
        clean_state = {
            "status": "idle",
            "version": "v1",
            "started_at": None,
            "finished_at": None,
            "start_date": None,
            "end_date": None,
            "current_day": "",
            "completed_days": [],
            "days_completed": 0,
            "total_days": 0,
            "percent_complete": 0.0,
            "elapsed_secs": 0.0,
            "eta_secs": None,
            "last_error": None,
            "notes": [],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(SWING_REPLAY_STATE, "w") as f:
            json.dump(clean_state, f, indent=2)

    return {
        "status": "ok",
        "removed": removed,
        "replay_state_reset": True,
    }


# --- Git pull ---
@router.post("/git-pull")
def git_pull(_: str = Depends(require_admin)):
    try:
        result = subprocess.check_output(
            ["git", "pull"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.output)

    return {
        "status": "ok",
        "output": result,
    }
