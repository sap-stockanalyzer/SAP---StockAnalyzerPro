# /backend/admin/admin_tools_router.py

from __future__ import annotations

import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from backend.admin.deps import admin_required
from backend.core.config import PATHS
from utils.live_log import tail_lines

router = APIRouter(prefix="/admin/tools", tags=["admin-tools"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from backend.historical_replay_swing.job_manager import REPLAY_STATE_PATH
SWING_REPLAY_STATE = REPLAY_STATE_PATH

# ------------------------------------------------------------------
# Lock locations (DIRECTORIES)
# ------------------------------------------------------------------

LOCK_DIRS = [
    PROJECT_ROOT / "data" / "locks",
    PROJECT_ROOT / "data" / "replay" / "locks",
]

# ------------------------------------------------------------------
# Lock locations (EXPLICIT FILES)
# ------------------------------------------------------------------

LOCK_FILES = [
    PATHS.get("nightly_lock"),        # da_brains/nightly_job.lock
    PATHS.get("swing_replay_lock"),   # data/replay/swing/replay.lock
]

# --------------------------------------------------
# Logs
# --------------------------------------------------

@router.get("/logs")
def get_logs(_: None = Depends(admin_required)):
    return {"lines": tail_lines(300)}

# --------------------------------------------------
# Clear locks + reset replay
# --------------------------------------------------

@router.post("/clear-locks")
def clear_locks(_: None = Depends(admin_required)):
    removed: list[str] = []

    # 1️⃣ Remove lock files
    for lf in LOCK_FILES:
        if lf and lf.exists():
            try:
                lf.unlink()
                removed.append(str(lf))
            except Exception:
                pass

    # 2️⃣ Remove locks inside known lock directories
    for lock_dir in LOCK_DIRS:
        if lock_dir.exists():
            for p in lock_dir.glob("*"):
                try:
                    p.unlink()
                    removed.append(str(p))
                except Exception:
                    pass

    # 3️⃣ Reset replay state
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

        SWING_REPLAY_STATE.parent.mkdir(parents=True, exist_ok=True)
        with open(SWING_REPLAY_STATE, "w", encoding="utf-8") as f:
            json.dump(clean_state, f, indent=2)

    return {
        "status": "ok",
        "removed": removed,
        "replay_state_reset": True,
    }

# --------------------------------------------------
# Fetch Tickers
# --------------------------------------------------

@router.post("/refresh-universes")
def refresh_universes_tool(user=Depends(admin_required)):
    try:
        from utils.refresh_universes_from_alpaca import refresh_universes
        res = refresh_universes(write_files=True)
        return {
            "status": "ok",
            "total_assets": res.total_assets,
            "base_symbols": len(res.base_symbols),
            "swing_symbols": len(res.swing_symbols),
            "dt_symbols": len(res.dt_symbols),
            "wrote": res.wrote,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------
# Git pull
# --------------------------------------------------

@router.post("/git-pull")
def git_pull(_: None = Depends(admin_required)):
    try:
        result = subprocess.check_output(
            ["git", "pull", "origin", "main"],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.output)

    return {"status": "ok", "output": result}
