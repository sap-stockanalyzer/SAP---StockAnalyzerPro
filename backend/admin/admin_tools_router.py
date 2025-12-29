# /backend/admin/admin_tools_router.py

from __future__ import annotations

import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from backend.admin.deps import admin_required
from backend.core.config import PATHS
from config import ROOT
from utils.live_log import tail_lines

from admin_keys import SUPABASE_BUCKET, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL
from backend.services.supabase_sync import sync_to_supabase, supabase_configured

router = APIRouter(prefix="/admin/tools", tags=["admin-tools"])


class SupabaseSyncRequest(BaseModel):
    dry_run: bool = False
    targets: Optional[List[str]] = None


from backend.historical_replay_swing.job_manager import REPLAY_STATE_PATH
SWING_REPLAY_STATE = REPLAY_STATE_PATH

# ------------------------------------------------------------------
# Lock locations (DIRECTORIES)
# ------------------------------------------------------------------

LOCK_DIRS = [
    ROOT / "data" / "locks",
    ROOT / "data" / "replay" / "locks",
    Path(PATHS.get("nightly_lock")).parent if PATHS.get("nightly_lock") else (ROOT / "da_brains"),
    Path(PATHS.get("swing_replay_lock")).parent if PATHS.get("swing_replay_lock") else (ROOT / "data" / "replay" / "locks"),
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
# Supabase manual sync
# --------------------------------------------------

@router.get("/supabase/status")
def supabase_status(_: None = Depends(admin_required)):
    return {
        "configured": bool(supabase_configured()),
        "url_present": bool(SUPABASE_URL),
        "service_key_present": bool(SUPABASE_SERVICE_ROLE_KEY),
        "bucket": (SUPABASE_BUCKET or "aion"),
    }


@router.post("/supabase/sync")
def supabase_sync(req: SupabaseSyncRequest, _: None = Depends(admin_required)):
    res = sync_to_supabase(dry_run=bool(req.dry_run), targets=req.targets)
    # dataclass -> jsonable dict
    return {
        "status": "ok" if not res.errors else "partial",
        "uploaded": res.uploaded,
        "skipped": res.skipped,
        "missing": res.missing,
        "errors": res.errors,
    }


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
