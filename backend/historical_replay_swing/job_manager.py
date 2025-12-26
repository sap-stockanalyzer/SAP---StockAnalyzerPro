"""backend.historical_replay_swing.job_manager

Swing backend historical replay manager.

This is the *plumbing* layer:
  - persistent replay_state.json
  - resume/restart/version rules
  - one-day-at-a-time checkpointing
  - status metrics (percent/current day/elapsed/ETA)

Important: a *true* historical replay requires the data pipeline to be
"as-of" date aware (bars/fundamentals/macro/dataset builder). This
manager is designed so we can plug that in incrementally.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, Optional, List

from backend.core.config import PATHS, TIMEZONE
from utils.logger import log


REPLAY_VERSION = "swing-replay-v0.1"


def _now() -> datetime:
    return datetime.now(TIMEZONE)


def _ensure_dirs() -> None:
    Path(PATHS["swing_replay_root"]).mkdir(parents=True, exist_ok=True)


def _state_path() -> Path:
    return Path(PATHS["swing_replay_state"])


def _lock_path() -> Path:
    return Path(PATHS["swing_replay_lock"])


def _maintenance_flag_path() -> Path:
    return Path(PATHS["maintenance_flag"])


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, data: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _parse_iso_dt(x: Any) -> Optional[datetime]:
    if not x:
        return None
    try:
        return datetime.fromisoformat(str(x))
    except Exception:
        return None


def _daterange(start: date, end: date) -> List[date]:
    days: List[date] = []
    d = start
    while d <= end:
        days.append(d)
        d = d + timedelta(days=1)
    return days

def reset_replay_state() -> dict:
    """
    Hard reset the swing replay state so a new replay can start clean.
    This deletes the replay_state.json (and any resumable marker) but does NOT
    delete output artifacts (logs/metrics) unless you explicitly choose to.
    """
    from backend.historical_replay_swing.replay_state import REPLAY_STATE_PATH

    try:
        if REPLAY_STATE_PATH.exists():
            REPLAY_STATE_PATH.unlink()
        return {"status": "ok", "reset": True, "path": str(REPLAY_STATE_PATH)}
    except Exception as e:
        return {"status": "error", "reset": False, "error": str(e), "path": str(REPLAY_STATE_PATH)}

def _acquire_lock() -> bool:
    """
    Acquire replay lock (JSON context file).

    Returns False if a lock already exists.
    """
    lp = _lock_path()
    lp.parent.mkdir(parents=True, exist_ok=True)

    if lp.exists():
        return False

    ctx = {
        "kind": "swing_replay_lock",
        "mode": "replay",
        "version": str(REPLAY_VERSION),
        "started_at": datetime.utcnow().isoformat() + "Z",
        "pid": os.getpid(),
        "host": os.getenv("COMPUTERNAME") or os.getenv("HOSTNAME") or None,
    }
    try:
        lp.write_text(json.dumps(ctx, indent=2), encoding="utf-8")
    except Exception:
        # Fallback: at least write *something* so other processes see a lock.
        lp.write_text(datetime.utcnow().isoformat() + "Z", encoding="utf-8")

    return True


def _release_lock() -> None:
    try:
        lp = _lock_path()
        if lp.exists():
            lp.unlink()
    except Exception:
        pass


def get_state() -> Dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {
            "status": "never_ran",
            "version": None,
            "started_at": None,
            "updated_at": None,
            "start_date": None,
            "end_date": None,
            "current_day": None,
            "days_completed": 0,
            "total_days": 0,
            "percent_complete": 0.0,
            "elapsed_secs": 0.0,
            "eta_secs": None,
            "last_error": None,
            "notes": [],
        }
    try:
        return _read_json(p)
    except Exception:
        # If the state file is corrupted, don't crash the server.
        return {"status": "corrupt", "path": str(p)}


def _write_state(state: Dict[str, Any]) -> None:
    state["updated_at"] = _now().isoformat()
    _write_json(_state_path(), state)


def _set_maintenance(enabled: bool, reason: str = "") -> None:
    p = _maintenance_flag_path()
    if enabled:
        payload = {
            "enabled": True,
            "reason": reason or "historical_replay",
            "timestamp": _now().isoformat(),
        }
        _write_json(p, payload)
    else:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


def request_stop() -> None:
    st = get_state()
    if st.get("status") in ("running", "stopping"):
        st["status"] = "stopping"
        _write_state(st)


def stop_replay() -> Dict[str, Any]:
    """
    Public API expected by routers.

    Requests a graceful stop. This is a safe no-op if no replay is running.
    """
    st = get_state()
    if st.get("status") not in ("running", "stopping"):
        return {"status": "idle", "message": "No replay running", "state": st}

    request_stop()
    return {"status": "stopping", "message": "Stop requested", "state": get_state()}


def get_status() -> Dict[str, Any]:
    """
    Public API expected by routers.
    """
    return get_state()


def reset_state() -> Dict[str, Any]:
    """Hard reset replay state (does NOT delete rolling files)."""
    _ensure_dirs()
    st = {
        "status": "never_ran",
        "version": None,
        "started_at": None,
        "updated_at": _now().isoformat(),
        "start_date": None,
        "end_date": None,
        "current_day": None,
        "days_completed": 0,
        "total_days": 0,
        "percent_complete": 0.0,
        "elapsed_secs": 0.0,
        "eta_secs": None,
        "last_error": None,
        "notes": [],
    }
    _write_state(st)
    return st


@dataclass
class ReplayConfig:
    lookback_days: int = 28
    version: str = REPLAY_VERSION
    # For now we keep this always True, but it can be turned off once you
    # have a clean orchestration that truly pauses other workers.
    set_maintenance_mode: bool = True


_thread: Optional[threading.Thread] = None


def _compute_eta(elapsed_secs: float, done: int, total: int) -> Optional[float]:
    if total <= 0 or done <= 0:
        return None
    rate = elapsed_secs / float(done)
    remaining = max(0, total - done)
    return rate * remaining


def _run_one_day(target_day: date) -> Dict[str, Any]:
    """Run one day's worth of replay.

    Hook point: Replace the body with a true as-of-day pipeline.

    Current behavior (plumbing-safe):
      - Sets env AION_ASOF_DATE for downstream code to optionally read.
      - Executes backend nightly job once.

    This is *not* a true time-travel replay yet, but it gets the control
    plane, state machine, and checkpointing in place.
    """
    os.environ["AION_RUN_MODE"] = "replay"
    os.environ["AION_ASOF_DATE"] = target_day.isoformat()

    # Import here so the replay server can start even if nightly deps are broken.
    from backend.jobs.nightly_job import run_nightly_job  # type: ignore

    res = run_nightly_job(mode="replay", as_of_date=target_day.isoformat(), force=True)
    return {"day": target_day.isoformat(), "nightly": res}


def start_replay(cfg: Optional[ReplayConfig] = None) -> Dict[str, Any]:
    """Start (or resume) a swing replay in a background thread."""
    global _thread

    cfg = cfg or ReplayConfig()
    _ensure_dirs()

    # Prevent concurrent replays.
    st = get_state()
    if st.get("status") == "running":
        return {"status": "already_running", "state": st}

    if not _acquire_lock():
        return {"status": "locked", "reason": "replay_lock_present", "state": st}

    # Version rules
    prior_version = st.get("version")
    prior_status = st.get("status")

    if prior_status == "complete" and prior_version == cfg.version:
        _release_lock()
        return {
            "status": "refused",
            "reason": "already_complete_same_version",
            "version": cfg.version,
            "state": st,
        }

    # Determine date range
    end_day = _now().date()
    start_day = end_day - timedelta(days=max(1, int(cfg.lookback_days)))
    days = _daterange(start_day, end_day)

    # Resume logic
    resume_from: Optional[date] = None
    completed_days: List[str] = []

    if prior_status == "incomplete" and prior_version == cfg.version:
        completed_days = list(st.get("completed_days") or [])
        if completed_days:
            try:
                last_done = date.fromisoformat(str(completed_days[-1]))
                resume_from = last_done + timedelta(days=1)
            except Exception:
                resume_from = None

    # New run or version bump -> start clean
    if resume_from is None:
        completed_days = []
        resume_from = start_day

    # Clamp resume
    if resume_from < start_day:
        resume_from = start_day
    if resume_from > end_day:
        # Nothing to do; mark complete
        new_state = {
            "status": "complete",
            "version": cfg.version,
            "started_at": st.get("started_at") or _now().isoformat(),
            "finished_at": _now().isoformat(),
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "current_day": None,
            "completed_days": completed_days,
            "days_completed": len(completed_days),
            "total_days": len(days),
            "percent_complete": 100.0,
            "elapsed_secs": float(st.get("elapsed_secs") or 0.0),
            "eta_secs": 0.0,
            "last_error": None,
            "notes": st.get("notes") or [],
        }
        _write_state(new_state)
        _release_lock()
        return {"status": "complete", "state": new_state}

    state: Dict[str, Any] = {
        "status": "running",
        "version": cfg.version,
        "started_at": _now().isoformat(),
        "finished_at": None,
        "start_date": start_day.isoformat(),
        "end_date": end_day.isoformat(),
        "current_day": resume_from.isoformat(),
        "completed_days": completed_days,
        "days_completed": int(len(completed_days)),
        "total_days": int(len(days)),
        "percent_complete": float((len(completed_days) / max(1, len(days))) * 100.0),
        "elapsed_secs": 0.0,
        "eta_secs": None,
        "last_error": None,
        "notes": st.get("notes") or [],
    }
    _write_state(state)

    if cfg.set_maintenance_mode:
        _set_maintenance(True, reason="swing_historical_replay")

    def _worker() -> None:
        t_start = time.time()
        try:
            d = date.fromisoformat(str(state["current_day"]))
            while d <= end_day:
                # Stop request?
                cur = get_state()
                if cur.get("status") == "stopping":
                    cur["status"] = "stopped"
                    cur["finished_at"] = _now().isoformat()
                    _write_state(cur)
                    return

                cur["current_day"] = d.isoformat()
                _write_state(cur)

                try:
                    day_res = _run_one_day(d)
                    cur = get_state()  # refresh
                    completed = list(cur.get("completed_days") or [])
                    completed.append(d.isoformat())
                    cur["completed_days"] = completed
                    cur["days_completed"] = int(len(completed))

                    elapsed = time.time() - t_start
                    cur["elapsed_secs"] = float(round(elapsed, 3))
                    cur["percent_complete"] = float((len(completed) / max(1, len(days))) * 100.0)
                    cur["eta_secs"] = _compute_eta(elapsed, len(completed), len(days))
                    cur["last_error"] = None
                    # Optional: store last_day_result shallow
                    cur["last_day"] = {
                        "day": d.isoformat(),
                        "status": (day_res.get("nightly") or {}).get("status"),
                    }
                    _write_state(cur)

                except Exception as e_day:
                    cur = get_state()
                    cur["status"] = "incomplete"
                    cur["finished_at"] = _now().isoformat()
                    cur["last_error"] = f"{type(e_day).__name__}: {e_day}"
                    _write_state(cur)
                    return

                # next day
                d = d + timedelta(days=1)

            # Completed
            cur = get_state()
            cur["status"] = "complete"
            cur["finished_at"] = _now().isoformat()
            cur["current_day"] = None
            cur["percent_complete"] = 100.0
            cur["eta_secs"] = 0.0
            _write_state(cur)

        finally:
            if cfg.set_maintenance_mode:
                _set_maintenance(False)
            _release_lock()

    _thread = threading.Thread(target=_worker, name="swing_replay", daemon=True)
    _thread.start()
    return {"status": "started", "state": get_state()}

def get_replay_status() -> Dict[str, Any]:
    """
    Backward-compatible alias for admin / router API.
    """
    return get_status()

def get_replay_status() -> Dict[str, Any]:
    """
    Backward-compatible alias for older routers/UI.
    """
    return get_status()


def start_replay_legacy(weeks: int = 4, version: str = "v1") -> Dict[str, Any]:
    """
    Backward-compatible shim: start_replay(weeks=?, version=?)

    Converts weeks → lookback_days and passes ReplayConfig to start_replay().
    """
    try:
        w = int(weeks)
    except Exception:
        w = 4
    if w < 1:
        w = 1
    if w > 104:
        w = 104

    cfg = ReplayConfig(lookback_days=w * 7, version=str(version))
    return start_replay(cfg)

__all__ = [
    "start_replay",
    "start_replay_legacy",
    "stop_replay",
    "get_status",
    "get_replay_status",
    "get_state",
    "reset_state",
    "request_stop",
]

