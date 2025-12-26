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
from typing import Any, Dict, Optional, List, Tuple

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


def _atomic_write_text(path: Path, text: str) -> None:
    """
    Atomic write to avoid UI polling reading a half-written JSON file.
    This is the #1 cause of "status looks wrong / progress resets" in pollers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json(p: Path, data: Dict[str, Any]) -> None:
    _atomic_write_text(p, json.dumps(data, indent=2))


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
    """
    from backend.historical_replay_swing.replay_state import REPLAY_STATE_PATH

    try:
        if REPLAY_STATE_PATH.exists():
            REPLAY_STATE_PATH.unlink()
        return {"status": "ok", "reset": True, "path": str(REPLAY_STATE_PATH)}
    except Exception as e:
        return {"status": "error", "reset": False, "error": str(e), "path": str(REPLAY_STATE_PATH)}


def _acquire_lock() -> bool:
    lp = _lock_path()
    lp.parent.mkdir(parents=True, exist_ok=True)

    if lp.exists():
        return False

    ctx = {
        "kind": "swing_replay_lock",
        "mode": "replay",
        "version": REPLAY_VERSION,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "pid": os.getpid(),
        "host": os.getenv("COMPUTERNAME") or os.getenv("HOSTNAME"),
    }

    try:
        _atomic_write_text(lp, json.dumps(ctx, indent=2))
    except Exception:
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
            "finished_at": None,
            "updated_at": None,
            "start_date": None,
            "end_date": None,
            "current_day": None,
            "completed_days": [],
            "days_completed": 0,
            "total_days": 0,
            "percent_complete": 0.0,
            "elapsed_secs": 0.0,
            "eta_secs": None,
            "last_error": None,
            "notes": [],
        }

    try:
        st = _read_json(p)
        return st if isinstance(st, dict) else {"status": "corrupt", "path": str(p)}
    except Exception:
        return {"status": "corrupt", "path": str(p)}


def _write_state(state: Dict[str, Any]) -> None:
    state["updated_at"] = _now().isoformat()
    _write_json(_state_path(), state)


def _set_maintenance(enabled: bool, reason: str = "") -> None:
    p = _maintenance_flag_path()
    if enabled:
        _write_json(p, {
            "enabled": True,
            "reason": reason or "historical_replay",
            "timestamp": _now().isoformat(),
        })
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
    st = get_state()
    if st.get("status") not in ("running", "stopping"):
        return {"status": "idle", "state": st}

    request_stop()
    return {"status": "stopping", "state": get_state()}


def get_status() -> Dict[str, Any]:
    return get_state()


def reset_state() -> Dict[str, Any]:
    _ensure_dirs()
    st = {
        "status": "never_ran",
        "version": None,
        "started_at": None,
        "finished_at": None,
        "updated_at": _now().isoformat(),
        "start_date": None,
        "end_date": None,
        "current_day": None,
        "completed_days": [],
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
    set_maintenance_mode: bool = True


_thread: Optional[threading.Thread] = None


def _compute_eta(elapsed_secs: float, done: int, total: int) -> Optional[float]:
    if total <= 0 or done <= 0:
        return None
    return (elapsed_secs / done) * (total - done)


def _run_one_day(target_day: date) -> Dict[str, Any]:
    os.environ["AION_RUN_MODE"] = "replay"
    os.environ["AION_ASOF_DATE"] = target_day.isoformat()

    from backend.jobs.nightly_job import run_nightly_job
    res = run_nightly_job(mode="replay", as_of_date=target_day.isoformat(), force=True)
    return {"day": target_day.isoformat(), "nightly": res}


def start_replay(cfg: Optional[ReplayConfig] = None) -> Dict[str, Any]:
    global _thread

    cfg = cfg or ReplayConfig()
    _ensure_dirs()

    st = get_state()
    if st.get("status") == "running":
        return {"status": "already_running", "state": st}

    if not _acquire_lock():
        return {"status": "locked", "state": st}

    end_day = _now().date()
    start_day = end_day - timedelta(days=max(1, cfg.lookback_days))
    days = _daterange(start_day, end_day)

    started_at = st.get("started_at") or _now().isoformat()
    prev_elapsed = float(st.get("elapsed_secs") or 0.0)

    state = {
        "status": "running",
        "version": cfg.version,
        "started_at": started_at,
        "finished_at": None,
        "start_date": start_day.isoformat(),
        "end_date": end_day.isoformat(),
        "current_day": start_day.isoformat(),
        "completed_days": [],
        "days_completed": 0,
        "total_days": len(days),
        "percent_complete": 0.0,
        "elapsed_secs": prev_elapsed,
        "eta_secs": None,
        "last_error": None,
        "notes": st.get("notes") or [],
    }
    _write_state(state)

    if cfg.set_maintenance_mode:
        _set_maintenance(True, "swing_historical_replay")

    def _worker() -> None:
        t0 = time.time()
        try:
            for d in days:
                cur = get_state()
                if cur.get("status") == "stopping":
                    cur["status"] = "stopped"
                    cur["finished_at"] = _now().isoformat()
                    cur["elapsed_secs"] = prev_elapsed + (time.time() - t0)
                    _write_state(cur)
                    return

                cur["status"] = "running"
                cur["current_day"] = d.isoformat()
                _write_state(cur)

                _run_one_day(d)

                completed = list(cur.get("completed_days") or [])
                completed.append(d.isoformat())

                elapsed = prev_elapsed + (time.time() - t0)
                cur.update({
                    "completed_days": completed,
                    "days_completed": len(completed),
                    "elapsed_secs": round(elapsed, 3),
                    "percent_complete": (len(completed) / len(days)) * 100.0,
                    "eta_secs": _compute_eta(elapsed, len(completed), len(days)),
                })
                _write_state(cur)

            cur = get_state()
            cur.update({
                "status": "complete",
                "finished_at": _now().isoformat(),
                "current_day": None,
                "percent_complete": 100.0,
                "eta_secs": 0.0,
            })
            _write_state(cur)

        finally:
            if cfg.set_maintenance_mode:
                _set_maintenance(False)
            _release_lock()

    _thread = threading.Thread(target=_worker, daemon=True)
    _thread.start()
    return {"status": "started", "state": get_state()}


def get_replay_status() -> Dict[str, Any]:
    return get_status()


def start_replay_legacy(weeks: int = 4, version: str = "v1") -> Dict[str, Any]:
    w = max(1, min(int(weeks or 4), 104))
    return start_replay(ReplayConfig(lookback_days=w * 7, version=version))


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
