"""dt_backend/core/locks_dt.py — v1.0 (Phase 0)

Lock helpers for dt_backend.

Why this exists
---------------
Multiple dt_backend modules need lightweight file locks to prevent:
  • two dt_scheduler processes running at once
  • overlapping intraday cycles
  • "refetch storms" when multiple processes try to pull live bars

We intentionally keep this dependency-free and reuse the PID lock
implementation from dt_truth_store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from dt_backend.core import DT_PATHS
from dt_backend.services.dt_truth_store import LockHandle, acquire_lock


PathLike = Union[str, Path]


def _as_path(p: Optional[PathLike]) -> Optional[Path]:
    if p is None:
        return None
    try:
        return p if isinstance(p, Path) else Path(str(p))
    except Exception:
        return None


def _fallback_lock(name: str) -> Path:
    da = DT_PATHS.get("da_brains")
    if isinstance(da, Path):
        base = da / "intraday"
    else:
        base = Path("da_brains") / "intraday"
    base.mkdir(parents=True, exist_ok=True)
    return base / name


def acquire_lock_file(lock_path: Optional[PathLike], *, timeout_s: float = 2.0) -> Optional[LockHandle]:
    """Acquire a PID lock file.

    Returns:
        LockHandle if acquired, otherwise None.
    """
    p = _as_path(lock_path)
    if p is None:
        return None
    h = acquire_lock(p, timeout_s=float(timeout_s))
    return h if getattr(h, "acquired", False) else None


def release_lock_file(h: Optional[LockHandle]) -> None:
    try:
        if h is not None:
            h.release()
    except Exception:
        pass


def acquire_scheduler_lock(*, timeout_s: float = 2.0) -> Optional[LockHandle]:
    """Lock used by dt_scheduler to ensure single scheduler instance."""
    p = DT_PATHS.get("dt_scheduler_lock_file")
    if not isinstance(p, Path):
        p = _fallback_lock(".dt_scheduler.lock")
    return acquire_lock_file(p, timeout_s=float(timeout_s))
