"""DT knob hot-reload.

Problem
-------
Linux processes do not automatically re-read environment variables after startup.
Editing dt_knobs.env on disk won't change a running dt_scheduler worker.

Solution
--------
On each cycle (or every N seconds), reload dt_knobs.env (if it changed) and apply
DT_* (and optionally AION_*) keys into os.environ.

This is intentionally lightweight and safe:
  - only reloads when file mtime changes
  - only applies a strict prefix allowlist
  - never deletes env vars that aren't present in the file (avoids surprises)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


# Module-level cache
_LAST_MTIME: float = -1.0
_LAST_CHECK_T: float = 0.0


def _env_float(name: str, default: float) -> float:
    try:
        raw = (os.getenv(name, "") or "").strip()
        return float(raw) if raw else float(default)
    except Exception:
        return float(default)


def _truthy(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


def _default_knobs_path() -> str:
    # Most installs keep dt_knobs.env at repo root.
    return "/home/aion/aion/Aion_Analytics/dt_knobs.env"


def knobs_path() -> str:
    return (os.getenv("DT_KNOBS_PATH") or "").strip() or _default_knobs_path()


def _parse_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return out

    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        # preserve empty values as empty-string
        v = v.strip()
        if not k:
            continue
        # Drop surrounding quotes, but only if they wrap the whole value.
        if len(v) >= 2 and ((v[0] == v[-1]) and v[0] in {"'", '"'}):
            v = v[1:-1]
        out[k] = v
    return out


def _apply(prefixes: Iterable[str], kv: Dict[str, str]) -> Tuple[int, int]:
    """Apply filtered key/values into os.environ.

    Returns: (applied_count, skipped_count)
    """
    pfx = tuple(prefixes)
    applied = 0
    skipped = 0
    for k, v in kv.items():
        if not k.startswith(pfx):
            skipped += 1
            continue
        # Apply
        os.environ[k] = v
        applied += 1
    return applied, skipped


def maybe_reload_dt_knobs(*, force: bool = False) -> Dict[str, object]:
    """Reload dt_knobs.env into os.environ if it changed.

    Knobs:
      - DT_KNOBS_HOT_RELOAD (default: 1)  -> master toggle
      - DT_KNOBS_RELOAD_MIN_SEC (default: 1.0) -> throttle checks
      - DT_KNOBS_ALLOW_AION_PREFIX (default: 0) -> also apply AION_* keys
      - DT_KNOBS_PATH -> path to dt_knobs.env
    """
    global _LAST_MTIME, _LAST_CHECK_T

    if not force and not _truthy("DT_KNOBS_HOT_RELOAD", True):
        return {"status": "disabled"}

    # Throttle stat() calls
    import time

    now = time.time()
    min_sec = max(0.1, _env_float("DT_KNOBS_RELOAD_MIN_SEC", 1.0))
    if not force and (now - _LAST_CHECK_T) < min_sec:
        return {"status": "throttled"}
    _LAST_CHECK_T = now

    p = Path(knobs_path())
    if not p.exists():
        return {"status": "missing", "path": str(p)}

    try:
        mtime = p.stat().st_mtime
    except Exception:
        return {"status": "stat_failed", "path": str(p)}

    if not force and mtime == _LAST_MTIME:
        return {"status": "unchanged", "path": str(p)}

    kv = _parse_env_file(p)

    prefixes = ["DT_"]
    if _truthy("DT_KNOBS_ALLOW_AION_PREFIX", False):
        prefixes.append("AION_")

    applied, skipped = _apply(prefixes, kv)
    _LAST_MTIME = mtime
    return {
        "status": "reloaded",
        "path": str(p),
        "mtime": mtime,
        "applied": applied,
        "skipped": skipped,
    }
