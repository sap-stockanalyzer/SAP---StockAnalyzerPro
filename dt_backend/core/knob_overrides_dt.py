"""dt_backend.core.knob_overrides_dt — Phase 6

Loads persisted, auto-tuned DT knob overrides (JSON) and applies them to the
process environment so code using os.getenv() sees the updated values.

Controls:
  AION_DISABLE_KNOB_OVERRIDES=1  -> disable entirely
  AION_RESPECT_ENV_KNOBS=1       -> don't overwrite explicitly set env vars
  DT_KNOB_OVERRIDES_PATH         -> override file location
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple


def _truthy(v: str) -> bool:
    v = (v or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path or not path.exists():
            return {}
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def resolve_dt_overrides_path() -> Path:
    p = (os.getenv("DT_KNOB_OVERRIDES_PATH", "") or "").strip()
    if p:
        return Path(p)
    try:
        from dt_backend.core.config_dt import DT_PATHS

        x = DT_PATHS.get("dt_knob_overrides")
        if isinstance(x, Path):
            return x
    except Exception:
        pass
    return Path("ml_data_dt") / "config" / "dt_knob_overrides.json"


def apply_dt_knob_overrides() -> Tuple[bool, Dict[str, Any]]:
    if _truthy(os.getenv("AION_DISABLE_KNOB_OVERRIDES", "")):
        return False, {"status": "disabled"}

    path = resolve_dt_overrides_path()
    data = _read_json(path)
    overrides: Dict[str, Any] = {}
    profiles = data.get("profiles") if isinstance(data.get("profiles"), dict) else None
    if profiles:
        profile = (
            (os.getenv("DT_PROFILE", "") or "").strip()
            or str(data.get("profile") or "").strip()
            or "default"
        )
        cand = profiles.get(profile)
        if isinstance(cand, dict):
            overrides = cand
        else:
            cand2 = profiles.get("default")
            if isinstance(cand2, dict):
                overrides = cand2
    if not overrides:
        overrides = data.get("overrides") if isinstance(data.get("overrides"), dict) else {}
    if not overrides:
        return False, {"status": "empty", "path": str(path)}

    respect_env = _truthy(os.getenv("AION_RESPECT_ENV_KNOBS", ""))
    applied = 0
    for k, v in overrides.items():
        if not isinstance(k, str) or not k:
            continue
        if respect_env and (os.getenv(k) not in (None, "")):
            continue
        os.environ[str(k)] = str(v)
        applied += 1
    meta = {
        "status": "ok",
        "path": str(path),
        "applied": int(applied),
        "profile": (
            (os.getenv("DT_PROFILE", "") or "").strip() or data.get("profile") or "default"
        ),
    }
    return True, meta
