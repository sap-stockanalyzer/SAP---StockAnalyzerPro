"""backend.core.knob_overrides — Phase 6

Loads persisted, auto-tuned knob overrides (JSON) and applies them to the
process environment so existing code that uses os.getenv() picks them up.

Design goals
------------
* Import-safe: tiny file IO, no heavy dependencies.
* Best-effort: never raise.
* Reversible: overrides are persisted in a file and can be cleared.

Override precedence
-------------------
By default, overrides WIN over existing environment variables.
Set AION_RESPECT_ENV_KNOBS=1 to keep explicit env vars in charge.
Set AION_DISABLE_KNOB_OVERRIDES=1 to disable this feature entirely.
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


def resolve_swing_overrides_path() -> Path:
    # Allow explicit override.
    p = (os.getenv("SWING_KNOB_OVERRIDES_PATH", "") or "").strip()
    if p:
        return Path(p)

    # Default (repo root): ml_data/config/swing_knob_overrides.json
    try:
        from backend.core.config import PATHS  # shim → repo root config.py

        x = PATHS.get("swing_knob_overrides")
        if isinstance(x, Path):
            return x
    except Exception:
        pass

    return Path("ml_data") / "config" / "swing_knob_overrides.json"


def apply_swing_knob_overrides() -> Tuple[bool, Dict[str, Any]]:
    """Apply persisted swing knob overrides to os.environ.

    Returns (applied, meta).
    """
    if _truthy(os.getenv("AION_DISABLE_KNOB_OVERRIDES", "")):
        return False, {"status": "disabled"}

    path = resolve_swing_overrides_path()
    data = _read_json(path)
    # Support per-profile overrides: {"profiles": {"risk_on": {...}, "default": {...}}}
    overrides: Dict[str, Any] = {}
    profiles = data.get("profiles") if isinstance(data.get("profiles"), dict) else None
    if profiles:
        profile = (
            (os.getenv("SWING_PROFILE", "") or "").strip()
            or str(data.get("profile") or "").strip()
            or "default"
        )
        cand = profiles.get(profile)
        if isinstance(cand, dict):
            overrides = cand
        else:
            # fall back to default profile if present
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
            (os.getenv("SWING_PROFILE", "") or "").strip()
            or data.get("profile")
            or "default"
        ),
    }
    return True, meta
