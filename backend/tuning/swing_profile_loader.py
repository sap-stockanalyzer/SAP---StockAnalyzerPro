"""backend.tuning.swing_profile_loader — Phase 5 playbooks (Swing)

Loads regime-specific swing profiles (risk_on/risk_off/bear/high_vol) that
adjust *soft* gates and tier thresholds. Hard rails remain elsewhere.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from config import PATHS  # root config


def _profiles_dir() -> Path:
    p = PATHS.get("swing_profiles_dir")
    return Path(p) if p else Path("ml_data/config/swing_knob_profiles")


def _active_file() -> Path:
    p = PATHS.get("swing_profiles_active_file")
    return Path(p) if p else (_profiles_dir() / "_active_profile.json")


def _safe_load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def pick_swing_profile_name(regime_label: Optional[str]) -> str:
    env = (os.getenv("SWING_PROFILE", "") or "").strip()
    if env:
        return env

    active = _safe_load_json(_active_file()).get("active")
    if isinstance(active, str) and active.strip():
        return active.strip()

    rl = (regime_label or "").lower()

    if "high" in rl and "vol" in rl:
        return "high_vol"
    if "bear" in rl:
        return "bear"
    if "risk_off" in rl or "off" in rl or "panic" in rl:
        return "risk_off"
    return "risk_on"


def load_swing_profile(regime_label: Optional[str]) -> Dict[str, Any]:
    name = pick_swing_profile_name(regime_label)
    path = _profiles_dir() / f"{name}.json"
    prof = _safe_load_json(path)
    prof.setdefault("label", name)
    return prof
