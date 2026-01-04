"""dt_backend.tuning.dt_profile_loader — Phase 5 playbooks (DT)

Loads regime-specific knob profiles (soft gates + strategy weights).
Hard rails remain elsewhere and are NOT modified here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dt_backend.core.config_dt import DT_PATHS  # shim to root config


def _profiles_dir() -> Path:
    p = DT_PATHS.get("dt_profiles_dir")
    return Path(p) if p else Path("ml_data_dt/config/dt_knob_profiles")


def _active_file() -> Path:
    p = DT_PATHS.get("dt_profiles_active_file")
    return Path(p) if p else (_profiles_dir() / "_active_profile.json")


def _safe_load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def pick_dt_profile_name(regime_label: Optional[str]) -> str:
    """Pick a profile name from a regime label and optional overrides."""
    env = (os.getenv("DT_PROFILE", "") or "").strip()
    if env:
        return env

    active = _safe_load_json(_active_file()).get("active")
    if isinstance(active, str) and active.strip():
        return active.strip()

    rl = (regime_label or "").lower()

    if "news" in rl or "earn" in rl:
        return "news_risk"
    if "high" in rl and "vol" in rl:
        return "high_vol"
    if "low" in rl and "vol" in rl:
        return "low_vol"
    if "range" in rl or "chop" in rl:
        return "range"
    if "trend" in rl or "bull" in rl:
        return "trend"

    return "trend"


def load_dt_profile(regime_label: Optional[str]) -> Dict[str, Any]:
    name = pick_dt_profile_name(regime_label)
    path = _profiles_dir() / f"{name}.json"
    prof = _safe_load_json(path)
    prof.setdefault("label", name)
    return prof


def strategy_weight(profile: Dict[str, Any], strategy_name: str) -> float:
    try:
        m = profile.get("strategy_weights") or {}
        w = float(m.get(strategy_name, 1.0))
        return max(0.25, min(2.0, w))
    except Exception:
        return 1.0
