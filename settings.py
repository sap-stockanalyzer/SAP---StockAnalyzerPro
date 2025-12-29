"""
settings.py (ROOT)

Single knob/settings registry shared by backend + dt_backend.

NOTE: Phase A foundation file.
Later phases will expand BOT_KNOBS_* to match the full bots UI contract.
"""

from __future__ import annotations

import os
from typing import Any, Dict
import pytz

TIMEZONE = pytz.timezone(os.getenv("AION_TZ", "America/Denver"))

# Basic defaults (kept conservative; UI can override per-bot)
BOT_KNOBS_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "swing": {
        "enabled": True,
        "aggression": 0.50,
        "max_alloc": 1000.0,          # dollars (UI-facing)
        "max_positions": 10,
        "stop_loss": 3.0,             # percent (UI-facing)
        "take_profit": 6.0,           # percent (UI-facing)
        "min_confidence": 0.55,
        "allow_etfs": True,
        "penny_only": False,
    },
    "intraday": {
        "enabled": True,
        "aggression": 0.50,
        "max_alloc": 500.0,           # dollars (UI-facing)
        "max_positions": 5,
        "stop_loss": 0.8,             # percent
        "take_profit": 1.5,           # percent
        "min_confidence": 0.55,
        "penny_only": False,
        "allow_etfs": True,
        "max_daily_trades": 12,
    },
}

# Optional schema hints for UI validation (min/max/step).
BOT_KNOBS_SCHEMA: Dict[str, Dict[str, Any]] = {
    "aggression": {"min": 0.0, "max": 1.0, "step": 0.05},
    "max_alloc": {"min": 0.0, "max": 1_000_000.0, "step": 50.0},
    "max_positions": {"min": 1, "max": 200, "step": 1},
    "stop_loss": {"min": 0.0, "max": 50.0, "step": 0.1},
    "take_profit": {"min": 0.0, "max": 200.0, "step": 0.1},
    "min_confidence": {"min": 0.0, "max": 1.0, "step": 0.01},
    "max_daily_trades": {"min": 0, "max": 500, "step": 1},
}
