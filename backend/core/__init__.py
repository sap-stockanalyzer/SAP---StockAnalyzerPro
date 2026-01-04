"""
AION Analytics — Backend Core Modules

NOTE:
This package is imported by API routers and service boot code.
Keep this module IMPORT-SAFE:
- no AI/model training imports
- no filesystem-heavy initialization
- no long-running side effects
"""

from .config import PATHS

# Phase 6: apply persisted knob overrides early (import-safe, best-effort)
try:
    from .knob_overrides import apply_swing_knob_overrides

    apply_swing_knob_overrides()
except Exception:
    # keep import-safe
    pass
from .data_pipeline import (
    log,
    _read_rolling,
    save_rolling,
    safe_float,
)

__all__ = [
    "PATHS",
    "log",
    "_read_rolling",
    "save_rolling",
    "safe_float",
]
