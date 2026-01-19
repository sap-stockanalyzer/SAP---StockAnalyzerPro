"""dt_backend/core/logger_dt.py — Compatibility wrapper for unified logger

UPDATED: This module now forwards to the unified logger (utils.logger).

The unified logger (utils.logger.Logger) provides:
* Source-aware logging (swing/dt/backend)
* Dependency injection for DT-specific features
* Consistent format across all systems
* UTF-8 safe console output
* Daily rotating logfiles

This wrapper maintains backward compatibility:
* Functions (info, warn, error, log) forward to unified logger
* DT logs go to logs/dt_backend/ subdirectory
* Source is set to "dt"

For new code, prefer importing directly from utils.logger:
    from utils.logger import Logger
    logger = Logger("my_component", source="dt")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Import unified logger
from utils.logger import Logger as UnifiedLogger

# Resolve DT log directory for backward compatibility
def _resolve_log_dir() -> Path:
    """Prefer dt_backend's configured log directory; fall back safely."""
    try:
        from dt_backend.core.config_dt import DT_PATHS  # type: ignore
        p = DT_PATHS.get("logs_dt")
        if isinstance(p, Path):
            return p
    except Exception:
        pass
    
    # Fallback: project-root-ish logs/dt_backend
    try:
        here = Path(__file__).resolve()
        root = here.parents[2]
        return root / "logs" / "dt_backend"
    except Exception:
        return Path("logs") / "dt_backend"


# Create default DT logger instance
_dt_logger = UnifiedLogger(
    name="dt_backend",
    source="dt",
    log_dir=_resolve_log_dir()
)


# ---------------------------------------------------------------------------
# Public API (backward compatible)
# ---------------------------------------------------------------------------

def debug(message: str) -> None:
    """Log debug level message."""
    _dt_logger.debug(message)


def info(message: str) -> None:
    """Log info level message."""
    _dt_logger.info(message)


def warn(message: str) -> None:
    """Log warning level message."""
    _dt_logger.warn(message)


def error(message: str, exc: Optional[BaseException] = None) -> None:
    """Log error level message with optional exception traceback."""
    _dt_logger.error(message, exc=exc)


# Convenience alias used across dt_backend (mirrors earlier core.data_pipeline_dt.log)
def log(message: str) -> None:
    """Log info level message (alias for info)."""
    _dt_logger.info(message)

