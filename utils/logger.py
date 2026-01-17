"""
Unified logger for AION Analytics (swing + DT) with UTF-8 safe console output.

Features:
- Source-aware logging (swing/dt/backend)
- Dependency injection for DT-specific features (DT brain, cycle tracking)
- Always prints safely on Windows (no UnicodeEncodeError)
- UTF-8 logfile output
- Timestamped logs
- Auto-rotate daily log files
- Thread-safe + multiprocess-safe friendly
- Consistent format: [component] [source] [level] message

Architecture:
- Logger class with DI support for specialized features
- Module-level functions for backward compatibility
- Single implementation replaces backend.core.logger + dt_backend.core.logger_dt
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, Dict

# Determine log base directory
try:
    from backend.config import PATHS as BACKEND_PATHS
    LOG_BASE = BACKEND_PATHS.get("logs") or Path("logs")
except Exception:
    LOG_BASE = Path("logs")

# Ensure folder exists
Path(LOG_BASE).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# UTF-8 Safe Console Writer
# ---------------------------------------------------------

def _safe_print_utf8(msg: str, stderr: bool = False) -> None:
    """
    Write UTF-8 bytes directly to stdout/stderr (Windows-safe).
    Falls back to stripping emojis if terminal still can't render them.
    """
    stream = sys.stderr if stderr else sys.stdout

    try:
        # Write raw UTF-8 bytes (bypasses cp1252 encoder)
        stream.buffer.write((msg + "\n").encode("utf-8"))
        stream.flush()
    except Exception:
        # Absolute fallback: strip non-ASCII so program never crashes
        safe = msg.encode("ascii", "ignore").decode()
        try:
            print(safe, file=stream, flush=True)
        except Exception:
            pass


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _timestamp() -> str:
    """UTC timestamp for logging."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _logfile() -> Path:
    """Per-day rotating logfile."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return Path(LOG_BASE) / f"{day}.log"


def _write_file(msg: str) -> None:
    try:
        f = _logfile()
        f.parent.mkdir(parents=True, exist_ok=True)
        with f.open("a", encoding="utf-8") as fp:
            fp.write(msg + "\n")
    except Exception:
        pass


def _fmt(level: str, text: str) -> str:
    return f"[{_timestamp()}] [{level}] {text}"


# ---------------------------------------------------------
# Public Logging Functions
# ---------------------------------------------------------

def log(message: str) -> None:
    """Normal info log (delegates to default logger)."""
    _default_logger.info(message)


def warn(message: str) -> None:
    """Warning log (delegates to default logger)."""
    _default_logger.warn(message)


def error(message: str, exc: Optional[Exception] = None) -> None:
    """Error log with optional exception (delegates to default logger)."""
    _default_logger.error(message, exc=exc)


# ---------------------------------------------------------
# Logger Class with Dependency Injection
# ---------------------------------------------------------

class Logger:
    """
    Unified logger for AION Analytics (swing + DT).
    
    Supports dependency injection for specialized features:
    - DT brain instance for knob logging
    - Source tracking (swing/dt/backend)
    - Component naming
    
    Format: [component] [source] [level] message
    """
    
    def __init__(
        self,
        name: str = "aion",
        source: str = "backend",  # "swing" | "dt" | "backend"
        dt_brain: Optional[Any] = None,  # Optional: DT brain instance for knob logging
        log_dir: Optional[Path] = None,  # Optional: override log directory
    ):
        """
        Initialize logger with optional DI features.
        
        Args:
            name: Component name (e.g., "nightly_job", "swing_bot", "dt_executor")
            source: System source ("swing", "dt", or "backend")
            dt_brain: Optional DT brain instance for brain-specific logging
            log_dir: Optional override for log directory
        """
        self.name = name
        self.source = source
        self.dt_brain = dt_brain
        self.log_dir = log_dir or LOG_BASE
        self._ensure_log_dir()
    
    def _ensure_log_dir(self) -> None:
        """Ensure log directory exists."""
        try:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    
    def _get_logfile(self) -> Path:
        """Get current logfile path (daily rotation)."""
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # DT backend gets its own subdirectory
        if self.source == "dt":
            subdir = Path(self.log_dir) / "dt_backend"
            subdir.mkdir(parents=True, exist_ok=True)
            return subdir / f"dt_backend_{day}.log"
        return Path(self.log_dir) / f"{day}.log"
    
    def _format_msg(self, level: str, message: str, **context) -> str:
        """
        Format log message with component, source, and level.
        
        Format: [timestamp] [component] [source] [level] message
        """
        ts = _timestamp()
        pid = os.getpid()
        
        # Build context string if provided
        ctx_str = ""
        if context:
            ctx_parts = [f"{k}={v}" for k, v in context.items()]
            ctx_str = f" | {', '.join(ctx_parts)}"
        
        return f"[{ts}] [{self.name}] [{self.source}] [{level}] [pid={pid}] {message}{ctx_str}"
    
    def _write_log(self, msg: str, stderr: bool = False) -> None:
        """Write log to console and file."""
        _safe_print_utf8(msg, stderr=stderr)
        try:
            logfile = self._get_logfile()
            with logfile.open("a", encoding="utf-8") as fp:
                fp.write(msg + "\n")
        except Exception:
            pass
    
    def info(self, message: str, **context) -> None:
        """Log info level message."""
        msg = self._format_msg("INFO", message, **context)
        self._write_log(msg)
    
    def log(self, message: str, **context) -> None:
        """Alias for info (backward compat with dt_backend.core.logger_dt)."""
        self.info(message, **context)
    
    def warn(self, message: str, **context) -> None:
        """Log warning level message."""
        msg = self._format_msg("WARN", message, **context)
        self._write_log(msg)
    
    def warning(self, message: str, **context) -> None:
        """Alias for warn."""
        self.warn(message, **context)
    
    def error(self, message: str, exc: Optional[BaseException] = None, **context) -> None:
        """Log error level message with optional exception traceback."""
        if exc is not None:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            message = f"{message}\n{tb}".rstrip()
        
        msg = self._format_msg("ERROR", message, **context)
        self._write_log(msg, stderr=True)
    
    def dt_brain_update(self, knob: str, old_val: float, new_val: float, reason: str) -> None:
        """
        DT-specific: log brain knob adjustment.
        Only works if dt_brain is provided during initialization.
        """
        if self.dt_brain is None:
            self.warn(f"dt_brain_update called but no dt_brain configured: {knob}")
            return
        
        msg = f"🧠 DT Brain knob adjusted: {knob} {old_val} → {new_val} | reason: {reason}"
        self.info(msg, knob=knob, old_val=old_val, new_val=new_val)


# ---------------------------------------------------------
# Module-level default logger and convenience functions
# ---------------------------------------------------------

_default_logger = Logger(name="aion", source="backend")


def set_default_logger(logger: Logger) -> None:
    """
    Inject custom logger as default for module-level functions.
    
    Use this to configure a DT-aware logger or source-specific logger:
    
        dt_logger = Logger("dt_executor", source="dt", dt_brain=brain)
        set_default_logger(dt_logger)
    """
    global _default_logger
    _default_logger = logger


def get_default_logger() -> Logger:
    """Get the current default logger instance."""
    return _default_logger
