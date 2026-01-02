"""
Unified logger for backend + dt_backend with UTF-8 safe console output.

Features:
- Always prints safely on Windows (no UnicodeEncodeError)
- UTF-8 logfile output
- Timestamped logs
- Auto-rotate daily log files
- Thread-safe + multiprocess-safe friendly
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _logfile() -> Path:
    """Per-day rotating logfile."""
    day = datetime.utcnow().strftime("%Y-%m-%d")
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
    """Normal info log."""
    msg = _fmt("INFO", message)
    _safe_print_utf8(msg)
    _write_file(msg)


def warn(message: str) -> None:
    msg = _fmt("WARN", message)
    _safe_print_utf8(msg)
    _write_file(msg)


def error(message: str, exc: Optional[Exception] = None) -> None:
    if exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        message = f"{message}\n{tb}"

    msg = _fmt("ERROR", message)
    _safe_print_utf8(msg, stderr=True)
    _write_file(msg)
