"""Emergency stop mechanism for dt_backend (Phase 4).

Provides a file-based emergency stop that can immediately halt all trading.
The emergency stop file acts as a kill switch that can be triggered manually
or programmatically.

Usage:
    # Create emergency stop
    trigger_emergency_stop("market_event")
    
    # Check if stopped
    is_stopped, reason = check_emergency_stop()
    
    # Resume trading
    clear_emergency_stop()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple


def _get_stop_file() -> Path:
    """Get the path to the emergency stop file."""
    stop_file_path = os.getenv("DT_EMERGENCY_STOP_FILE", "/tmp/dt_emergency_stop")
    return Path(stop_file_path)


def check_emergency_stop() -> Tuple[bool, str]:
    """Check if emergency stop file exists.
    
    Returns:
        Tuple[bool, str]: (is_stopped, reason)
            is_stopped: True if emergency stop is active
            reason: Reason for the stop (from file content) or empty string
    """
    try:
        stop_file = _get_stop_file()
        
        if stop_file.exists():
            try:
                reason = stop_file.read_text(encoding="utf-8").strip()
                return True, reason or "emergency_stop_file_exists"
            except Exception:
                return True, "emergency_stop_file_exists"
        
        return False, ""
    except Exception:
        # If we can't check, assume not stopped (fail-safe for normal operation)
        return False, ""


def trigger_emergency_stop(reason: str = "manual"):
    """Create emergency stop file to halt trading.
    
    Args:
        reason: Reason for emergency stop (will be written to file)
    
    Raises:
        IOError: If unable to create stop file
    """
    try:
        stop_file = _get_stop_file()
        
        # Ensure parent directory exists
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write reason to file
        stop_file.write_text(reason, encoding="utf-8")
        
        # Make file readable by all (in case different user needs to clear it)
        try:
            stop_file.chmod(0o666)
        except Exception:
            pass  # Not critical if chmod fails
            
    except Exception as e:
        raise IOError(f"Failed to create emergency stop file: {e}")


def clear_emergency_stop():
    """Remove emergency stop file to resume trading.
    
    Raises:
        FileNotFoundError: If stop file doesn't exist
        IOError: If unable to remove stop file
    """
    try:
        stop_file = _get_stop_file()
        
        if not stop_file.exists():
            raise FileNotFoundError("Emergency stop file does not exist")
        
        stop_file.unlink()
        
    except FileNotFoundError:
        raise
    except Exception as e:
        raise IOError(f"Failed to remove emergency stop file: {e}")


def get_stop_status() -> dict:
    """Get detailed emergency stop status.
    
    Returns:
        dict: Status information including:
            - is_stopped: bool
            - reason: str
            - file_path: str
            - file_exists: bool
    """
    stop_file = _get_stop_file()
    is_stopped, reason = check_emergency_stop()
    
    return {
        "is_stopped": is_stopped,
        "reason": reason,
        "file_path": str(stop_file),
        "file_exists": stop_file.exists(),
    }


if __name__ == "__main__":
    # Test functionality
    print("Emergency Stop Status:", get_stop_status())
