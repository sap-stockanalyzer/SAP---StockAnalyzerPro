"""Unit tests for emergency stop mechanism."""

import pytest
from pathlib import Path


def test_emergency_stop_trigger_and_check(mock_env):
    """Test triggering emergency stop and checking status."""
    from dt_backend.risk.emergency_stop_dt import trigger_emergency_stop, check_emergency_stop
    
    # Initially not stopped
    is_stopped, reason = check_emergency_stop()
    assert is_stopped is False
    assert reason == ""
    
    # Trigger stop
    trigger_emergency_stop("test_reason")
    
    # Now should be stopped
    is_stopped, reason = check_emergency_stop()
    assert is_stopped is True
    assert reason == "test_reason"


def test_emergency_stop_clear(mock_env):
    """Test clearing emergency stop."""
    from dt_backend.risk.emergency_stop_dt import (
        trigger_emergency_stop, 
        check_emergency_stop, 
        clear_emergency_stop
    )
    
    # Trigger and verify
    trigger_emergency_stop("test")
    is_stopped, _ = check_emergency_stop()
    assert is_stopped is True
    
    # Clear and verify
    clear_emergency_stop()
    is_stopped, reason = check_emergency_stop()
    assert is_stopped is False
    assert reason == ""


def test_emergency_stop_file_location(mock_env):
    """Test that emergency stop uses correct file location."""
    from dt_backend.risk.emergency_stop_dt import trigger_emergency_stop, _get_stop_file
    
    trigger_emergency_stop("test")
    
    stop_file = _get_stop_file()
    assert stop_file.exists()
    assert str(stop_file) == "/tmp/test_emergency_stop"


def test_get_stop_status(mock_env):
    """Test getting detailed stop status."""
    from dt_backend.risk.emergency_stop_dt import trigger_emergency_stop, get_stop_status
    
    # Not stopped initially
    status = get_stop_status()
    assert status["is_stopped"] is False
    assert status["file_exists"] is False
    
    # Stopped
    trigger_emergency_stop("market_crash")
    status = get_stop_status()
    assert status["is_stopped"] is True
    assert status["reason"] == "market_crash"
    assert status["file_exists"] is True
