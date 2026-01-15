"""Unit tests for configuration validation."""

import pytest


def test_validate_knobs_valid_config(mock_env, monkeypatch):
    """Test validation with valid configuration."""
    from dt_backend.core.knob_validator_dt import validate_knobs
    
    # Set all valid values
    monkeypatch.setenv("DT_MAX_POSITIONS", "5")
    monkeypatch.setenv("DT_EXEC_MIN_CONF", "0.5")
    
    errors = validate_knobs()
    
    # Should have warnings about missing API keys but no hard errors
    non_warning_errors = [e for e in errors if not e.startswith("WARNING:")]
    assert len(non_warning_errors) == 0


def test_validate_knobs_invalid_ranges(mock_env, monkeypatch):
    """Test validation with invalid ranges."""
    from dt_backend.core.knob_validator_dt import validate_knobs
    
    # Set invalid values
    monkeypatch.setenv("DT_MAX_POSITIONS", "100")  # Out of range
    monkeypatch.setenv("DT_EXEC_MIN_CONF", "1.5")  # Out of range
    
    errors = validate_knobs()
    
    # Should have errors
    assert any("DT_MAX_POSITIONS" in e for e in errors)
    assert any("DT_EXEC_MIN_CONF" in e for e in errors)


def test_get_validation_report(mock_env):
    """Test getting validation report."""
    from dt_backend.core.knob_validator_dt import get_validation_report
    
    report = get_validation_report()
    
    assert "valid" in report
    assert "error_count" in report
    assert "warning_count" in report
    assert "errors" in report
    assert "summary" in report
