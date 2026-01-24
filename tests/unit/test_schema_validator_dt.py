"""Unit tests for schema validation (PR #4)."""

import pytest
from dt_backend.core.schema_validator_dt import (
    ValidationError,
    PolicyDTSchema,
    ExecutionDTSchema,
    PositionDTSchema,
    RollingNodeSchema,
    validate_rolling,
    validate_config_value,
)


def test_policy_dt_validates_required_fields():
    """Missing required field should raise ValidationError."""
    invalid = {"action": "BUY"}  # Missing confidence, trade_gate, etc.
    
    with pytest.raises(ValidationError) as exc_info:
        PolicyDTSchema.validate(invalid, "TEST")
    
    assert "Missing required field" in str(exc_info.value)


def test_policy_dt_validates_field_types():
    """Wrong type should raise ValidationError."""
    invalid = {
        "action": "BUY",
        "intent": "BUY",
        "confidence": "0.75",  # Should be float, not str
        "trade_gate": True,
        "score": 0.5,
    }
    
    with pytest.raises(ValidationError) as exc_info:
        PolicyDTSchema.validate(invalid, "TEST")
    
    assert "expected float" in str(exc_info.value)


def test_policy_dt_validates_action_values():
    """Invalid action should raise ValidationError."""
    invalid = {
        "action": "INVALID_ACTION",  # Should be BUY/SELL/HOLD/STAND_DOWN
        "intent": "BUY",
        "confidence": 0.75,
        "trade_gate": True,
        "score": 0.5,
    }
    
    with pytest.raises(ValidationError) as exc_info:
        PolicyDTSchema.validate(invalid, "TEST")
    
    assert "failed validation" in str(exc_info.value)


def test_policy_dt_validates_confidence_range():
    """Confidence out of range should raise ValidationError."""
    invalid = {
        "action": "BUY",
        "intent": "BUY",
        "confidence": 1.5,  # Should be 0.0-1.0
        "trade_gate": True,
        "score": 0.5,
    }
    
    with pytest.raises(ValidationError):
        PolicyDTSchema.validate(invalid, "TEST")


def test_policy_dt_valid_schema():
    """Valid policy_dt should pass validation."""
    valid = {
        "action": "BUY",
        "intent": "BUY",
        "confidence": 0.75,
        "trade_gate": True,
        "score": 0.5,
        "reason": "Strong momentum signal",
        "ts": "2026-01-24T12:00:00Z",
    }
    
    # Should not raise
    PolicyDTSchema.validate(valid, "TEST")


def test_execution_dt_validates_side():
    """Invalid side should raise ValidationError."""
    invalid = {
        "side": "INVALID",  # Should be BUY/SELL/FLAT
        "size": 0.5,
        "confidence_adj": 0.75,
        "cooldown": False,
    }
    
    with pytest.raises(ValidationError) as exc_info:
        ExecutionDTSchema.validate(invalid, "TEST")
    
    assert "failed validation" in str(exc_info.value)


def test_execution_dt_validates_size_range():
    """Size out of range should raise ValidationError."""
    invalid = {
        "side": "BUY",
        "size": 1.5,  # Should be 0.0-1.0
        "confidence_adj": 0.75,
        "cooldown": False,
    }
    
    with pytest.raises(ValidationError):
        ExecutionDTSchema.validate(invalid, "TEST")


def test_execution_dt_valid_schema():
    """Valid execution_dt should pass validation."""
    valid = {
        "side": "BUY",
        "size": 0.5,
        "confidence_adj": 0.75,
        "cooldown": False,
        "p_hit": 0.65,
    }
    
    # Should not raise
    ExecutionDTSchema.validate(valid, "TEST")


def test_position_dt_validates_qty():
    """Negative qty should raise ValidationError."""
    invalid = {
        "qty": -10.0,  # Should be >= 0
        "avg_price": 150.0,
        "entry_ts": "2026-01-24T12:00:00Z",
    }
    
    with pytest.raises(ValidationError):
        PositionDTSchema.validate(invalid, "TEST")


def test_position_dt_validates_price():
    """Zero or negative price should raise ValidationError."""
    invalid = {
        "qty": 10.0,
        "avg_price": 0.0,  # Should be > 0
        "entry_ts": "2026-01-24T12:00:00Z",
    }
    
    with pytest.raises(ValidationError):
        PositionDTSchema.validate(invalid, "TEST")


def test_position_dt_valid_schema():
    """Valid position_dt should pass validation."""
    valid = {
        "qty": 10.0,
        "avg_price": 150.0,
        "entry_ts": "2026-01-24T12:00:00Z",
        "stop": 145.0,
        "take_profit": 160.0,
        "status": "OPEN",
    }
    
    # Should not raise
    PositionDTSchema.validate(valid, "TEST")


def test_rolling_node_validates_required_sections():
    """Missing required section should raise ValidationError."""
    invalid = {
        "policy_dt": {"action": "BUY"},
    }  # Missing features_dt
    
    with pytest.raises(ValidationError) as exc_info:
        RollingNodeSchema.validate(invalid, "TEST")
    
    assert "Missing required section" in str(exc_info.value)


def test_rolling_node_valid_schema():
    """Valid rolling node should pass validation."""
    valid = {
        "features_dt": {
            "momentum": 0.5,
            "volatility": 0.3,
        },
        "policy_dt": {
            "action": "BUY",
            "intent": "BUY",
            "confidence": 0.75,
            "trade_gate": True,
            "score": 0.5,
        },
    }
    
    # Should not raise
    RollingNodeSchema.validate(valid, "TEST")


def test_validate_rolling_with_multiple_symbols():
    """Validate entire rolling dict with multiple symbols."""
    rolling = {
        "AAPL": {
            "features_dt": {"momentum": 0.5},
            "policy_dt": {
                "action": "BUY",
                "intent": "BUY",
                "confidence": 0.75,
                "trade_gate": True,
                "score": 0.5,
            },
        },
        "TSLA": {
            "features_dt": {"momentum": -0.3},
            "policy_dt": {
                "action": "SELL",
                "intent": "SELL",
                "confidence": 0.60,
                "trade_gate": True,
                "score": -0.4,
            },
        },
        "_GLOBAL_DT": {
            "regime": "normal",
        },
    }
    
    # Should not raise
    validate_rolling(rolling)


def test_validate_rolling_with_invalid_symbol():
    """Invalid symbol data should raise ValidationError."""
    rolling = {
        "AAPL": {
            "features_dt": {"momentum": 0.5},
            "policy_dt": {
                "action": "INVALID",  # Invalid action
                "intent": "BUY",
                "confidence": 0.75,
                "trade_gate": True,
                "score": 0.5,
            },
        },
    }
    
    with pytest.raises(ValidationError):
        validate_rolling(rolling)


def test_validate_config_value_in_range():
    """Valid config value should pass."""
    # Should not raise
    validate_config_value("DT_MAX_POSITIONS", "10")
    validate_config_value("DT_EXEC_MIN_CONF", "0.5")


def test_validate_config_value_out_of_range():
    """Config value out of range should raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        validate_config_value("DT_MAX_POSITIONS", "100")  # Max is 50
    
    assert "out of range" in str(exc_info.value)


def test_validate_config_value_invalid_type():
    """Invalid config value type should raise ValidationError."""
    with pytest.raises(ValidationError):
        validate_config_value("DT_MAX_POSITIONS", "not_a_number")


def test_validate_config_value_unknown_key():
    """Unknown config key should not raise."""
    # Should not raise - no validation rule for unknown keys
    validate_config_value("UNKNOWN_KEY", "any_value")


def test_policy_dt_optional_fields():
    """Optional fields should be validated when present."""
    valid_with_optionals = {
        "action": "BUY",
        "intent": "BUY",
        "confidence": 0.75,
        "trade_gate": True,
        "score": 0.5,
        "p_hit": 0.65,
        "reason": "Strong signal",
        "ts": "2026-01-24T12:00:00Z",
        "_state": {"hysteresis": True},
    }
    
    # Should not raise
    PolicyDTSchema.validate(valid_with_optionals, "TEST")


def test_policy_dt_optional_fields_invalid():
    """Invalid optional fields should raise ValidationError."""
    invalid = {
        "action": "BUY",
        "intent": "BUY",
        "confidence": 0.75,
        "trade_gate": True,
        "score": 0.5,
        "p_hit": 1.5,  # Out of range (should be 0.0-1.0)
    }
    
    with pytest.raises(ValidationError):
        PolicyDTSchema.validate(invalid, "TEST")


def test_rolling_node_with_empty_policy():
    """Empty policy_dt should not be validated."""
    valid = {
        "features_dt": {"momentum": 0.5},
        "policy_dt": None,  # Empty/None policy should be skipped
    }
    
    # Should not raise
    RollingNodeSchema.validate(valid, "TEST")


def test_rolling_node_with_execution_dt():
    """Valid execution_dt should pass validation."""
    valid = {
        "features_dt": {"momentum": 0.5},
        "execution_dt": {
            "side": "BUY",
            "size": 0.5,
            "confidence_adj": 0.75,
            "cooldown": False,
        },
    }
    
    # Should not raise
    RollingNodeSchema.validate(valid, "TEST")
