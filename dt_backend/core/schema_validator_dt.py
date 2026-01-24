"""Schema validation for rolling data structures (PR #4).

Validates all data structures in the rolling dict to catch errors early
and provide clear error messages. Uses constants from constants_dt.py
for valid ranges.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from dt_backend.core.constants_dt import VALID_RANGES


class ValidationError(Exception):
    """Raised when schema validation fails."""
    pass


class ActionType(Enum):
    """Valid policy actions."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STAND_DOWN = "STAND_DOWN"
    FLAT = "FLAT"


class PolicyDTSchema:
    """Validates policy_dt node schema."""
    
    REQUIRED_FIELDS = {
        "action": (str, lambda x: x in ["BUY", "SELL", "HOLD", "STAND_DOWN"]),
        "intent": (str, lambda x: x in ["BUY", "SELL", "HOLD", "STAND_DOWN"]),
        "confidence": (float, lambda x: 0.0 <= x <= 1.0),
        "trade_gate": (bool, lambda x: isinstance(x, bool)),
        "score": (float, lambda x: -1.0 <= x <= 1.0),
    }
    
    OPTIONAL_FIELDS = {
        "p_hit": (float, lambda x: 0.0 <= x <= 1.0),
        "reason": (str, lambda x: len(x) <= 500),
        "ts": (str, lambda x: "T" in x),  # ISO format check
        "_state": (dict, lambda x: isinstance(x, dict)),
    }
    
    @classmethod
    def validate(cls, policy_dt: Dict[str, Any], symbol: str = "UNKNOWN") -> None:
        """
        Validate policy_dt matches schema.
        
        Raises: ValidationError with clear message if invalid
        """
        if not isinstance(policy_dt, dict):
            raise ValidationError(
                f"[{symbol}] policy_dt must be dict, got {type(policy_dt).__name__}"
            )
        
        # Check required fields
        for field, (expected_type, validator) in cls.REQUIRED_FIELDS.items():
            if field not in policy_dt:
                raise ValidationError(
                    f"[{symbol}] Missing required field in policy_dt: {field}"
                )
            
            value = policy_dt[field]
            if not isinstance(value, expected_type):
                raise ValidationError(
                    f"[{symbol}] policy_dt.{field}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
            
            if not validator(value):
                raise ValidationError(
                    f"[{symbol}] policy_dt.{field} failed validation: {value}"
                )
        
        # Check optional fields
        for field, (expected_type, validator) in cls.OPTIONAL_FIELDS.items():
            if field in policy_dt and policy_dt[field] is not None:
                value = policy_dt[field]
                if not isinstance(value, expected_type):
                    raise ValidationError(
                        f"[{symbol}] policy_dt.{field}: expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
                if not validator(value):
                    raise ValidationError(
                        f"[{symbol}] policy_dt.{field} failed validation: {value}"
                    )


class ExecutionDTSchema:
    """Validates execution_dt output schema."""
    
    REQUIRED_FIELDS = {
        "side": (str, lambda x: x in ["BUY", "SELL", "FLAT"]),
        "size": (float, lambda x: 0.0 <= x <= 1.0),
        "confidence_adj": (float, lambda x: 0.0 <= x <= 1.0),
        "cooldown": (bool, lambda x: isinstance(x, bool)),
    }
    
    OPTIONAL_FIELDS = {
        "p_hit": (float, lambda x: 0.0 <= x <= 1.0),
        "expected_r": (float, lambda x: x >= 0.1),
        "valid_until": (str, lambda x: "T" in x),
    }
    
    @classmethod
    def validate(cls, execution_dt: Dict[str, Any], symbol: str = "UNKNOWN") -> None:
        """Validate execution_dt matches schema."""
        if not isinstance(execution_dt, dict):
            raise ValidationError(
                f"[{symbol}] execution_dt must be dict, got {type(execution_dt).__name__}"
            )
        
        for field, (expected_type, validator) in cls.REQUIRED_FIELDS.items():
            if field not in execution_dt:
                raise ValidationError(f"[{symbol}] Missing required field: {field}")
            
            value = execution_dt[field]
            if not isinstance(value, expected_type):
                raise ValidationError(
                    f"[{symbol}] execution_dt.{field}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
            
            if not validator(value):
                raise ValidationError(f"[{symbol}] execution_dt.{field} failed validation")


class PositionDTSchema:
    """Validates position_dt node schema."""
    
    REQUIRED_FIELDS = {
        "qty": (float, lambda x: x >= 0),
        "avg_price": (float, lambda x: x > 0),
        "entry_ts": (str, lambda x: "T" in x),
    }
    
    OPTIONAL_FIELDS = {
        "stop": (float, lambda x: x > 0),
        "take_profit": (float, lambda x: x > 0),
        "status": (str, lambda x: x in ["OPEN", "CLOSING", "CLOSED"]),
    }
    
    @classmethod
    def validate(cls, position_dt: Dict[str, Any], symbol: str = "UNKNOWN") -> None:
        """Validate position_dt matches schema."""
        if not isinstance(position_dt, dict):
            raise ValidationError(f"[{symbol}] position_dt must be dict")
        
        for field, (expected_type, validator) in cls.REQUIRED_FIELDS.items():
            if field in position_dt:
                value = position_dt[field]
                if not isinstance(value, expected_type):
                    raise ValidationError(
                        f"[{symbol}] position_dt.{field}: expected {expected_type.__name__}"
                    )
                if not validator(value):
                    raise ValidationError(f"[{symbol}] position_dt.{field} validation failed")


class RollingNodeSchema:
    """Validates entire rolling[symbol] node."""
    
    REQUIRED_SECTIONS = ["features_dt"]
    OPTIONAL_SECTIONS = ["policy_dt", "execution_dt", "position_dt", "context_dt"]
    
    @classmethod
    def validate(cls, node: Dict[str, Any], symbol: str) -> None:
        """Validate rolling node for symbol."""
        if not isinstance(node, dict):
            raise ValidationError(f"[{symbol}] Node must be dict")
        
        # Validate required sections
        for section in cls.REQUIRED_SECTIONS:
            if section not in node:
                raise ValidationError(f"[{symbol}] Missing required section: {section}")
        
        # Validate optional sections if present
        if "policy_dt" in node and node["policy_dt"]:
            PolicyDTSchema.validate(node["policy_dt"], symbol)
        
        if "execution_dt" in node and node["execution_dt"]:
            ExecutionDTSchema.validate(node["execution_dt"], symbol)
        
        if "position_dt" in node and node["position_dt"]:
            PositionDTSchema.validate(node["position_dt"], symbol)


def validate_rolling(rolling: Dict[str, Any]) -> None:
    """Validate entire rolling dict."""
    if not isinstance(rolling, dict):
        raise ValidationError(f"rolling must be dict, got {type(rolling).__name__}")
    
    for symbol, node in rolling.items():
        if isinstance(symbol, str) and not symbol.startswith("_"):
            RollingNodeSchema.validate(node, symbol)


def validate_config_value(key: str, value: Any) -> None:
    """Validate a single configuration value against VALID_RANGES."""
    if key not in VALID_RANGES:
        return  # No validation rule for this key
    
    min_val, max_val = VALID_RANGES[key]
    try:
        float_val = float(value)
        if not (min_val <= float_val <= max_val):
            raise ValidationError(
                f"{key}={float_val} out of range [{min_val}, {max_val}]"
            )
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{key}={value} invalid: {e}")
