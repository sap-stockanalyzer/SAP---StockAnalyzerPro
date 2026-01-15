"""Configuration validation for dt_backend (Phase 4).

Validates all configuration knobs have sensible values and required
dependencies are present. Helps catch configuration errors at startup
rather than during trading.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List


def _safe_float(val: str, default: float = 0.0) -> float:
    """Convert string to float safely."""
    try:
        return float(val) if val else default
    except (ValueError, TypeError):
        return default


def _safe_int(val: str, default: int = 0) -> int:
    """Convert string to int safely."""
    try:
        return int(float(val)) if val else default
    except (ValueError, TypeError):
        return default


def validate_knobs() -> List[str]:
    """Validate all configuration knobs have sensible values.
    
    Returns:
        List[str]: List of validation errors (empty if all valid)
    """
    errors = []
    
    # Validate numeric ranges
    max_positions = _safe_int(os.getenv("DT_MAX_POSITIONS", "3"), 3)
    if max_positions < 1 or max_positions > 50:
        errors.append(f"DT_MAX_POSITIONS={max_positions} out of range [1, 50]")
    
    exec_min_conf = _safe_float(os.getenv("DT_EXEC_MIN_CONF", "0.25"), 0.25)
    if exec_min_conf < 0.0 or exec_min_conf > 1.0:
        errors.append(f"DT_EXEC_MIN_CONF={exec_min_conf} out of range [0.0, 1.0]")
    
    max_orders = _safe_int(os.getenv("DT_MAX_ORDERS_PER_CYCLE", "3"), 3)
    if max_orders < 1 or max_orders > 100:
        errors.append(f"DT_MAX_ORDERS_PER_CYCLE={max_orders} out of range [1, 100]")
    
    # Validate loss limits
    daily_loss_limit = _safe_float(os.getenv("DT_DAILY_LOSS_LIMIT_USD", "300"), 300)
    if daily_loss_limit < 0 or daily_loss_limit > 10000:
        errors.append(f"DT_DAILY_LOSS_LIMIT_USD={daily_loss_limit} out of range [0, 10000]")
    
    weekly_dd_pct = _safe_float(os.getenv("DT_MAX_WEEKLY_DRAWDOWN_PCT", "8.0"), 8.0)
    if weekly_dd_pct < 0 or weekly_dd_pct > 50:
        errors.append(f"DT_MAX_WEEKLY_DRAWDOWN_PCT={weekly_dd_pct} out of range [0, 50]")
    
    monthly_dd_pct = _safe_float(os.getenv("DT_MAX_MONTHLY_DRAWDOWN_PCT", "15.0"), 15.0)
    if monthly_dd_pct < 0 or monthly_dd_pct > 100:
        errors.append(f"DT_MAX_MONTHLY_DRAWDOWN_PCT={monthly_dd_pct} out of range [0, 100]")
    
    # Validate VIX threshold
    vix_threshold = _safe_float(os.getenv("DT_VIX_SPIKE_THRESHOLD", "35.0"), 35.0)
    if vix_threshold < 10 or vix_threshold > 100:
        errors.append(f"DT_VIX_SPIKE_THRESHOLD={vix_threshold} out of range [10, 100]")
    
    # Validate exposure limits
    max_exposure = _safe_float(os.getenv("DT_MAX_EXPOSURE_FRAC", "0.55"), 0.55)
    if max_exposure < 0.0 or max_exposure > 1.0:
        errors.append(f"DT_MAX_EXPOSURE_FRAC={max_exposure} out of range [0.0, 1.0]")
    
    # Validate paths exist
    da_brains = os.getenv("DA_BRAINS_ROOT")
    if da_brains and not Path(da_brains).exists():
        errors.append(f"DA_BRAINS_ROOT={da_brains} does not exist")
    
    dt_truth_dir = os.getenv("DT_TRUTH_DIR")
    if dt_truth_dir and not Path(dt_truth_dir).exists():
        errors.append(f"DT_TRUTH_DIR={dt_truth_dir} does not exist")
    
    # Validate rolling path
    rolling_path = os.getenv("DT_ROLLING_PATH")
    if rolling_path:
        rolling_file = Path(rolling_path)
        if not rolling_file.parent.exists():
            errors.append(f"DT_ROLLING_PATH parent directory does not exist: {rolling_file.parent}")
    
    # Validate API keys are set (warnings only, not errors)
    if not os.getenv("ALPACA_API_KEY_ID"):
        errors.append("WARNING: ALPACA_API_KEY_ID is not set")
    
    if not os.getenv("ALPACA_API_SECRET_KEY"):
        errors.append("WARNING: ALPACA_API_SECRET_KEY is not set")
    
    # Validate dry run and live trading settings
    dry_run = _safe_int(os.getenv("DT_DRY_RUN", "1"), 1)
    enable_live = _safe_int(os.getenv("DT_ENABLE_LIVE_TRADING", "0"), 0)
    
    if dry_run == 0 and enable_live == 0:
        errors.append("WARNING: DT_DRY_RUN=0 and DT_ENABLE_LIVE_TRADING=0 - no trading will occur")
    
    if dry_run == 0 and enable_live == 1:
        errors.append("WARNING: LIVE TRADING IS ENABLED (DT_DRY_RUN=0, DT_ENABLE_LIVE_TRADING=1)")
    
    # Validate timeframes
    feature_tf = os.getenv("DT_FEATURE_TF", "5Min")
    valid_tfs = ["1Min", "5Min", "15Min", "30Min", "1H"]
    if feature_tf not in valid_tfs:
        errors.append(f"DT_FEATURE_TF={feature_tf} not in valid timeframes: {valid_tfs}")
    
    # Validate universe size
    universe_size = _safe_int(os.getenv("DT_UNIVERSE_SIZE", "150"), 150)
    if universe_size < 10 or universe_size > 5000:
        errors.append(f"DT_UNIVERSE_SIZE={universe_size} out of range [10, 5000]")
    
    return errors


def validate_and_warn():
    """Validate configuration and print warnings for any errors."""
    errors = validate_knobs()
    
    if errors:
        print("\n" + "="*60)
        print("⚠️  CONFIGURATION VALIDATION WARNINGS")
        print("="*60)
        for err in errors:
            print(f"  • {err}")
        print("="*60 + "\n")
    else:
        print("✅ Configuration validation passed")
    
    return len(errors) == 0


def get_validation_report() -> Dict[str, Any]:
    """Get detailed validation report.
    
    Returns:
        dict: Validation report with errors and summary
    """
    errors = validate_knobs()
    
    # Count errors vs warnings
    error_count = sum(1 for e in errors if not e.startswith("WARNING:"))
    warning_count = len(errors) - error_count
    
    return {
        "valid": len(errors) == 0,
        "error_count": error_count,
        "warning_count": warning_count,
        "total_issues": len(errors),
        "errors": errors,
        "summary": (
            "Configuration is valid" if len(errors) == 0
            else f"Found {error_count} errors and {warning_count} warnings"
        ),
    }


if __name__ == "__main__":
    # Run validation when executed directly
    validate_and_warn()
