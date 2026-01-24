"""Configuration constants and validation ranges for dt_backend (PR #3).

This module defines valid ranges for all configuration parameters,
enabling centralized validation and preventing misconfigurations.
"""

from typing import Dict, Tuple

# Valid ranges for configuration parameters (min, max)
VALID_RANGES: Dict[str, Tuple[float, float]] = {
    # Position and order limits
    "DT_MAX_POSITIONS": (1, 50),
    "DT_MAX_ORDERS_PER_CYCLE": (1, 100),
    "DT_UNIVERSE_SIZE": (10, 5000),
    
    # Confidence and probability thresholds
    "DT_EXEC_MIN_CONF": (0.0, 1.0),
    "DT_MIN_CONFIDENCE": (0.0, 1.0),
    "DT_MIN_PHIT": (0.0, 1.0),
    
    # Loss limits
    "DT_DAILY_LOSS_LIMIT_USD": (0, 10000),
    "DT_MAX_WEEKLY_DRAWDOWN_PCT": (0, 50),
    "DT_MAX_MONTHLY_DRAWDOWN_PCT": (0, 100),
    
    # Risk thresholds
    "DT_VIX_SPIKE_THRESHOLD": (10, 100),
    "DT_MAX_EXPOSURE_FRAC": (0.0, 1.0),
    "DT_MAX_POSITION_SIZE_PCT": (0.0, 1.0),
    
    # Trading parameters
    "DT_DEFAULT_QTY": (0.1, 1000),
    "DT_MIN_FLIP_MINUTES": (0, 1440),  # Up to 24 hours
    "DT_EOD_FLATTEN_MINUTES": (0, 120),
    
    # Stop loss and take profit
    "DT_FALLBACK_STOP_ATR": (0.5, 5.0),
    "DT_FALLBACK_TP_ATR": (0.5, 10.0),
    "DT_TRAIL_ATR_MULT": (0.5, 5.0),
    
    # Feature engineering
    "DT_LOOKBACK_BARS": (10, 1000),
    "DT_ATR_PERIOD": (5, 100),
    "DT_RSI_PERIOD": (5, 100),
    
    # Model parameters
    "DT_MODEL_RETRAIN_DAYS": (1, 365),
    "DT_MIN_TRAINING_SAMPLES": (100, 100000),
    
    # Regime detection
    "DT_REGIME_WINDOW": (10, 200),
    "DT_VOLATILITY_WINDOW": (10, 100),
}

# Action types for policy decisions
VALID_ACTIONS = ["BUY", "SELL", "HOLD", "STAND_DOWN"]

# Execution sides
VALID_SIDES = ["BUY", "SELL", "FLAT"]

# Position statuses
VALID_STATUSES = ["OPEN", "CLOSING", "CLOSED"]

# Valid timeframes
VALID_TIMEFRAMES = ["1Min", "5Min", "15Min", "30Min", "1H", "4H", "1D"]
