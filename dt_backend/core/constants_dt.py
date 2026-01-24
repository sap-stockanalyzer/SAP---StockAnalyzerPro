# dt_backend/core/constants_dt.py
"""
Central constants file for dt_backend - single source of truth for all thresholds and magic numbers.

This module centralizes configuration values that were previously scattered throughout the codebase.
All trading constants, thresholds, and limits are defined here for easy maintenance and visibility.

Version: 1.0.0
"""

from typing import Dict, Tuple

# ============================================================
# CONFIDENCE THRESHOLDS
# ============================================================

CONFIDENCE_MIN = 0.45                    # Minimum confidence for policy decisions (raised from 0.25 to reduce false signals)
CONFIDENCE_MIN_EXEC = 0.25               # Minimum confidence for execution layer
CONFIDENCE_MIN_PROBE = 0.18              # Micro-entry tier
CONFIDENCE_MAX = 0.99                    # Hard cap after adjustments
CONFIDENCE_EXIT_BUFFER = 0.05            # SELL margin above HOLD

# ============================================================
# POSITION SIZING
# ============================================================

POSITION_MAX_FRACTION = 0.15             # 15% of account max
POSITION_PROBE_FRACTION = 0.25           # 25% of full conviction
POSITION_PRESS_MULT = 1.35               # Scale-up for high P(Hit)
POSITION_DEFAULT_QTY = 1.0               # Base shares per trade
POSITIONS_MAX_OPEN = 3                   # Concurrent limit

# ============================================================
# HYSTERESIS & SIGNAL STABILITY
# ============================================================

EDGE_MIN_TO_FLIP = 0.06                  # Minimum edge to flip direction
EDGE_HOLD_BIAS = 0.03                    # Extra margin to flip from HOLD
CONFIRMATIONS_TO_FLIP = 2                # Consecutive confirmations needed
COOLDOWN_AFTER_BUY_MINUTES = 10          # Flip cooldown after entry
HOLD_MIN_TIME_MINUTES = 10               # Minimum hold before exit allowed

# ============================================================
# P(HIT) CALIBRATION
# ============================================================

PHIT_MIN = 0.52                          # Minimum to size above zero
PHIT_PRESS_MIN = 0.62                    # P(Hit) for PRESS tier
LOSS_EST_PCT = 0.06                      # Expected loss buffer

# ============================================================
# VOLATILITY ADJUSTMENT
# ============================================================

VOL_PENALTY_HIGH = 0.65                  # High vol (>2%) penalty
VOL_PENALTY_MEDIUM = 0.85                # Medium vol (0.7-2%) penalty

# ============================================================
# TREND & REGIME
# ============================================================

TREND_BOOST_STRONG = 1.25                # Strong trend alignment boost
TREND_BOOST_MILD = 1.10                  # Mild trend alignment boost
REGIME_PENALTY_CHOP = 0.90               # Chop regime penalty
REGIME_PENALTY_BEAR_BUY = 0.85           # Bear market BUY penalty
REGIME_PENALTY_BULL_SELL = 0.90          # Bull market SELL penalty

# ============================================================
# SIGNAL THRESHOLDS
# ============================================================

BUY_THRESHOLD = 0.12                     # Buy signal edge threshold
SELL_THRESHOLD = -0.12                   # Sell signal edge threshold
SCORE_MIN_ABS = 12.0                     # Minimum absolute score

# ============================================================
# RISK MANAGEMENT
# ============================================================

DAILY_LOSS_LIMIT = 300.0                 # Daily stop loss (USD)
WEEKLY_DRAWDOWN_MAX_PCT = 8.0            # Weekly drawdown cap
MONTHLY_DRAWDOWN_MAX_PCT = 15.0          # Monthly drawdown cap
VIX_SPIKE_THRESHOLD = 35.0               # VIX panic threshold
POSITIONS_MAX_PER_SECTOR = 2             # Sector concentration limit
EXPOSURE_MAX = 0.55                      # Maximum portfolio exposure

# ============================================================
# EXECUTION & ORDER MANAGEMENT
# ============================================================

ORDERS_MAX_PER_CYCLE = 3                 # Orders per cycle
TRADE_GAP_MIN_MINUTES = 15               # Min time between same-symbol trades
ORDER_TIMEOUT_SEC = 4.0                  # Order execution timeout

# ============================================================
# REGIME EXPOSURE MAPPING
# ============================================================

REGIME_EXPOSURE: Dict[str, float] = {
    "bull": 1.00,                        # Full exposure
    "chop": 0.70,                        # 70% of normal
    "bear": 0.45,                        # 45% of normal
    "panic": 0.20,                       # 20% of normal (defensive)
    "stress": 0.10,                      # 10% of normal (near shutdown)
}

# ============================================================
# ML MODEL PARAMETERS
# ============================================================

MODEL_MIN_CONFIDENCE = 0.30              # Minimum model output confidence
DRIFT_THRESHOLD = 0.15                   # KL divergence drift threshold
MIN_TRADES_FOR_EVALUATION = 50           # Min trades for win_rate calc
FEATURE_IMPORTANCE_TOP_N = 10            # Top N features to track
FEATURE_IMPORTANCE_ENABLED = True        # Feature tracking toggle

# ============================================================
# AUTO-RETRAINING TRIGGERS
# ============================================================

WIN_RATE_RETRAINING_THRESHOLD = 0.45     # Retrain if below 45%
SHARPE_RETRAINING_THRESHOLD = 0.5        # Retrain if below 0.5
FEATURE_DRIFT_RETRAINING_THRESHOLD = 0.15 # Retrain if drift > 15%
MAX_DAYS_WITHOUT_RETRAIN = 7             # Force retrain every 7 days

# ============================================================
# VALID RANGES for configuration validation
# ============================================================

VALID_RANGES: Dict[str, Tuple[float, float]] = {
    "DT_EXEC_MIN_CONF": (0.15, 0.60),
    "DT_MAX_POSITIONS": (1, 50),
    "DT_STOP_LOSS_PCT": (0.01, 0.10),
    "DT_TAKE_PROFIT_PCT": (0.02, 0.30),
    "DT_MAX_EXPOSURE_FRAC": (0.1, 1.0),
    "DT_VIX_SPIKE_THRESHOLD": (20, 60),
    "HOLD_MIN_TIME_MINUTES": (0, 60),
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to [min, max] range.
    
    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Clamped value within [min_val, max_val]
        
    Examples:
        >>> clamp(0.5, 0.0, 1.0)
        0.5
        >>> clamp(1.5, 0.0, 1.0)
        1.0
        >>> clamp(-0.5, 0.0, 1.0)
        0.0
    """
    return max(min_val, min(max_val, value))


def validate_confidence(conf: float) -> float:
    """Validate and clamp confidence to [0, CONFIDENCE_MAX].
    
    Args:
        conf: Confidence value to validate
        
    Returns:
        Validated confidence clamped to [0.0, CONFIDENCE_MAX]
        
    Examples:
        >>> validate_confidence(0.75)
        0.75
        >>> validate_confidence(1.5)
        0.99
        >>> validate_confidence(-0.1)
        0.0
    """
    return clamp(conf, 0.0, CONFIDENCE_MAX)


def validate_position_fraction(frac: float) -> float:
    """Validate and clamp position fraction.
    
    Args:
        frac: Position fraction to validate
        
    Returns:
        Validated fraction clamped to [0.0, POSITION_MAX_FRACTION]
        
    Examples:
        >>> validate_position_fraction(0.10)
        0.10
        >>> validate_position_fraction(0.20)
        0.15
        >>> validate_position_fraction(-0.05)
        0.0
    """
    return clamp(frac, 0.0, POSITION_MAX_FRACTION)
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
