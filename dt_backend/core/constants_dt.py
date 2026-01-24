"""
dt_backend/core/constants_dt.py

Constants for DT trading engine.
Extracted from PR requirements to support bug fixes and ML features.
"""

# Bug Fix #1: Minimum hold time for positions
# Prevents 2-minute buy/sell flips
HOLD_MIN_TIME_MINUTES = 10  # Minimum minutes to hold a position before allowing exit

# Bug Fix #2: Position sizing constants
POSITION_MAX_FRACTION = 0.15  # Maximum fraction of portfolio per symbol

# ML Feature pipeline constants
FEATURE_IMPORTANCE_TOP_N = 20  # Top N features to track for importance
