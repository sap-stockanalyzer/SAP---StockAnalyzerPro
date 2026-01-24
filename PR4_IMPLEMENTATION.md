# PR #4: Schema Validation + P&L Attribution Dashboard

## Overview
This PR adds comprehensive schema validation for all data structures and creates a P&L attribution dashboard for tracking strategy performance by feature and strategy type.

## Implementation Summary

### New Files Created

1. **dt_backend/core/constants_dt.py** (70 LOC)
   - `VALID_RANGES` dictionary with min/max for all config parameters
   - Valid action types, sides, statuses, and timeframes
   - Used by schema validator for centralized validation

2. **dt_backend/core/schema_validator_dt.py** (240 LOC)
   - `ValidationError` exception class
   - `ActionType` enum
   - `PolicyDTSchema` - validates policy_dt nodes
   - `ExecutionDTSchema` - validates execution_dt nodes
   - `PositionDTSchema` - validates position_dt nodes
   - `RollingNodeSchema` - validates entire rolling nodes
   - `validate_rolling()` - validates complete rolling dict
   - `validate_config_value()` - validates config against VALID_RANGES

3. **backend/routers/pnl_dashboard_router.py** (330 LOC)
   - `/api/pnl/dashboard` - comprehensive P&L dashboard
   - `/api/pnl/metrics` - key metrics for monitoring
   - `/api/pnl/attribution/{symbol}` - per-symbol attribution
   - Helper functions for:
     - P&L by strategy
     - P&L by feature
     - Win rate calculation
     - Sharpe ratio calculation
     - Max drawdown calculation
     - Daily history aggregation

4. **frontend/app/dashboard/page.tsx** (250 LOC)
   - React component for P&L dashboard UI
   - Real-time metrics display
   - Strategy and feature performance breakdown
   - Auto-refresh every 30 seconds
   - Responsive design

5. **tests/unit/test_schema_validator_dt.py** (280 LOC)
   - 23 comprehensive unit tests
   - Tests for all schema validators
   - Tests for config validation
   - All tests passing ✅

### Files Updated

1. **dt_backend/core/policy_engine_dt.py**
   - Added `validate_rolling()` call before returning
   - Logs validation errors but allows graceful degradation
   - ~10 lines added

2. **dt_backend/engines/trade_executor.py**
   - Added `validate_rolling()` call after reading rolling data
   - Returns error dict if validation fails
   - ~12 lines added

3. **dt_backend/core/knob_validator_dt.py**
   - Replaced manual range checks with `validate_config_value()`
   - Uses `VALID_RANGES` from constants_dt
   - Simplified from ~60 lines to ~15 lines
   - Maintains backward compatibility

4. **backend/backend_service.py**
   - Added import for `pnl_dashboard_router`
   - Added router to ROUTERS list
   - ~3 lines added

## Features Implemented

### Schema Validation
- ✅ Validates policy_dt, execution_dt, position_dt, and rolling nodes
- ✅ Clear error messages with symbol context
- ✅ Type checking (str, float, bool, dict)
- ✅ Range validation (0.0-1.0 for confidence, etc.)
- ✅ Enum validation (BUY/SELL/HOLD/STAND_DOWN)
- ✅ Optional field validation
- ✅ Config value validation against VALID_RANGES

### P&L Dashboard
- ✅ Daily/weekly/monthly/YTD P&L tracking
- ✅ Win rate calculation
- ✅ Sharpe ratio (annualized)
- ✅ Maximum drawdown percentage
- ✅ P&L attribution by strategy (ORB, VWAP_MR, etc.)
- ✅ P&L attribution by feature (momentum, volatility, etc.)
- ✅ Trade count and average trade size
- ✅ Daily history with trade counts
- ✅ Symbol-specific attribution endpoint

### Frontend Dashboard
- ✅ Key metrics cards (daily P&L, win rate, Sharpe, drawdown)
- ✅ Period performance (weekly, monthly, YTD)
- ✅ Strategy performance breakdown with trade counts
- ✅ Feature contribution analysis
- ✅ Trading statistics
- ✅ Real-time updates (30s refresh)
- ✅ Color-coded P&L (green/red)

## Testing

### Unit Tests
```bash
$ pytest tests/unit/test_schema_validator_dt.py -v
# 23 tests PASSED ✅

$ pytest tests/unit/test_knob_validator.py -v
# 3 tests PASSED ✅
```

### Manual Testing
- ✅ Schema validator imports correctly
- ✅ P&L router registers successfully
- ✅ All endpoints defined correctly
- ✅ Helper functions calculate correctly
- ✅ Config validation works with valid/invalid ranges
- ✅ Policy validation catches invalid actions

## API Endpoints

### GET /api/pnl/dashboard
Returns comprehensive P&L dashboard with:
- Daily/weekly/monthly/YTD P&L
- Win rate, total trades, avg trade size
- Max drawdown %, Sharpe ratio
- P&L by strategy (with trades and win rate)
- P&L by feature (with contribution %)
- Daily history (last 30 days)

### GET /api/pnl/metrics
Returns key monitoring metrics:
- daily_pnl
- win_rate
- sharpe_ratio
- max_drawdown_pct

### GET /api/pnl/attribution/{symbol}
Returns symbol-specific attribution:
- Total P&L for symbol
- Trade count
- Win rate
- Avg P&L per trade
- Last 10 trades detail

## Benefits

1. **Fail Fast** - Schema validation catches bad data immediately at policy/execution time
2. **Clear Errors** - Validation errors show exactly which field failed and why
3. **Type Safety** - Strong validation prevents invalid data from propagating through the system
4. **P&L Visibility** - Real-time dashboard shows strategy and feature performance
5. **Attribution** - Understand which features and strategies drive P&L
6. **Debugging** - Quickly identify which features are contributing to profits/losses
7. **Monitoring** - Track Sharpe ratio, max drawdown, win rate in real-time
8. **Centralized Config** - VALID_RANGES provides single source of truth for all config parameters

## Dependencies

- **Depends on:** PR #3 (uses `constants_dt.VALID_RANGES`)
- **Enables:** PR #5 (validated data prevents bugs)

## Code Quality

- ✅ All tests passing (26/26)
- ✅ Type hints throughout
- ✅ Comprehensive docstrings
- ✅ Error handling with graceful degradation
- ✅ Backward compatible
- ✅ No breaking changes

## Files Summary

**New Files:** 5
- `dt_backend/core/constants_dt.py`
- `dt_backend/core/schema_validator_dt.py`
- `backend/routers/pnl_dashboard_router.py`
- `frontend/app/dashboard/page.tsx`
- `tests/unit/test_schema_validator_dt.py`

**Updated Files:** 4
- `dt_backend/core/policy_engine_dt.py`
- `dt_backend/engines/trade_executor.py`
- `dt_backend/core/knob_validator_dt.py`
- `backend/backend_service.py`

**Total New Code:** ~1170 LOC
**Tests:** 26 passing ✅
