"""Unit tests for dt_backend/core/constants_dt.py and auto_retrain_trigger.py."""

import pytest
from datetime import datetime, timezone, timedelta

from dt_backend.core.constants_dt import (
    # Confidence thresholds
    CONFIDENCE_MIN,
    CONFIDENCE_MIN_EXEC,
    CONFIDENCE_MIN_PROBE,
    CONFIDENCE_MAX,
    CONFIDENCE_EXIT_BUFFER,
    # Position sizing
    POSITION_MAX_FRACTION,
    POSITION_PROBE_FRACTION,
    POSITION_PRESS_MULT,
    POSITION_DEFAULT_QTY,
    POSITIONS_MAX_OPEN,
    # Hysteresis & signal stability
    EDGE_MIN_TO_FLIP,
    EDGE_HOLD_BIAS,
    CONFIRMATIONS_TO_FLIP,
    COOLDOWN_AFTER_BUY_MINUTES,
    HOLD_MIN_TIME_MINUTES,
    # P(Hit) calibration
    PHIT_MIN,
    PHIT_PRESS_MIN,
    LOSS_EST_PCT,
    # Volatility adjustment
    VOL_PENALTY_HIGH,
    VOL_PENALTY_MEDIUM,
    # Trend & regime
    TREND_BOOST_STRONG,
    TREND_BOOST_MILD,
    REGIME_PENALTY_CHOP,
    REGIME_PENALTY_BEAR_BUY,
    REGIME_PENALTY_BULL_SELL,
    # Signal thresholds
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    SCORE_MIN_ABS,
    # Risk management
    DAILY_LOSS_LIMIT,
    WEEKLY_DRAWDOWN_MAX_PCT,
    MONTHLY_DRAWDOWN_MAX_PCT,
    VIX_SPIKE_THRESHOLD,
    POSITIONS_MAX_PER_SECTOR,
    EXPOSURE_MAX,
    # Execution & order management
    ORDERS_MAX_PER_CYCLE,
    TRADE_GAP_MIN_MINUTES,
    ORDER_TIMEOUT_SEC,
    # Regime exposure
    REGIME_EXPOSURE,
    # ML model parameters
    MODEL_MIN_CONFIDENCE,
    DRIFT_THRESHOLD,
    MIN_TRADES_FOR_EVALUATION,
    FEATURE_IMPORTANCE_TOP_N,
    FEATURE_IMPORTANCE_ENABLED,
    # Auto-retraining triggers
    WIN_RATE_RETRAINING_THRESHOLD,
    SHARPE_RETRAINING_THRESHOLD,
    FEATURE_DRIFT_RETRAINING_THRESHOLD,
    MAX_DAYS_WITHOUT_RETRAIN,
    # Valid ranges
    VALID_RANGES,
    # Helper functions
    clamp,
    validate_confidence,
    validate_position_fraction,
)

from dt_backend.ml.auto_retrain_trigger import AutoRetrainTrigger


class TestConstants:
    """Tests for constant values and their validity."""

    def test_confidence_min_within_bounds(self):
        """CONFIDENCE_MIN should be between 0.15 and 0.60."""
        assert 0.15 <= CONFIDENCE_MIN <= 0.60

    def test_confidence_min_exec_within_bounds(self):
        """CONFIDENCE_MIN_EXEC should be between 0.15 and 0.60."""
        assert 0.15 <= CONFIDENCE_MIN_EXEC <= 0.60

    def test_confidence_max_reasonable(self):
        """CONFIDENCE_MAX should be close to 1.0 but not exceed it."""
        assert 0.95 <= CONFIDENCE_MAX <= 1.0

    def test_confidence_probe_is_lower_than_min(self):
        """CONFIDENCE_MIN_PROBE should be lower than CONFIDENCE_MIN."""
        assert CONFIDENCE_MIN_PROBE < CONFIDENCE_MIN
        assert CONFIDENCE_MIN_PROBE < CONFIDENCE_MIN_EXEC

    def test_position_max_fraction_reasonable(self):
        """POSITION_MAX_FRACTION should be between 0.05 and 0.30."""
        assert 0.05 <= POSITION_MAX_FRACTION <= 0.30

    def test_position_probe_fraction_reasonable(self):
        """POSITION_PROBE_FRACTION should be between 0.1 and 0.5."""
        assert 0.1 <= POSITION_PROBE_FRACTION <= 0.5

    def test_positions_max_open_reasonable(self):
        """POSITIONS_MAX_OPEN should be between 1 and 10."""
        assert 1 <= POSITIONS_MAX_OPEN <= 10

    def test_hysteresis_values_reasonable(self):
        """Hysteresis values should be reasonable."""
        assert EDGE_MIN_TO_FLIP > 0
        assert EDGE_HOLD_BIAS > 0
        assert EDGE_HOLD_BIAS < EDGE_MIN_TO_FLIP
        assert CONFIRMATIONS_TO_FLIP >= 1

    def test_time_values_non_negative(self):
        """Time-related constants should be non-negative."""
        assert COOLDOWN_AFTER_BUY_MINUTES >= 0
        assert HOLD_MIN_TIME_MINUTES >= 0
        assert TRADE_GAP_MIN_MINUTES >= 0

    def test_phit_calibration_values(self):
        """P(Hit) values should be between 0.5 and 1.0."""
        assert 0.5 <= PHIT_MIN <= 1.0
        assert 0.5 <= PHIT_PRESS_MIN <= 1.0
        assert PHIT_PRESS_MIN > PHIT_MIN

    def test_volatility_penalties_between_0_and_1(self):
        """Volatility penalties should be between 0 and 1."""
        assert 0 < VOL_PENALTY_HIGH < 1
        assert 0 < VOL_PENALTY_MEDIUM < 1
        assert VOL_PENALTY_HIGH < VOL_PENALTY_MEDIUM

    def test_trend_boosts_greater_than_1(self):
        """Trend boosts should be greater than 1."""
        assert TREND_BOOST_STRONG > 1.0
        assert TREND_BOOST_MILD > 1.0
        assert TREND_BOOST_STRONG > TREND_BOOST_MILD

    def test_regime_penalties_between_0_and_1(self):
        """Regime penalties should be between 0 and 1."""
        assert 0 < REGIME_PENALTY_CHOP <= 1
        assert 0 < REGIME_PENALTY_BEAR_BUY <= 1
        assert 0 < REGIME_PENALTY_BULL_SELL <= 1

    def test_signal_thresholds_symmetric(self):
        """Buy and sell thresholds should be symmetric."""
        assert BUY_THRESHOLD > 0
        assert SELL_THRESHOLD < 0
        assert abs(BUY_THRESHOLD) == abs(SELL_THRESHOLD)

    def test_risk_management_values_positive(self):
        """Risk management values should be positive."""
        assert DAILY_LOSS_LIMIT > 0
        assert WEEKLY_DRAWDOWN_MAX_PCT > 0
        assert MONTHLY_DRAWDOWN_MAX_PCT > 0
        assert VIX_SPIKE_THRESHOLD > 0
        assert POSITIONS_MAX_PER_SECTOR > 0
        assert 0 < EXPOSURE_MAX <= 1.0

    def test_execution_values_reasonable(self):
        """Execution parameters should be reasonable."""
        assert ORDERS_MAX_PER_CYCLE > 0
        assert ORDER_TIMEOUT_SEC > 0

    def test_regime_exposure_sums_correctly(self):
        """Regime exposure values should be positive and <= 1.0."""
        for regime, exposure in REGIME_EXPOSURE.items():
            assert 0 < exposure <= 1.0

    def test_regime_exposure_ordered(self):
        """Regime exposure should be ordered from highest to lowest risk."""
        assert REGIME_EXPOSURE["bull"] >= REGIME_EXPOSURE["chop"]
        assert REGIME_EXPOSURE["chop"] >= REGIME_EXPOSURE["bear"]
        assert REGIME_EXPOSURE["bear"] >= REGIME_EXPOSURE["panic"]
        assert REGIME_EXPOSURE["panic"] >= REGIME_EXPOSURE["stress"]

    def test_ml_parameters_reasonable(self):
        """ML model parameters should be reasonable."""
        assert 0 < MODEL_MIN_CONFIDENCE < 1
        assert DRIFT_THRESHOLD > 0
        assert MIN_TRADES_FOR_EVALUATION > 0
        assert FEATURE_IMPORTANCE_TOP_N > 0
        assert isinstance(FEATURE_IMPORTANCE_ENABLED, bool)

    def test_retraining_thresholds_reasonable(self):
        """Auto-retraining thresholds should be reasonable."""
        assert 0 < WIN_RATE_RETRAINING_THRESHOLD < 1
        assert SHARPE_RETRAINING_THRESHOLD > 0
        assert FEATURE_DRIFT_RETRAINING_THRESHOLD > 0
        assert MAX_DAYS_WITHOUT_RETRAIN > 0

    def test_valid_ranges_structure(self):
        """VALID_RANGES should have proper structure."""
        assert isinstance(VALID_RANGES, dict)
        for key, (min_val, max_val) in VALID_RANGES.items():
            assert isinstance(key, str)
            assert min_val < max_val
            assert isinstance(min_val, (int, float))
            assert isinstance(max_val, (int, float))


class TestHelperFunctions:
    """Tests for helper functions in constants_dt."""

    def test_clamp_within_range(self):
        """Clamp should return value when within range."""
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_clamp_below_min(self):
        """Clamp should return min when value is below."""
        assert clamp(-0.5, 0.0, 1.0) == 0.0

    def test_clamp_above_max(self):
        """Clamp should return max when value is above."""
        assert clamp(1.5, 0.0, 1.0) == 1.0

    def test_validate_confidence_within_range(self):
        """validate_confidence should return value when within range."""
        assert validate_confidence(0.75) == 0.75

    def test_validate_confidence_above_max(self):
        """validate_confidence should clamp to CONFIDENCE_MAX."""
        assert validate_confidence(1.5) == CONFIDENCE_MAX

    def test_validate_confidence_below_zero(self):
        """validate_confidence should clamp to 0.0."""
        assert validate_confidence(-0.1) == 0.0

    def test_validate_position_fraction_within_range(self):
        """validate_position_fraction should return value when within range."""
        assert validate_position_fraction(0.10) == 0.10

    def test_validate_position_fraction_above_max(self):
        """validate_position_fraction should clamp to POSITION_MAX_FRACTION."""
        assert validate_position_fraction(0.20) == POSITION_MAX_FRACTION

    def test_validate_position_fraction_below_zero(self):
        """validate_position_fraction should clamp to 0.0."""
        assert validate_position_fraction(-0.05) == 0.0


class TestAutoRetrainTrigger:
    """Tests for AutoRetrainTrigger class."""

    def test_initialization(self):
        """AutoRetrainTrigger should initialize with empty state."""
        trigger = AutoRetrainTrigger()
        assert trigger.metrics_history == []
        assert trigger.last_retrain_date is None

    def test_trigger_on_low_win_rate(self):
        """Auto-retrain should trigger on low win rate."""
        trigger = AutoRetrainTrigger()
        should_retrain, reason = trigger.check_and_trigger({
            "win_rate": 0.40,  # Below 0.45 threshold
            "sharpe_ratio": 1.0,
            "feature_drift": 0.05,
        })
        assert should_retrain is True
        assert "win_rate" in reason.lower()

    def test_trigger_on_low_sharpe(self):
        """Auto-retrain should trigger on low Sharpe ratio."""
        trigger = AutoRetrainTrigger()
        should_retrain, reason = trigger.check_and_trigger({
            "win_rate": 0.60,
            "sharpe_ratio": 0.3,  # Below 0.5 threshold
            "feature_drift": 0.05,
        })
        assert should_retrain is True
        assert "sharpe" in reason.lower()

    def test_trigger_on_high_drift(self):
        """Auto-retrain should trigger on high feature drift."""
        trigger = AutoRetrainTrigger()
        should_retrain, reason = trigger.check_and_trigger({
            "win_rate": 0.60,
            "sharpe_ratio": 1.0,
            "feature_drift": 0.20,  # Above 0.15 threshold
        })
        assert should_retrain is True
        assert "drift" in reason.lower()

    def test_trigger_on_schedule(self):
        """Auto-retrain should trigger after max days without retrain."""
        trigger = AutoRetrainTrigger()
        # Set last retrain to 8 days ago
        trigger.last_retrain_date = datetime.now(timezone.utc) - timedelta(days=8)
        
        should_retrain, reason = trigger.check_and_trigger({
            "win_rate": 0.60,
            "sharpe_ratio": 1.0,
            "feature_drift": 0.05,
        })
        assert should_retrain is True
        assert "schedule" in reason.lower()

    def test_no_trigger_healthy_metrics(self):
        """Auto-retrain should not trigger with healthy metrics."""
        trigger = AutoRetrainTrigger()
        # Set recent retrain
        trigger.last_retrain_date = datetime.now(timezone.utc) - timedelta(days=2)
        
        should_retrain, reason = trigger.check_and_trigger({
            "win_rate": 0.60,
            "sharpe_ratio": 1.0,
            "feature_drift": 0.05,
        })
        assert should_retrain is False
        assert reason == "metrics_healthy"

    def test_multiple_triggers(self):
        """Auto-retrain reason should include multiple triggers."""
        trigger = AutoRetrainTrigger()
        should_retrain, reason = trigger.check_and_trigger({
            "win_rate": 0.40,  # Below threshold
            "sharpe_ratio": 0.3,  # Below threshold
            "feature_drift": 0.05,
        })
        assert should_retrain is True
        assert "win_rate" in reason.lower()
        assert "sharpe" in reason.lower()

    def test_record_retrain(self):
        """record_retrain should update last_retrain_date."""
        trigger = AutoRetrainTrigger()
        assert trigger.last_retrain_date is None
        
        trigger.record_retrain()
        assert trigger.last_retrain_date is not None
        assert isinstance(trigger.last_retrain_date, datetime)

    def test_record_retrain_with_timestamp(self):
        """record_retrain should accept custom timestamp."""
        trigger = AutoRetrainTrigger()
        custom_time = datetime(2024, 1, 15, tzinfo=timezone.utc)
        
        trigger.record_retrain(custom_time)
        assert trigger.last_retrain_date == custom_time

    def test_days_since_last_retrain_infinity(self):
        """_days_since_last_retrain should return infinity if never retrained."""
        trigger = AutoRetrainTrigger()
        assert trigger._days_since_last_retrain() == float('inf')

    def test_days_since_last_retrain_calculation(self):
        """_days_since_last_retrain should calculate days correctly."""
        trigger = AutoRetrainTrigger()
        trigger.last_retrain_date = datetime.now(timezone.utc) - timedelta(days=3)
        
        days = trigger._days_since_last_retrain()
        assert 2.5 < days < 3.5  # Allow some tolerance for test execution time

    def test_metrics_history_tracking(self):
        """AutoRetrainTrigger should track metrics history."""
        trigger = AutoRetrainTrigger()
        
        metrics1 = {"win_rate": 0.50, "sharpe_ratio": 0.8, "feature_drift": 0.10}
        metrics2 = {"win_rate": 0.45, "sharpe_ratio": 0.6, "feature_drift": 0.12}
        
        trigger.check_and_trigger(metrics1)
        trigger.check_and_trigger(metrics2)
        
        assert len(trigger.metrics_history) == 2
        assert trigger.metrics_history[0] == metrics1
        assert trigger.metrics_history[1] == metrics2

    def test_get_metrics_summary(self):
        """get_metrics_summary should return status information."""
        trigger = AutoRetrainTrigger()
        
        summary = trigger.get_metrics_summary()
        assert "last_retrain_date" in summary
        assert "days_since_retrain" in summary
        assert "metrics_count" in summary
        assert summary["last_retrain_date"] is None
        assert summary["days_since_retrain"] == float('inf')
        assert summary["metrics_count"] == 0

    def test_get_metrics_summary_after_retrain(self):
        """get_metrics_summary should include retrain info."""
        trigger = AutoRetrainTrigger()
        trigger.record_retrain()
        
        summary = trigger.get_metrics_summary()
        assert summary["last_retrain_date"] is not None
        assert summary["days_since_retrain"] < 1.0  # Just recorded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
