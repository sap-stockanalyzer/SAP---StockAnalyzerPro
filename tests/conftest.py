"""Pytest configuration and fixtures for AION Analytics tests."""

import pytest
import os
from pathlib import Path
from typing import Dict, Any


@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables for testing."""
    # Save original env
    original_env = os.environ.copy()
    
    # Set test defaults
    test_env = {
        "DT_DRY_RUN": "1",
        "DT_ENABLE_LIVE_TRADING": "0",
        "DT_MAX_POSITIONS": "3",
        "DT_EXEC_MIN_CONF": "0.25",
        "DT_DAILY_LOSS_LIMIT_USD": "300.0",
        "DT_MAX_WEEKLY_DRAWDOWN_PCT": "8.0",
        "DT_MAX_MONTHLY_DRAWDOWN_PCT": "15.0",
        "DT_VIX_SPIKE_THRESHOLD": "35.0",
        "DT_MAX_LOSS_PER_SYMBOL_DAY": "500.0",
        "DT_TRUTH_DIR": "/tmp/test_da_brains",
        "DT_EMERGENCY_STOP_FILE": "/tmp/test_emergency_stop",
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    yield test_env
    
    # Cleanup is handled automatically by monkeypatch


@pytest.fixture
def sample_rolling_data() -> Dict[str, Any]:
    """Sample rolling cache data for tests."""
    return {
        "AAPL": {
            "context_dt": {
                "intraday_return": 0.015,
                "intraday_range": 0.025,
                "intraday_vol": 0.008,
                "last_price": 150.25,
                "intraday_trend": "bull",
                "vol_bucket": "medium",
                "has_intraday_data": True,
            },
            "features_dt": {
                "rsi_14": 55.0,
                "atr_14": 2.5,
                "vwap_dist": 0.005,
            },
            "predictions_dt": {
                "signal": 0.65,
                "confidence": 0.75,
            },
            "policy_dt": {
                "side": "BUY",
                "size": 100.0,
                "conf": 0.75,
            },
        },
        "MSFT": {
            "context_dt": {
                "intraday_return": -0.010,
                "intraday_range": 0.020,
                "intraday_vol": 0.006,
                "last_price": 380.50,
                "intraday_trend": "bear",
                "vol_bucket": "low",
                "has_intraday_data": True,
            },
            "features_dt": {
                "rsi_14": 42.0,
                "atr_14": 3.0,
                "vwap_dist": -0.008,
            },
            "predictions_dt": {
                "signal": -0.45,
                "confidence": 0.60,
            },
            "policy_dt": {
                "side": "SELL",
                "size": 50.0,
                "conf": 0.60,
            },
        },
        "_GLOBAL_DT": {
            "vix_level": 18.5,
            "vix_spike": False,
            "vix_threshold": 35.0,
            "candidate_universe_dt": {
                "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA"],
                "n": 4,
            },
        },
    }


@pytest.fixture
def sample_risk_state() -> Dict[str, Any]:
    """Sample risk rails state for tests."""
    return {
        "date": "2024-01-15",
        "week_start": "2024-01-15",
        "month_start": "2024-01-01",
        "equity_source": "ledger",
        "start_equity": 100000.0,
        "peak_equity": 102000.0,
        "week_start_equity": 100000.0,
        "week_peak_equity": 102000.0,
        "month_start_equity": 100000.0,
        "month_peak_equity": 103000.0,
        "last_realized": 1500.0,
        "consec_loss_deltas": 0,
        "cooldown_until": "",
        "ts": "2024-01-15T14:30:00Z",
    }


@pytest.fixture
def sample_metrics() -> Dict[str, Any]:
    """Sample metrics data for tests."""
    return {
        "bots": {
            "DT": {
                "equity_est": 101500.0,
                "positions_value_est": 15000.0,
                "realized_pnl_est": 1500.0,
                "positions": 2,
            }
        },
        "equity": 101500.0,
        "open_positions": 2,
        "realized_pnl_today": 500.0,
    }


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create temporary test directory."""
    test_dir = tmp_path / "dt_backend_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    intraday_dir = test_dir / "intraday"
    intraday_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Cleanup test files after each test."""
    yield
    
    # Cleanup emergency stop file
    emergency_file = Path("/tmp/test_emergency_stop")
    if emergency_file.exists():
        try:
            emergency_file.unlink()
        except (PermissionError, FileNotFoundError, OSError):
            # Ignore cleanup errors - test environment might have restricted permissions
            pass
