"""Unit tests for missed opportunity tracker."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def temp_learning_path():
    """Create a temporary directory for learning data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_missed_opportunity_tracker_init(temp_learning_path):
    """Test MissedOpportunityTracker initialization."""
    from dt_backend.ml.missed_opportunity_tracker import MissedOpportunityTracker
    
    tracker = MissedOpportunityTracker(temp_learning_path)
    
    assert tracker.data_path == temp_learning_path
    assert tracker.data_path.exists()


def test_log_missed_signal_high_confidence(temp_learning_path):
    """Test logging a high-confidence missed signal."""
    from dt_backend.ml.missed_opportunity_tracker import MissedOpportunityTracker
    
    tracker = MissedOpportunityTracker(temp_learning_path)
    
    signal = {
        "symbol": "AAPL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": "BUY",
        "confidence": 0.75,
        "price": 180.50,
        "lgb_prob": 0.75,
        "regime": "trending",
    }
    
    # Should not raise
    tracker.log_missed_signal(signal, "max_positions")
    
    # Check file was created
    assert tracker.missed_signals_file.exists()


def test_log_missed_signal_low_confidence(temp_learning_path):
    """Test that low-confidence signals are not logged."""
    from dt_backend.ml.missed_opportunity_tracker import MissedOpportunityTracker
    
    tracker = MissedOpportunityTracker(temp_learning_path)
    
    signal = {
        "symbol": "AAPL",
        "confidence": 0.50,  # Below 0.60 threshold
        "price": 180.50,
    }
    
    tracker.log_missed_signal(signal, "confidence_threshold")
    
    # File should not be created for low confidence
    assert not tracker.missed_signals_file.exists()


def test_check_missed_outcomes_no_signals(temp_learning_path):
    """Test checking outcomes when no signals exist."""
    from dt_backend.ml.missed_opportunity_tracker import MissedOpportunityTracker
    
    tracker = MissedOpportunityTracker(temp_learning_path)
    
    evaluated = tracker.check_missed_outcomes()
    
    assert evaluated == 0


def test_analyze_missed_patterns_empty(temp_learning_path):
    """Test analyzing patterns with no data."""
    from dt_backend.ml.missed_opportunity_tracker import MissedOpportunityTracker
    
    tracker = MissedOpportunityTracker(temp_learning_path)
    
    patterns = tracker.analyze_missed_patterns()
    
    assert "missed_pnl_usd" in patterns
    assert "profitable_missed_pct" in patterns
    assert patterns["missed_pnl_usd"] == 0


def test_track_missed_signal_entry_point(temp_learning_path):
    """Test the main entry point function."""
    from dt_backend.ml.missed_opportunity_tracker import track_missed_signal
    
    # Monkey-patch DT_PATHS
    import dt_backend.ml.missed_opportunity_tracker as module
    original_paths = module.DT_PATHS
    module.DT_PATHS = {"learning": temp_learning_path}
    
    try:
        signal = {
            "symbol": "AAPL",
            "confidence": 0.75,
            "price": 180.50,
            "label": "BUY",
        }
        
        # Should not raise
        track_missed_signal(signal, "max_positions")
        
    finally:
        module.DT_PATHS = original_paths


def test_missed_signal_dataclass():
    """Test MissedSignal dataclass."""
    from dt_backend.ml.missed_opportunity_tracker import MissedSignal
    
    signal = MissedSignal(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc).isoformat(),
        label="BUY",
        confidence=0.75,
        price_at_signal=180.50,
        reason_not_traded="max_positions",
        lgb_prob=0.75,
    )
    
    assert signal.symbol == "AAPL"
    assert signal.confidence == 0.75
    assert signal.evaluated is False


def test_analyze_missed_today_entry_point(temp_learning_path):
    """Test analyze_missed_today entry point."""
    from dt_backend.ml.missed_opportunity_tracker import analyze_missed_today
    
    # Monkey-patch DT_PATHS
    import dt_backend.ml.missed_opportunity_tracker as module
    original_paths = module.DT_PATHS
    module.DT_PATHS = {"learning": temp_learning_path}
    
    try:
        result = analyze_missed_today()
        
        assert "status" in result
        assert result["status"] == "success"
        
    finally:
        module.DT_PATHS = original_paths
