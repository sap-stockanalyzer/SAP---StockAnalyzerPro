"""Unit tests for trade outcome analyzer."""

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest


@pytest.fixture
def temp_learning_path():
    """Create a temporary directory for learning data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_trade_outcome_analyzer_init(temp_learning_path):
    """Test TradeOutcomeAnalyzer initialization."""
    from dt_backend.ml.trade_outcome_analyzer import TradeOutcomeAnalyzer
    
    analyzer = TradeOutcomeAnalyzer(temp_learning_path)
    
    assert analyzer.data_path == temp_learning_path
    assert analyzer.data_path.exists()
    assert analyzer.trades_file.parent.exists()


def test_process_closed_trade(temp_learning_path):
    """Test processing a closed trade."""
    from dt_backend.ml.trade_outcome_analyzer import TradeOutcomeAnalyzer
    
    analyzer = TradeOutcomeAnalyzer(temp_learning_path)
    
    trade_dict = {
        "type": "exit",
        "symbol": "AAPL",
        "side": "BUY",
        "price": 180.50,
        "entry_price": 180.00,
        "qty": 10,
        "confidence": 0.75,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "entry_timestamp": (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat(),
        "label": "BUY",
        "lgb_prob": 0.75,
        "regime": "trending",
        "exit_reason": "take_profit",
    }
    
    outcome = analyzer.process_closed_trade(trade_dict)
    
    assert outcome is not None
    assert outcome.symbol == "AAPL"
    assert outcome.side == "BUY"
    assert outcome.success  # Entry < Exit for BUY
    assert outcome.pnl_pct > 0


def test_classify_time_of_day():
    """Test time of day classification."""
    from dt_backend.ml.trade_outcome_analyzer import TradeOutcomeAnalyzer
    
    analyzer = TradeOutcomeAnalyzer()
    
    # Test open (9:30-10:30 ET)
    timestamp = "2026-01-16T09:45:00-05:00"
    tod = analyzer._classify_time_of_day(timestamp)
    assert tod == "open"
    
    # Test afternoon (14:00-15:45 ET)
    timestamp = "2026-01-16T14:30:00-05:00"
    tod = analyzer._classify_time_of_day(timestamp)
    assert tod == "afternoon"


def test_classify_quality():
    """Test trade quality classification."""
    from dt_backend.ml.trade_outcome_analyzer import TradeOutcomeAnalyzer
    
    analyzer = TradeOutcomeAnalyzer()
    
    assert analyzer._classify_quality(0.05) == "great"
    assert analyzer._classify_quality(0.02) == "good"
    assert analyzer._classify_quality(0.005) == "mediocre"
    assert analyzer._classify_quality(-0.02) == "bad"


def test_update_metrics(temp_learning_path):
    """Test updating metrics after trade."""
    from dt_backend.ml.trade_outcome_analyzer import TradeOutcomeAnalyzer, TradeOutcome
    
    analyzer = TradeOutcomeAnalyzer(temp_learning_path)
    
    outcome = TradeOutcome(
        symbol="AAPL",
        entry_time=datetime.now(timezone.utc).isoformat(),
        exit_time=datetime.now(timezone.utc).isoformat(),
        side="BUY",
        entry_price=180.0,
        exit_price=181.0,
        pnl=10.0,
        pnl_pct=0.0055,
        hold_duration_minutes=30,
        model_label="BUY",
        model_confidence=0.75,
        lgb_prob=0.75,
        regime="trending",
        time_of_day="open",
        success=True,
        quality="good",
        exit_reason="take_profit",
    )
    
    analyzer.update_metrics(outcome)
    
    metrics = analyzer._load_metrics()
    assert metrics["global"]["total_trades"] == 1
    assert metrics["global"]["wins"] == 1
    assert metrics["global"]["win_rate"] == 1.0


def test_get_performance_window(temp_learning_path):
    """Test getting performance window."""
    from dt_backend.ml.trade_outcome_analyzer import TradeOutcomeAnalyzer
    
    analyzer = TradeOutcomeAnalyzer(temp_learning_path)
    
    # Get performance (should be empty initially)
    perf = analyzer.get_performance_window(days=7)
    
    assert "win_rate" in perf
    assert "total_trades" in perf
    assert perf["total_trades"] == 0


def test_analyze_trade_outcome_entry_point(temp_learning_path):
    """Test the main entry point function."""
    from dt_backend.ml.trade_outcome_analyzer import analyze_trade_outcome
    
    # Monkey-patch DT_PATHS
    import dt_backend.ml.trade_outcome_analyzer as module
    original_paths = module.DT_PATHS
    module.DT_PATHS = {"learning": temp_learning_path}
    
    try:
        trade = {
            "type": "exit",
            "symbol": "AAPL",
            "side": "BUY",
            "price": 180.50,
            "entry_price": 180.00,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": 0.75,
        }
        
        # Should not raise
        analyze_trade_outcome(trade)
        
    finally:
        module.DT_PATHS = original_paths


def test_consecutive_wins_losses(temp_learning_path):
    """Test counting consecutive wins and losses."""
    from dt_backend.ml.trade_outcome_analyzer import TradeOutcomeAnalyzer
    
    analyzer = TradeOutcomeAnalyzer(temp_learning_path)
    
    trades = [
        {"success": True},
        {"success": True},
        {"success": False},
        {"success": True},
        {"success": True},
        {"success": True},
    ]
    
    assert analyzer._count_consecutive_wins(trades) == 3
    assert analyzer._count_consecutive_losses(trades) == 0
    
    trades_losses = [
        {"success": True},
        {"success": False},
        {"success": False},
        {"success": False},
    ]
    
    assert analyzer._count_consecutive_losses(trades_losses) == 3
    assert analyzer._count_consecutive_wins(trades_losses) == 0
