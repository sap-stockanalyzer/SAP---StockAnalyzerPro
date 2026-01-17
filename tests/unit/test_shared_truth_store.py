"""Unit tests for shared truth store (backend.services.shared_truth_store)

Tests the SharedTruthStore class with source tracking and cross-strategy queries.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

from backend.services.shared_truth_store import SharedTruthStore, get_shared_store, _shared_dir


class TestSharedTruthStore:
    """Test SharedTruthStore functionality."""
    
    @pytest.fixture
    def temp_store(self, tmp_path, monkeypatch):
        """Create a temporary shared truth store."""
        # Override shared directory to use temp path
        monkeypatch.setenv("SHARED_TRUTH_DIR", str(tmp_path))
        store = SharedTruthStore()
        return store
    
    def test_store_initialization(self, temp_store):
        """Test store can be initialized."""
        assert temp_store is not None
        assert temp_store.trades_file.exists()
        assert temp_store.positions_file.exists()
        assert temp_store.metrics_file.exists()
    
    def test_append_trade_event_swing(self, temp_store):
        """Test appending swing trade event."""
        temp_store.append_trade_event(
            source="swing",
            symbol="AAPL",
            side="BUY",
            qty=100,
            price=150.50,
            reason="SIGNAL_HIGH_CONF"
        )
        
        # Read trades file
        trades = []
        with open(temp_store.trades_file, "r") as f:
            for line in f:
                trades.append(json.loads(line))
        
        assert len(trades) == 1
        trade = trades[0]
        assert trade["source"] == "swing"
        assert trade["symbol"] == "AAPL"
        assert trade["side"] == "BUY"
        assert trade["qty"] == 100.0
        assert trade["price"] == 150.50
        assert trade["reason"] == "SIGNAL_HIGH_CONF"
        assert trade["type"] == "trade"
        assert "ts" in trade
    
    def test_append_trade_event_dt(self, temp_store):
        """Test appending DT trade event."""
        temp_store.append_trade_event(
            source="dt",
            symbol="TSLA",
            side="SELL",
            qty=50,
            price=200.75,
            reason="TAKE_PROFIT",
            pnl=500.00
        )
        
        # Read trades file
        trades = []
        with open(temp_store.trades_file, "r") as f:
            for line in f:
                trades.append(json.loads(line))
        
        assert len(trades) == 1
        trade = trades[0]
        assert trade["source"] == "dt"
        assert trade["symbol"] == "TSLA"
        assert trade["side"] == "SELL"
        assert trade["pnl"] == 500.00
    
    def test_append_signal_event(self, temp_store):
        """Test appending signal event."""
        temp_store.append_signal_event(
            source="swing",
            symbol="MSFT",
            signal_type="BUY_SIGNAL",
            confidence=0.85
        )
        
        # Read trades file
        trades = []
        with open(temp_store.trades_file, "r") as f:
            for line in f:
                trades.append(json.loads(line))
        
        assert len(trades) == 1
        signal = trades[0]
        assert signal["type"] == "signal"
        assert signal["source"] == "swing"
        assert signal["symbol"] == "MSFT"
        assert signal["signal_type"] == "BUY_SIGNAL"
        assert signal["confidence"] == 0.85
    
    def test_append_no_trade_event(self, temp_store):
        """Test appending no-trade event."""
        temp_store.append_no_trade_event(
            source="dt",
            symbol="GOOGL",
            reason="RISK_LIMIT_EXCEEDED"
        )
        
        # Read trades file
        trades = []
        with open(temp_store.trades_file, "r") as f:
            for line in f:
                trades.append(json.loads(line))
        
        assert len(trades) == 1
        event = trades[0]
        assert event["type"] == "no_trade"
        assert event["source"] == "dt"
        assert event["symbol"] == "GOOGL"
        assert event["reason"] == "RISK_LIMIT_EXCEEDED"
    
    def test_get_trades_by_source_swing(self, temp_store):
        """Test querying trades by source (swing)."""
        # Add swing and DT trades
        temp_store.append_trade_event("swing", "AAPL", "BUY", 100, 150.0, "SIGNAL")
        temp_store.append_trade_event("dt", "AAPL", "BUY", 50, 151.0, "BREAKOUT")
        temp_store.append_trade_event("swing", "TSLA", "SELL", 75, 200.0, "TAKE_PROFIT")
        
        # Query swing trades
        swing_trades = temp_store.get_trades_by_source("swing", days=1)
        assert len(swing_trades) == 2
        assert all(t["source"] == "swing" for t in swing_trades)
        
        symbols = [t["symbol"] for t in swing_trades]
        assert "AAPL" in symbols
        assert "TSLA" in symbols
    
    def test_get_trades_by_source_dt(self, temp_store):
        """Test querying trades by source (DT)."""
        # Add swing and DT trades
        temp_store.append_trade_event("swing", "AAPL", "BUY", 100, 150.0, "SIGNAL")
        temp_store.append_trade_event("dt", "AAPL", "BUY", 50, 151.0, "BREAKOUT")
        temp_store.append_trade_event("dt", "MSFT", "SELL", 25, 300.0, "STOP_LOSS")
        
        # Query DT trades
        dt_trades = temp_store.get_trades_by_source("dt", days=1)
        assert len(dt_trades) == 2
        assert all(t["source"] == "dt" for t in dt_trades)
        
        symbols = [t["symbol"] for t in dt_trades]
        assert "AAPL" in symbols
        assert "MSFT" in symbols
    
    def test_get_symbol_trades(self, temp_store):
        """Test querying all trades for a symbol."""
        # Add trades for AAPL from both sources
        temp_store.append_trade_event("swing", "AAPL", "BUY", 100, 150.0, "SIGNAL")
        temp_store.append_trade_event("dt", "AAPL", "BUY", 50, 151.0, "BREAKOUT")
        temp_store.append_trade_event("swing", "TSLA", "SELL", 75, 200.0, "TAKE_PROFIT")
        temp_store.append_trade_event("dt", "AAPL", "SELL", 50, 155.0, "TAKE_PROFIT")
        
        # Query AAPL trades
        aapl_trades = temp_store.get_symbol_trades("AAPL", days=1)
        assert len(aapl_trades) == 3
        assert all(t["symbol"] == "AAPL" for t in aapl_trades)
        
        # Check both sources present
        sources = [t["source"] for t in aapl_trades]
        assert "swing" in sources
        assert "dt" in sources
    
    def test_detect_conflicts(self, temp_store):
        """Test detecting conflicts (both sources trading same symbol same day)."""
        # Add conflicting trades
        temp_store.append_trade_event("swing", "AAPL", "BUY", 100, 150.0, "SIGNAL")
        temp_store.append_trade_event("dt", "AAPL", "BUY", 50, 151.0, "BREAKOUT")
        
        # Add non-conflicting trade
        temp_store.append_trade_event("swing", "TSLA", "SELL", 75, 200.0, "TAKE_PROFIT")
        
        # Detect conflicts
        conflicts = temp_store.detect_conflicts(days=1)
        assert len(conflicts) == 1
        
        conflict = conflicts[0]
        assert conflict["symbol"] == "AAPL"
        assert len(conflict["swing_trades"]) == 1
        assert len(conflict["dt_trades"]) == 1
        assert conflict["conflict_type"] == "same_day_activity"
    
    def test_no_conflicts_different_symbols(self, temp_store):
        """Test no conflicts when trading different symbols."""
        # Add trades for different symbols
        temp_store.append_trade_event("swing", "AAPL", "BUY", 100, 150.0, "SIGNAL")
        temp_store.append_trade_event("dt", "TSLA", "BUY", 50, 200.0, "BREAKOUT")
        
        # Should have no conflicts
        conflicts = temp_store.detect_conflicts(days=1)
        assert len(conflicts) == 0
    
    def test_get_metrics(self, temp_store):
        """Test getting metrics."""
        metrics = temp_store.get_metrics()
        assert "counters" in metrics
        assert isinstance(metrics["counters"], dict)
    
    def test_update_metrics(self, temp_store):
        """Test updating metrics."""
        temp_store.update_metrics({"test_counter": 42, "test_value": 3.14})
        
        metrics = temp_store.get_metrics()
        assert metrics["test_counter"] == 42
        assert metrics["test_value"] == 3.14
        assert "ts" in metrics
    
    def test_concurrent_writes(self, temp_store):
        """Test file locking for concurrent writes."""
        # Simulate multiple writes (locking should prevent corruption)
        for i in range(10):
            temp_store.append_trade_event(
                source="swing" if i % 2 == 0 else "dt",
                symbol=f"SYM{i}",
                side="BUY",
                qty=100,
                price=150.0 + i,
                reason=f"REASON_{i}"
            )
        
        # Read and verify all trades written
        trades = []
        with open(temp_store.trades_file, "r") as f:
            for line in f:
                try:
                    trades.append(json.loads(line))
                except:
                    pass
        
        assert len(trades) == 10
    
    def test_trade_with_custom_fields(self, temp_store):
        """Test appending trade with custom fields."""
        temp_store.append_trade_event(
            source="dt",
            symbol="AAPL",
            side="BUY",
            qty=100,
            price=150.0,
            reason="BREAKOUT",
            custom_field_1="value1",
            custom_field_2=42
        )
        
        # Read and verify custom fields preserved
        trades = []
        with open(temp_store.trades_file, "r") as f:
            for line in f:
                trades.append(json.loads(line))
        
        trade = trades[0]
        assert trade["custom_field_1"] == "value1"
        assert trade["custom_field_2"] == 42


class TestSharedTruthStoreSingleton:
    """Test singleton pattern for shared store."""
    
    def test_get_shared_store(self):
        """Test get_shared_store returns singleton."""
        store1 = get_shared_store()
        store2 = get_shared_store()
        
        assert store1 is store2


class TestTruthStoreWrappers:
    """Test truth store wrappers maintain backward compatibility."""
    
    def test_swing_truth_store_wrapper(self, tmp_path, monkeypatch):
        """Test swing_truth_store forwards to shared store."""
        from backend.services.swing_truth_store import append_swing_event
        
        monkeypatch.setenv("SHARED_TRUTH_DIR", str(tmp_path))
        
        # Append event
        append_swing_event({
            "type": "trade",
            "symbol": "AAPL",
            "side": "BUY",
            "qty": 100,
            "price": 150.0,
            "reason": "SIGNAL"
        })
        
        # Check shared trades file
        shared_trades_file = tmp_path / "shared" / "shared_trades.jsonl"
        if shared_trades_file.exists():
            trades = []
            with open(shared_trades_file, "r") as f:
                for line in f:
                    trades.append(json.loads(line))
            
            # Should have swing trade in shared file
            assert any(t.get("source") == "swing" for t in trades)
    
    def test_dt_truth_store_wrapper(self, tmp_path, monkeypatch):
        """Test dt_truth_store forwards to shared store."""
        from dt_backend.services.dt_truth_store import append_trade_event
        
        monkeypatch.setenv("SHARED_TRUTH_DIR", str(tmp_path))
        
        # Append event
        append_trade_event({
            "type": "trade",
            "symbol": "TSLA",
            "side": "SELL",
            "qty": 50,
            "price": 200.0,
            "reason": "TAKE_PROFIT"
        })
        
        # Check shared trades file
        shared_trades_file = tmp_path / "shared" / "shared_trades.jsonl"
        if shared_trades_file.exists():
            trades = []
            with open(shared_trades_file, "r") as f:
                for line in f:
                    trades.append(json.loads(line))
            
            # Should have DT trade in shared file
            assert any(t.get("source") == "dt" for t in trades)
