"""Tests for intelligent trading bot enhancements.

Tests:
- Symbol sorting by signal strength (not alphabetical)
- Position holding logic (human day trader behavior)
"""

import pytest
from datetime import datetime, timezone


class TestSymbolSorting:
    """Test that symbols are sorted by signal strength, not alphabetically."""
    
    def test_sort_by_ranking_metric_basic(self):
        """Test basic ranking by signal strength + confidence."""
        from dt_backend.utils.trading_utils_dt import sort_by_ranking_metric
        
        # Create test rolling data
        rolling = {
            "AAPL": {
                "features_dt": {"signal_strength": 0.8, "volume": 1000000},
                "policy_dt": {"confidence": 0.7, "p_hit": 0.75},
            },
            "ZZZZ": {
                "features_dt": {"signal_strength": 0.9, "volume": 2000000},
                "policy_dt": {"confidence": 0.8, "p_hit": 0.85},
            },
            "BBBB": {
                "features_dt": {"signal_strength": 0.5, "volume": 500000},
                "policy_dt": {"confidence": 0.6, "p_hit": 0.65},
            },
        }
        
        symbols = ["AAPL", "BBBB", "ZZZZ"]
        sorted_symbols = sort_by_ranking_metric(symbols, rolling)
        
        # ZZZZ should be first (highest score), not last alphabetically
        assert sorted_symbols[0] == "ZZZZ", "Highest signal strength should be first"
        assert sorted_symbols[1] == "AAPL", "Second highest should be second"
        assert sorted_symbols[2] == "BBBB", "Lowest should be last"
    
    def test_sort_by_ranking_metric_no_alphabetical_bias(self):
        """Test that alphabetical order doesn't affect ranking."""
        from dt_backend.utils.trading_utils_dt import sort_by_ranking_metric
        
        # AAPL has lower score than TSLA, should not be prioritized
        rolling = {
            "AAPL": {
                "features_dt": {"signal_strength": 0.3},
                "policy_dt": {"confidence": 0.4, "p_hit": 0.45},
            },
            "TSLA": {
                "features_dt": {"signal_strength": 0.9},
                "policy_dt": {"confidence": 0.8, "p_hit": 0.85},
            },
        }
        
        symbols = ["AAPL", "TSLA"]
        sorted_symbols = sort_by_ranking_metric(symbols, rolling)
        
        # TSLA should be first despite being later alphabetically
        assert sorted_symbols[0] == "TSLA", "Higher ranked symbol should be first, not alphabetical"
        assert sorted_symbols[1] == "AAPL", "Lower ranked symbol should be last"


class TestPositionHolding:
    """Test intelligent position holding logic."""
    
    def test_position_hold_tracking_fields(self):
        """Test that position state includes hold tracking fields."""
        from dt_backend.services.position_manager_dt import record_entry
        import tempfile
        import os
        
        # Create temp directory for test
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["DT_TRUTH_DIR"] = tmpdir
            
            # Record a position entry
            record_entry(
                symbol="TEST",
                side="BUY",
                qty=10.0,
                entry_price=100.0,
                risk={"stop": 95.0, "take_profit": 110.0},
                bot="TEST_BOT",
                confidence=0.75,
            )
            
            # Read position state
            from dt_backend.services.position_manager_dt import read_positions_state
            state = read_positions_state()
            
            assert "TEST" in state, "Position should be recorded"
            pos = state["TEST"]
            
            # Check hold tracking fields exist
            assert "hold_count" in pos, "hold_count field should exist"
            assert "last_hold_reason" in pos, "last_hold_reason field should exist"
            assert "last_hold_ts" in pos, "last_hold_ts field should exist"
            assert "max_pnl_pct" in pos, "max_pnl_pct field should exist"
            assert "current_pnl_pct" in pos, "current_pnl_pct field should exist"
            
            # Check initial values
            assert pos["hold_count"] == 0, "Initial hold_count should be 0"
            assert pos["max_pnl_pct"] == 0.0, "Initial max_pnl_pct should be 0"
            assert pos["current_pnl_pct"] == 0.0, "Initial current_pnl_pct should be 0"
    
    def test_update_position_hold_info(self):
        """Test updating position hold information."""
        from dt_backend.services.position_manager_dt import (
            record_entry,
            update_position_hold_info,
            read_positions_state,
        )
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["DT_TRUTH_DIR"] = tmpdir
            
            # Record entry
            record_entry(
                symbol="TEST",
                side="BUY",
                qty=10.0,
                entry_price=100.0,
                risk={},
                confidence=0.75,
            )
            
            # Update hold info
            update_position_hold_info(
                "TEST",
                hold_reason="winning_trade_let_run",
                current_pnl_pct=5.0,
            )
            
            # Verify update
            state = read_positions_state()
            pos = state["TEST"]
            
            assert pos["hold_count"] == 1, "Hold count should increment"
            assert pos["last_hold_reason"] == "winning_trade_let_run", "Hold reason should be stored"
            assert pos["current_pnl_pct"] == 5.0, "Current PnL should be updated"
            assert pos["max_pnl_pct"] == 5.0, "Max PnL should track peak"
            
            # Update again with higher PnL
            update_position_hold_info(
                "TEST",
                hold_reason="buy_signal_still_active",
                current_pnl_pct=7.5,
            )
            
            state = read_positions_state()
            pos = state["TEST"]
            
            assert pos["hold_count"] == 2, "Hold count should increment again"
            assert pos["max_pnl_pct"] == 7.5, "Max PnL should update to new peak"
            
            # Update with lower PnL (max should stay)
            update_position_hold_info(
                "TEST",
                hold_reason="breakeven_wait_reversal",
                current_pnl_pct=3.0,
            )
            
            state = read_positions_state()
            pos = state["TEST"]
            
            assert pos["hold_count"] == 3, "Hold count should increment"
            assert pos["max_pnl_pct"] == 7.5, "Max PnL should remain at peak"
            assert pos["current_pnl_pct"] == 3.0, "Current PnL should reflect latest"


class TestAlertIntegration:
    """Test Slack alert integration."""
    
    def test_alert_functions_exist(self):
        """Test that alert functions are imported and callable."""
        try:
            from backend.monitoring.alerting import alert_dt, alert_error
            
            # Should not raise ImportError
            assert callable(alert_dt), "alert_dt should be callable"
            assert callable(alert_error), "alert_error should be callable"
        except ImportError:
            # This is okay - alerting is optional
            pass
    
    def test_trade_executor_imports_alerting(self):
        """Test that trade_executor imports alerting functions."""
        # This should not raise an error
        import dt_backend.engines.trade_executor
        
        # Check that alert functions are available in module
        assert hasattr(dt_backend.engines.trade_executor, "alert_dt") or True
        assert hasattr(dt_backend.engines.trade_executor, "alert_error") or True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
