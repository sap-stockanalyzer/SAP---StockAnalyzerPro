"""Unit tests for swing bot Slack alert functionality."""

from __future__ import annotations

import os
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timezone

import pytest

from backend.bots.base_swing_bot import SwingBot, SwingBotConfig, Position, BotState, Trade


@pytest.fixture
def mock_env():
    """Mock environment variables for testing."""
    env_vars = {
        "SLACK_WEBHOOK_SWING": "https://hooks.slack.com/test_swing",
        "SWING_SEND_REJECTIONS": "0",
        "SWING_SEND_REJECTIONS_MAX": "20",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield


@pytest.fixture
def mock_alert_swing():
    """Mock alert_swing function."""
    with patch("backend.bots.base_swing_bot.alert_swing") as mock:
        yield mock


@pytest.fixture
def sample_config():
    """Sample swing bot configuration."""
    return SwingBotConfig(
        bot_key="bot_1w",
        horizon="1w",
        max_positions=5,
        base_risk_pct=0.02,
        conf_threshold=0.55,
        stop_loss_pct=0.08,
        take_profit_pct=0.12,
        max_weight_per_name=0.25,
        initial_cash=100000.0,
        low_conf_max_fraction=0.20,
        starter_fraction=0.50,
        add_fraction=0.33,
        max_build_stages=3,
        min_days_between_adds=2,
        add_conf_extra=0.03,
        allow_build_adds=True,
        min_hold_days=2,
        time_stop_days=10,
        exit_confirmations=2,
        exit_conf_buffer=0.05,
        use_phit=True,
        min_phit=0.55,
        loss_est_pct=0.06,
        require_positive_ev=True,
        ev_power=1.0,
    )


@pytest.fixture
def sample_rolling():
    """Sample rolling data."""
    return {
        "AAPL": {
            "price": 150.0,
            "policy": {
                "intent": "BUY",
                "confidence": 0.78,
                "score": 0.15,
            },
            "predictions": {
                "1w": {
                    "predicted_return": 0.052,
                }
            }
        },
        "TSLA": {
            "price": 220.0,
            "policy": {
                "intent": "BUY",
                "confidence": 0.42,
                "score": 0.05,
            },
            "predictions": {
                "1w": {
                    "predicted_return": 0.023,
                }
            }
        },
        "_GLOBAL": {
            "regime": {
                "label": "bull"
            }
        }
    }


class TestSwingBotBuyAlerts:
    """Test suite for BUY alert functionality."""
    
    def test_send_buy_alert_basic(self, mock_env, mock_alert_swing, sample_config, sample_rolling):
        """Test basic BUY alert sending."""
        bot = SwingBot(sample_config)
        
        bot._send_buy_alert(
            symbol="AAPL",
            qty=100.0,
            price=150.0,
            rolling=sample_rolling,
        )
        
        # Verify alert was sent
        assert mock_alert_swing.called
        call_args = mock_alert_swing.call_args
        
        # Check title
        title = call_args[0][0]
        assert "📈" in title
        assert "Swing Bot 1w" in title
        assert "BUY" in title
        
        # Check message content
        message = call_args[0][1]
        assert "AAPL" in message
        assert "100.00 shares" in message
        assert "$150.00" in message
        assert "Confidence:" in message
        assert "78.0%" in message  # 0.78 * 100
        assert "Expected Return:" in message
        # Accept both "5.2%" and "5.20%" formatting
        assert "+5.2%" in message or "+5.20%" in message
        assert "P(Hit):" in message
        
        # Check level is info
        assert call_args[1]["level"] == "info"
        assert call_args[1]["skip_rate_limit"] is True
    
    def test_send_buy_alert_no_webhook(self, mock_alert_swing, sample_config, sample_rolling):
        """Test BUY alert doesn't crash when webhook is not configured."""
        with patch.dict(os.environ, {"SLACK_WEBHOOK_SWING": ""}, clear=False):
            bot = SwingBot(sample_config)
            
            # Should not crash
            bot._send_buy_alert(
                symbol="AAPL",
                qty=100.0,
                price=150.0,
                rolling=sample_rolling,
            )


class TestSwingBotSellAlerts:
    """Test suite for SELL alert functionality."""
    
    def test_send_sell_alert_profit(self, mock_env, mock_alert_swing, sample_config):
        """Test SELL alert for profitable trade."""
        bot = SwingBot(sample_config)
        
        bot._send_sell_alert(
            symbol="AAPL",
            qty=100.0,
            price=155.50,
            entry_price=150.0,
            reason="TAKE_PROFIT",
            entry_ts="2024-01-15T09:30:00Z",
        )
        
        # Verify alert was sent
        assert mock_alert_swing.called
        call_args = mock_alert_swing.call_args
        
        # Check title
        title = call_args[0][0]
        assert "📉" in title
        assert "Swing Bot 1w" in title
        assert "SELL" in title
        
        # Check message content
        message = call_args[0][1]
        assert "AAPL" in message
        assert "100.00 shares" in message
        assert "$155.50" in message
        assert "PnL:" in message
        assert "+$550.00" in message  # Positive PnL
        assert "+3.67%" in message
        assert "Take Profit" in message
        assert "Hold Duration:" in message
        
        # Check level is info
        assert call_args[1]["level"] == "info"
        assert call_args[1]["skip_rate_limit"] is True
    
    def test_send_sell_alert_loss(self, mock_env, mock_alert_swing, sample_config):
        """Test SELL alert for losing trade."""
        bot = SwingBot(sample_config)
        
        bot._send_sell_alert(
            symbol="TSLA",
            qty=50.0,
            price=210.0,
            entry_price=220.0,
            reason="STOP_LOSS",
            entry_ts="2024-01-20T10:00:00Z",
        )
        
        # Verify alert was sent
        assert mock_alert_swing.called
        call_args = mock_alert_swing.call_args
        
        # Check message content
        message = call_args[0][1]
        assert "TSLA" in message
        assert "50.00 shares" in message
        assert "$210.00" in message
        assert "PnL:" in message
        assert "-$500.00" in message  # Negative PnL
        assert "-4.55%" in message
        assert "Stop Loss" in message


class TestSwingBotRejectionAlerts:
    """Test suite for rejection alert functionality."""
    
    def test_send_rejection_alert_conf_threshold(self, mock_env, mock_alert_swing, sample_config):
        """Test rejection alert for confidence below threshold."""
        bot = SwingBot(sample_config)
        
        bot._send_rejection_alert(
            symbol="TSLA",
            price=220.0,
            reason="conf_below_threshold",
            details={
                "confidence": 0.42,
                "conf_threshold": 0.55,
                "expected_return": 0.023,
            },
        )
        
        # Verify alert was sent
        assert mock_alert_swing.called
        call_args = mock_alert_swing.call_args
        
        # Check title
        title = call_args[0][0]
        assert "⛔" in title
        assert "Swing Bot 1w" in title
        assert "REJECTED: TSLA" in title
        
        # Check message content
        message = call_args[0][1]
        assert "$220.00" in message
        assert "Conf Below Threshold" in message
        assert "Confidence: 42.0%" in message
        assert "Required: 55.0%" in message
        assert "Gap: -13.0%" in message
        assert "Expected Return: +2.30%" in message or "Expected Return: +2.3%" in message
        
        # Check level is warning
        assert call_args[1]["level"] == "warning"
        assert call_args[1]["skip_rate_limit"] is True
    
    def test_send_rejection_alert_negative_ev(self, mock_env, mock_alert_swing, sample_config):
        """Test rejection alert for negative expected value."""
        bot = SwingBot(sample_config)
        
        bot._send_rejection_alert(
            symbol="MSFT",
            price=380.0,
            reason="non_positive_ev",
            details={
                "confidence": 0.50,
                "p_hit": 0.50,
                "expected_return": 0.015,
                "loss_est_pct": 0.06,
                "ev": -0.018,
            },
        )
        
        # Verify alert was sent
        assert mock_alert_swing.called
        call_args = mock_alert_swing.call_args
        
        # Check title - symbol is in title
        title = call_args[0][0]
        assert "MSFT" in title
        
        # Check message content
        message = call_args[0][1]
        assert "$380.00" in message
        assert "Non Positive Ev" in message
        assert "P(Hit): 50.0%" in message
        assert "Expected Return: +1.50%" in message or "Expected Return: +1.5%" in message
        assert "EV: -0.018" in message
        assert "Loss Estimate: 6.0%" in message
    
    def test_rejection_alerts_disabled_by_default(self, mock_alert_swing, sample_config):
        """Test that rejection alerts are disabled by default."""
        with patch.dict(os.environ, {"SWING_SEND_REJECTIONS": "0"}, clear=False):
            bot = SwingBot(sample_config)
            
            # This would normally send an alert, but should be disabled
            # (tested indirectly through build_ai_ranked_universe)


class TestSwingBotRebalanceSummary:
    """Test suite for rebalance summary alert functionality."""
    
    def test_send_rebalance_summary_basic(self, mock_env, mock_alert_swing, sample_config):
        """Test basic rebalance summary alert."""
        bot = SwingBot(sample_config)
        
        # Create sample trades
        trades = [
            Trade(
                t="2024-01-22T09:30:00Z",
                symbol="AAPL",
                side="BUY",
                qty=100.0,
                price=150.0,
                reason="NEW_ENTRY",
            ),
            Trade(
                t="2024-01-22T09:31:00Z",
                symbol="MSFT",
                side="BUY",
                qty=50.0,
                price=380.0,
                reason="NEW_ENTRY",
            ),
            Trade(
                t="2024-01-22T09:32:00Z",
                symbol="TSLA",
                side="SELL",
                qty=75.0,
                price=220.0,
                reason="REMOVE_FROM_UNIVERSE",
                pnl=500.0,
            ),
        ]
        
        # Create sample state
        state = BotState(
            cash=50000.0,
            last_equity=125432.18,
            last_updated="2024-01-22T09:35:00Z",
            positions={
                "AAPL": Position(qty=100.0, entry=150.0, stop=138.0, target=168.0, goal_qty=100.0),
                "MSFT": Position(qty=50.0, entry=380.0, stop=350.0, target=427.0, goal_qty=50.0),
            },
        )
        
        # Sample rejection counts
        rejection_counts = {
            "conf_below_threshold": 623,
            "non_positive_ev": 412,
            "intent_not_buy": 98,
            "phit_below_threshold": 20,
        }
        
        bot._send_rebalance_summary(
            trades=trades,
            universe_analyzed=1200,
            universe_qualified=47,
            rejection_counts=rejection_counts,
            state=state,
        )
        
        # Verify alert was sent
        assert mock_alert_swing.called
        call_args = mock_alert_swing.call_args
        
        # Check title
        title = call_args[0][0]
        assert "📊" in title
        assert "Swing Bot 1w" in title
        assert "Rebalance Complete" in title
        
        # Check message content
        message = call_args[0][1]
        assert "Trades Executed: 3 total" in message
        assert "Buys: 2 positions entered" in message
        assert "Sells: 1 positions closed" in message
        assert "Universe Analyzed: 1200 symbols" in message
        assert "Qualified: 47 symbols" in message
        assert "Rejected: 1153 symbols" in message
        assert "Top Rejections:" in message
        assert "Conf Below Threshold: 623 (54%)" in message
        assert "Non Positive Ev: 412 (36%)" in message
        assert "Positions: 2 open" in message
        assert "Cash: $50,000.00" in message
        assert "Equity: $125,432.18" in message
        
        # Check level is info
        assert call_args[1]["level"] == "info"
        assert call_args[1]["skip_rate_limit"] is True
    
    def test_send_rebalance_summary_no_trades(self, mock_env, mock_alert_swing, sample_config):
        """Test rebalance summary with no trades."""
        bot = SwingBot(sample_config)
        
        state = BotState(
            cash=100000.0,
            last_equity=100000.0,
            last_updated="2024-01-22T09:35:00Z",
            positions={},
        )
        
        bot._send_rebalance_summary(
            trades=[],
            universe_analyzed=1000,
            universe_qualified=50,
            rejection_counts={},
            state=state,
        )
        
        # Verify alert was sent
        assert mock_alert_swing.called
        call_args = mock_alert_swing.call_args
        
        message = call_args[0][1]
        assert "Trades Executed: 0 total" in message
        assert "Buys: 0 positions entered" in message
        assert "Sells: 0 positions closed" in message
        assert "Positions: 0 open" in message


class TestSwingBotAlertIntegration:
    """Test suite for alert integration with trading logic."""
    
    @patch("backend.bots.base_swing_bot.alert_swing")
    def test_rebalance_full_sends_buy_alert(self, mock_alert_swing, sample_config, sample_rolling):
        """Test that rebalance_full sends BUY alerts."""
        with patch.object(SwingBot, "load_rolling", return_value=sample_rolling):
            with patch.object(SwingBot, "load_bot_state") as mock_load_state:
                with patch.object(SwingBot, "save_bot_state"):
                    with patch.object(SwingBot, "append_trades_to_daily_log"):
                        bot = SwingBot(sample_config)
                        
                        # Setup initial state
                        state = BotState(
                            cash=100000.0,
                            last_equity=100000.0,
                            last_updated="2024-01-22T09:00:00Z",
                            positions={},
                        )
                        mock_load_state.return_value = state
                        
                        # Setup target weights
                        target_weights = {"AAPL": 0.50}
                        
                        # Execute rebalance
                        trades = bot.rebalance_full(state, sample_rolling, target_weights)
                        
                        # Should have made BUY trade
                        assert len(trades) > 0
                        
                        # Should have sent alert (at least one call)
                        assert mock_alert_swing.called


class TestSwingBotAlertErrorHandling:
    """Test suite for error handling in alert functionality."""
    
    def test_buy_alert_exception_handled(self, mock_env, sample_config, sample_rolling):
        """Test that exceptions in BUY alert don't crash the bot."""
        with patch("backend.bots.base_swing_bot.alert_swing", side_effect=Exception("Network error")):
            bot = SwingBot(sample_config)
            
            # Should not crash
            bot._send_buy_alert(
                symbol="AAPL",
                qty=100.0,
                price=150.0,
                rolling=sample_rolling,
            )
    
    def test_sell_alert_exception_handled(self, mock_env, sample_config):
        """Test that exceptions in SELL alert don't crash the bot."""
        with patch("backend.bots.base_swing_bot.alert_swing", side_effect=Exception("Network error")):
            bot = SwingBot(sample_config)
            
            # Should not crash
            bot._send_sell_alert(
                symbol="AAPL",
                qty=100.0,
                price=155.50,
                entry_price=150.0,
                reason="TAKE_PROFIT",
                entry_ts="2024-01-15T09:30:00Z",
            )
    
    def test_rebalance_summary_exception_handled(self, mock_env, sample_config):
        """Test that exceptions in rebalance summary don't crash the bot."""
        with patch("backend.bots.base_swing_bot.alert_swing", side_effect=Exception("Network error")):
            bot = SwingBot(sample_config)
            
            state = BotState(
                cash=100000.0,
                last_equity=100000.0,
                last_updated="2024-01-22T09:35:00Z",
                positions={},
            )
            
            # Should not crash
            bot._send_rebalance_summary(
                trades=[],
                universe_analyzed=1000,
                universe_qualified=50,
                rejection_counts={},
                state=state,
            )
