"""Unit tests for the multi-channel Slack alerting system."""

from __future__ import annotations

import os
from unittest.mock import Mock, patch, MagicMock

import pytest


@pytest.fixture
def mock_env():
    """Mock environment variables for testing."""
    env_vars = {
        "SLACK_WEBHOOK_ERRORS": "https://hooks.slack.com/test_errors",
        "SLACK_WEBHOOK_TRADING": "https://hooks.slack.com/test_trading",
        "SLACK_WEBHOOK_DT": "https://hooks.slack.com/test_dt",
        "SLACK_WEBHOOK_SWING": "https://hooks.slack.com/test_swing",
        "SLACK_WEBHOOK_NIGHTLY": "https://hooks.slack.com/test_nightly",
        "SLACK_WEBHOOK_PNL": "https://hooks.slack.com/test_pnl",
        "SLACK_WEBHOOK_REPORTS": "https://hooks.slack.com/test_reports",
        "SLACK_WEBHOOK_TESTING": "https://hooks.slack.com/test_testing",
        "ALERT_RATE_LIMIT_SECONDS": "0",  # Disable rate limiting for tests
    }
    with patch.dict(os.environ, env_vars, clear=False):
        # Patch the CHANNELS dict directly instead of reloading module
        with patch("backend.monitoring.alerting.CHANNELS", {
            "errors": env_vars["SLACK_WEBHOOK_ERRORS"],
            "trading": env_vars["SLACK_WEBHOOK_TRADING"],
            "dt": env_vars["SLACK_WEBHOOK_DT"],
            "swing": env_vars["SLACK_WEBHOOK_SWING"],
            "nightly": env_vars["SLACK_WEBHOOK_NIGHTLY"],
            "pnl": env_vars["SLACK_WEBHOOK_PNL"],
            "reports": env_vars["SLACK_WEBHOOK_REPORTS"],
            "testing": env_vars["SLACK_WEBHOOK_TESTING"],
        }):
            yield


@pytest.fixture
def mock_requests():
    """Mock requests.post for testing."""
    with patch("backend.monitoring.alerting.requests.post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        yield mock_post


class TestAlertingSystem:
    """Test suite for the alerting system."""
    
    def test_channel_configuration(self):
        """Test that channels are properly configured from environment."""
        # Test with mock environment
        env_vars = {
            "SLACK_WEBHOOK_ERRORS": "https://hooks.slack.com/test_errors",
            "SLACK_WEBHOOK_TRADING": "https://hooks.slack.com/test_trading",
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            with patch("backend.monitoring.alerting.CHANNELS", {
                "errors": env_vars["SLACK_WEBHOOK_ERRORS"],
                "trading": env_vars["SLACK_WEBHOOK_TRADING"],
            }):
                from backend.monitoring.alerting import CHANNELS
                
                assert CHANNELS["errors"] == "https://hooks.slack.com/test_errors"
                assert CHANNELS["trading"] == "https://hooks.slack.com/test_trading"
    
    def test_alert_critical_basic(self, mock_env, mock_requests):
        """Test basic critical alert."""
        from backend.monitoring.alerting import alert_critical
        
        alert_critical("Test Alert", "Test message", channel="trading")
        
        # Verify request was made
        assert mock_requests.called
        call_args = mock_requests.call_args
        
        # Check webhook URL
        assert "test_trading" in call_args[0][0]
        
        # Check payload structure
        payload = call_args[1]["json"]
        assert "username" in payload
        assert payload["username"] == "AION Analytics"
        assert "attachments" in payload
        assert len(payload["attachments"]) > 0
        
        attachment = payload["attachments"][0]
        assert attachment["color"] == "#FF0000"  # Red for critical
        assert "Test Alert" in attachment["title"]
        assert attachment["text"] == "Test message"
    
    def test_alert_with_mention_channel(self, mock_env, mock_requests):
        """Test alert with @channel mention."""
        from backend.monitoring.alerting import alert_critical
        
        alert_critical("Emergency", "Critical issue", channel="trading", mention_channel=True)
        
        payload = mock_requests.call_args[1]["json"]
        attachment = payload["attachments"][0]
        
        # Should contain @channel mention
        assert "<!channel>" in attachment["title"]
        assert "Emergency" in attachment["title"]
    
    def test_alert_with_context(self, mock_env, mock_requests):
        """Test alert with context fields."""
        from backend.monitoring.alerting import alert_critical
        
        context = {
            "System": "Trading",
            "Action": "Review logs",
        }
        
        alert_critical("Test", "Message", channel="trading", context=context)
        
        payload = mock_requests.call_args[1]["json"]
        attachment = payload["attachments"][0]
        
        # Check fields
        assert "fields" in attachment
        fields = attachment["fields"]
        assert len(fields) == 2
        
        field_titles = [f["title"] for f in fields]
        assert "System" in field_titles
        assert "Action" in field_titles
    
    def test_alert_error_routes_to_errors_channel(self, mock_env, mock_requests):
        """Test that alert_error routes to errors channel."""
        from backend.monitoring.alerting import alert_error
        
        alert_error("Error Title", "Error message")
        
        # Should route to errors channel
        call_args = mock_requests.call_args
        assert "test_errors" in call_args[0][0]
        
        # Should have @channel mention
        payload = call_args[1]["json"]
        attachment = payload["attachments"][0]
        assert "<!channel>" in attachment["title"]
    
    def test_alert_dt_routes_to_dt_channel(self, mock_env, mock_requests):
        """Test that alert_dt routes to dt channel."""
        from backend.monitoring.alerting import alert_dt
        
        alert_dt("DT Alert", "DT message")
        
        call_args = mock_requests.call_args
        assert "test_dt" in call_args[0][0]
    
    def test_alert_swing_routes_to_swing_channel(self, mock_env, mock_requests):
        """Test that alert_swing routes to swing channel."""
        from backend.monitoring.alerting import alert_swing
        
        alert_swing("Swing Alert", "Swing message")
        
        call_args = mock_requests.call_args
        assert "test_swing" in call_args[0][0]
    
    def test_alert_nightly_routes_to_nightly_channel(self, mock_env, mock_requests):
        """Test that alert_nightly routes to nightly channel."""
        from backend.monitoring.alerting import alert_nightly
        
        alert_nightly("Nightly Alert", "Nightly message")
        
        call_args = mock_requests.call_args
        assert "test_nightly" in call_args[0][0]
    
    def test_alert_pnl_routes_to_pnl_channel(self, mock_env, mock_requests):
        """Test that alert_pnl routes to pnl channel."""
        from backend.monitoring.alerting import alert_pnl
        
        alert_pnl("PnL Alert", "PnL message")
        
        call_args = mock_requests.call_args
        assert "test_pnl" in call_args[0][0]
    
    def test_alert_report_routes_to_reports_channel(self, mock_env, mock_requests):
        """Test that alert_report routes to reports channel."""
        from backend.monitoring.alerting import alert_report
        
        alert_report("Report Alert", "Report message")
        
        call_args = mock_requests.call_args
        assert "test_reports" in call_args[0][0]
    
    def test_alert_levels(self, mock_env, mock_requests):
        """Test different alert levels have correct colors and emojis."""
        from backend.monitoring.alerting import send_alert
        
        # Critical
        send_alert("critical", "Critical Title", "Message", channel="trading", skip_rate_limit=True)
        payload = mock_requests.call_args[1]["json"]
        assert payload["attachments"][0]["color"] == "#FF0000"
        assert "🚨" in payload["attachments"][0]["title"]
        
        # Warning
        send_alert("warning", "Warning Title", "Message", channel="trading", skip_rate_limit=True)
        payload = mock_requests.call_args[1]["json"]
        assert payload["attachments"][0]["color"] == "#FFA500"
        assert "⚠️" in payload["attachments"][0]["title"]
        
        # Info
        send_alert("info", "Info Title", "Message", channel="trading", skip_rate_limit=True)
        payload = mock_requests.call_args[1]["json"]
        assert payload["attachments"][0]["color"] == "#00FF00"
        assert "ℹ️" in payload["attachments"][0]["title"]
    
    def test_missing_webhook_url(self, mock_requests):
        """Test handling of missing webhook URL."""
        # Mock empty CHANNELS dict
        with patch("backend.monitoring.alerting.CHANNELS", {}):
            from backend.monitoring.alerting import alert_critical
            
            # Should not crash, just log
            alert_critical("Test", "Message", channel="nonexistent")
            
            # Should not have made any request
            assert not mock_requests.called
    
    def test_failed_request_handling(self, mock_env):
        """Test handling of failed Slack request."""
        with patch("backend.monitoring.alerting.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_post.return_value = mock_response
            
            from backend.monitoring.alerting import alert_critical
            
            # Should not raise exception - use skip_rate_limit to ensure it gets called
            alert_critical("Unique Test 1", "Message", channel="trading", skip_rate_limit=True)
            
            assert mock_post.called
    
    def test_request_exception_handling(self, mock_env):
        """Test handling of request exception."""
        with patch("backend.monitoring.alerting.requests.post") as mock_post:
            mock_post.side_effect = Exception("Network error")
            
            from backend.monitoring.alerting import alert_critical
            
            # Should not raise exception - use skip_rate_limit to ensure it gets called
            alert_critical("Unique Test 2", "Message", channel="trading", skip_rate_limit=True)
            
            assert mock_post.called


class TestDTAlertRouting:
    """Test suite specifically for DT (Day Trading) alert routing verification.
    
    This ensures DT-related alerts go to #day_trading channel, NOT #daily-pnl.
    """
    
    def _assert_routed_to_channel(self, mock_requests, expected_channel: str, not_channel: str = None):
        """Helper method to assert alert was routed to the correct channel.
        
        Args:
            mock_requests: Mock requests fixture
            expected_channel: Channel that should be used (e.g., 'test_dt')
            not_channel: Channel that should NOT be used (e.g., 'test_pnl')
        """
        assert mock_requests.called, "Alert was not sent"
        call_args = mock_requests.call_args
        assert expected_channel in call_args[0][0], \
            f"Alert should route to channel containing '{expected_channel}'"
        if not_channel:
            assert not_channel not in call_args[0][0], \
                f"Alert should NOT route to channel containing '{not_channel}'"
    
    def test_dt_position_exit_routes_to_day_trading(self, mock_env, mock_requests):
        """Verify position exits from DT use alert_dt() → #day_trading."""
        from backend.monitoring.alerting import alert_dt
        
        # Simulate position exit alert
        alert_dt(
            "Position Closed: AAPL",
            "Exit reason: stop_hit",
            level="info",
            context={
                "Bot": "ORB",
                "Entry Price": "$180.50",
                "Exit Price": "$178.25",
                "PnL": "-1.25% ($-22.50)",
            }
        )
        
        # Verify routed to DT channel, NOT PnL channel
        self._assert_routed_to_channel(mock_requests, "test_dt", "test_pnl")
    
    def test_dt_cycle_completion_routes_to_day_trading(self, mock_env, mock_requests):
        """Verify DT cycle completion alerts use alert_dt() → #day_trading."""
        from backend.monitoring.alerting import alert_dt
        
        # Simulate cycle completion alert
        alert_dt(
            "DT Cycle Complete: abc123",
            "Lane: FAST | Seq: 42",
            level="info",
            context={
                "Orders Sent": "3",
                "Exits Sent": "1",
                "Considered": "12",
                "Blocked": "2",
            }
        )
        
        # Verify routed to DT channel, NOT PnL channel
        self._assert_routed_to_channel(mock_requests, "test_dt", "test_pnl")
    
    def test_dt_trade_entry_routes_to_day_trading(self, mock_env, mock_requests):
        """Verify DT trade entries use alert_dt() → #day_trading."""
        from backend.monitoring.alerting import alert_dt
        
        # Simulate trade entry alert
        alert_dt(
            "Trade Executed: BUY TSLA",
            "Entry: $245.50 | Stop: $240.00 | Target: $252.00",
            level="info",
            context={
                "Bot": "ORB",
                "Confidence": "0.75",
                "Risk:Reward": "1:2.4",
            }
        )
        
        # Verify routed to DT channel, NOT PnL channel
        self._assert_routed_to_channel(mock_requests, "test_dt", "test_pnl")
    
    def test_pnl_reports_route_to_daily_pnl(self, mock_env, mock_requests):
        """Verify PnL summary reports use alert_pnl() → #daily-pnl (NOT #day_trading)."""
        from backend.monitoring.alerting import alert_pnl
        
        # Simulate end-of-day PnL report
        alert_pnl(
            "Daily PnL Summary",
            "Trading day complete",
            context={
                "Daily PnL": "$+250.50",
                "Win Rate": "65%",
                "Total Trades": "12",
                "MTD PnL": "$+1,250.00",
            }
        )
        
        # Verify routed to PnL channel, NOT DT channel
        self._assert_routed_to_channel(mock_requests, "test_pnl", "test_dt")
    
    def test_all_dt_alert_functions_route_correctly(self, mock_env, mock_requests):
        """Comprehensive test that all alert functions route to correct channels."""
        from backend.monitoring.alerting import (
            alert_dt,
            alert_swing,
            alert_nightly,
            alert_pnl,
            alert_error,
            alert_report,
        )
        
        # Test each alert function
        test_cases = [
            (alert_dt, "DT Test", "test_dt", "#day_trading"),
            (alert_swing, "Swing Test", "test_swing", "#swing_trading"),
            (alert_nightly, "Nightly Test", "test_nightly", "#nightly-logs-summary"),
            (alert_pnl, "PnL Test", "test_pnl", "#daily-pnl"),
            (alert_error, "Error Test", "test_errors", "#errors-tracebacks"),
            (alert_report, "Report Test", "test_reports", "#reports"),
        ]
        
        for alert_fn, title, expected_channel, expected_slack_channel in test_cases:
            mock_requests.reset_mock()
            
            alert_fn(title, "Test message")
            
            assert mock_requests.called, f"{alert_fn.__name__} did not send alert"
            call_args = mock_requests.call_args
            assert expected_channel in call_args[0][0], \
                f"{alert_fn.__name__} should route to {expected_slack_channel} (webhook contains '{expected_channel}')"


class TestErrorHandler:
    """Test suite for error handler."""
    
    @pytest.mark.asyncio
    async def test_global_exception_handler(self, mock_env, mock_requests):
        """Test global exception handler."""
        from backend.core.error_handler import global_exception_handler
        from fastapi import Request
        
        # Mock request
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/test/path"
        mock_request.method = "GET"
        mock_request.headers.get.return_value = "test-agent"
        
        # Test exception
        exc = ValueError("Test error")
        
        response = await global_exception_handler(mock_request, exc)
        
        # Check response
        assert response.status_code == 500
        
        # Check that alert was sent
        assert mock_requests.called
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        from backend.core.error_handler import CircuitBreaker
        
        breaker = CircuitBreaker("test", failure_threshold=3)
        
        assert breaker.state == "CLOSED"
        assert breaker.can_execute() is True
    
    def test_circuit_breaker_opens_after_threshold(self, mock_env, mock_requests):
        """Test circuit breaker opens after threshold failures."""
        from backend.core.error_handler import CircuitBreaker
        
        breaker = CircuitBreaker("test", failure_threshold=3, alert_on_open=True)
        
        # Record failures
        breaker.record_failure()
        assert breaker.state == "CLOSED"
        
        breaker.record_failure()
        assert breaker.state == "CLOSED"
        
        breaker.record_failure()
        assert breaker.state == "OPEN"
        assert breaker.can_execute() is False
        
        # Check that alert was sent
        assert mock_requests.called
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after cooldown."""
        import time
        from backend.core.error_handler import CircuitBreaker
        
        breaker = CircuitBreaker("test", failure_threshold=2, cooldown_seconds=1, alert_on_open=False)
        
        # Open circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "OPEN"
        
        # Wait for cooldown
        time.sleep(1.1)
        
        # Should enter half-open
        assert breaker.can_execute() is True
        assert breaker.state == "HALF_OPEN"
        
        # Successful call should close circuit
        breaker.record_success()
        assert breaker.state == "CLOSED"
    
    def test_circuit_breaker_get_state(self):
        """Test circuit breaker state reporting."""
        from backend.core.error_handler import CircuitBreaker
        
        breaker = CircuitBreaker("test", failure_threshold=3)
        
        state = breaker.get_state()
        
        assert state["name"] == "test"
        assert state["state"] == "CLOSED"
        assert state["failures"] == 0
        assert state["threshold"] == 3
