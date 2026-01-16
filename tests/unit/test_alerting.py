"""Unit tests for the multi-channel Slack alerting system."""

from __future__ import annotations

import os
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_env():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        "SLACK_WEBHOOK_ERRORS": "https://hooks.slack.com/test_errors",
        "SLACK_WEBHOOK_TRADING": "https://hooks.slack.com/test_trading",
        "SLACK_WEBHOOK_DT": "https://hooks.slack.com/test_dt",
        "SLACK_WEBHOOK_SWING": "https://hooks.slack.com/test_swing",
        "SLACK_WEBHOOK_NIGHTLY": "https://hooks.slack.com/test_nightly",
        "SLACK_WEBHOOK_PNL": "https://hooks.slack.com/test_pnl",
        "SLACK_WEBHOOK_REPORTS": "https://hooks.slack.com/test_reports",
        "SLACK_WEBHOOK_TESTING": "https://hooks.slack.com/test_testing",
        "ALERT_RATE_LIMIT_SECONDS": "0",  # Disable rate limiting for tests
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
    
    def test_channel_configuration(self, mock_env):
        """Test that channels are properly configured from environment."""
        from backend.monitoring.alerting import CHANNELS
        
        # Need to reload module to pick up new env vars
        import importlib
        import backend.monitoring.alerting as alerting_module
        importlib.reload(alerting_module)
        
        channels = alerting_module.CHANNELS
        assert channels["errors"] == "https://hooks.slack.com/test_errors"
        assert channels["trading"] == "https://hooks.slack.com/test_trading"
        assert channels["dt"] == "https://hooks.slack.com/test_dt"
    
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
        send_alert("critical", "Title", "Message", channel="trading")
        payload = mock_requests.call_args[1]["json"]
        assert payload["attachments"][0]["color"] == "#FF0000"
        assert "🚨" in payload["attachments"][0]["title"]
        
        # Warning
        send_alert("warning", "Title", "Message", channel="trading")
        payload = mock_requests.call_args[1]["json"]
        assert payload["attachments"][0]["color"] == "#FFA500"
        assert "⚠️" in payload["attachments"][0]["title"]
        
        # Info
        send_alert("info", "Title", "Message", channel="trading")
        payload = mock_requests.call_args[1]["json"]
        assert payload["attachments"][0]["color"] == "#00FF00"
        assert "ℹ️" in payload["attachments"][0]["title"]
    
    def test_missing_webhook_url(self, mock_requests):
        """Test handling of missing webhook URL."""
        with patch.dict(os.environ, {}, clear=True):
            from backend.monitoring.alerting import alert_critical
            import importlib
            import backend.monitoring.alerting as alerting_module
            importlib.reload(alerting_module)
            
            # Should not crash, just log
            alerting_module.alert_critical("Test", "Message", channel="nonexistent")
            
            # Should not have made any request
            assert not mock_requests.called
    
    def test_failed_request_handling(self, mock_env):
        """Test handling of failed Slack request."""
        with patch("backend.monitoring.alerting.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_post.return_value = mock_response
            
            from backend.monitoring.alerting import alert_critical
            
            # Should not raise exception
            alert_critical("Test", "Message", channel="trading")
            
            assert mock_post.called
    
    def test_request_exception_handling(self, mock_env):
        """Test handling of request exception."""
        with patch("backend.monitoring.alerting.requests.post") as mock_post:
            mock_post.side_effect = Exception("Network error")
            
            from backend.monitoring.alerting import alert_critical
            
            # Should not raise exception
            alert_critical("Test", "Message", channel="trading")
            
            assert mock_post.called


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
