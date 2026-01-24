"""Unit tests for log aggregator."""

import os
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

import pytest

from backend.monitoring.log_aggregator import (
    LogAggregator,
    LogEntry,
    TestResult,
    get_aggregator,
)


class TestLogAggregator:
    """Test LogAggregator functionality."""
    
    def test_init_default_config(self):
        """Test LogAggregator initialization with default config."""
        aggregator = LogAggregator()
        
        assert aggregator.enabled is True  # Default from env
        assert aggregator.buffer_size == 10  # Default
        assert aggregator.buffer_timeout_sec == 60  # Default
        assert len(aggregator.channels) == 6
        assert "error" in aggregator.channels
        assert "testing" in aggregator.channels
    
    def test_init_custom_config(self, monkeypatch):
        """Test LogAggregator initialization with custom config."""
        monkeypatch.setenv("LOG_SLACK_ENABLED", "0")
        monkeypatch.setenv("LOG_BUFFER_SIZE", "20")
        monkeypatch.setenv("LOG_BUFFER_TIMEOUT_SEC", "30")
        
        aggregator = LogAggregator()
        
        assert aggregator.enabled is False
        assert aggregator.buffer_size == 20
        assert aggregator.buffer_timeout_sec == 30
    
    def test_should_forward_enabled(self):
        """Test _should_forward when enabled."""
        aggregator = LogAggregator()
        aggregator.enabled = True
        aggregator.errors_only = False
        
        assert aggregator._should_forward("ERROR") is True
        assert aggregator._should_forward("WARNING") is True
        assert aggregator._should_forward("INFO") is True
    
    def test_should_forward_errors_only(self):
        """Test _should_forward with errors_only mode."""
        aggregator = LogAggregator()
        aggregator.enabled = True
        aggregator.errors_only = True
        
        assert aggregator._should_forward("ERROR") is True
        assert aggregator._should_forward("CRITICAL") is True
        assert aggregator._should_forward("WARNING") is False
        assert aggregator._should_forward("INFO") is False
    
    def test_should_forward_disabled(self):
        """Test _should_forward when disabled."""
        aggregator = LogAggregator()
        aggregator.enabled = False
        
        assert aggregator._should_forward("ERROR") is False
        assert aggregator._should_forward("INFO") is False
    
    def test_get_webhook_url(self):
        """Test _get_webhook_url retrieves correct URL."""
        aggregator = LogAggregator()
        
        # Should return webhook URL or empty string
        url = aggregator._get_webhook_url("error")
        assert isinstance(url, str)
    
    def test_format_slack_message_basic(self):
        """Test _format_slack_message creates proper payload."""
        aggregator = LogAggregator()
        
        payload = aggregator._format_slack_message(
            "ERROR",
            "Test error message",
            "test_component"
        )
        
        assert "username" in payload
        assert "attachments" in payload
        assert len(payload["attachments"]) == 1
        
        attachment = payload["attachments"][0]
        assert attachment["color"] == "#FF0000"  # Red for ERROR
        assert "❌" in attachment["title"]  # Error emoji
        assert "test_component" in attachment["title"]
        assert attachment["text"] == "Test error message"
    
    def test_format_slack_message_with_context(self):
        """Test _format_slack_message with context."""
        aggregator = LogAggregator()
        
        context = {"symbol": "AAPL", "price": "150.00"}
        payload = aggregator._format_slack_message(
            "INFO",
            "Test message",
            "test_component",
            context
        )
        
        attachment = payload["attachments"][0]
        assert "fields" in attachment
        assert len(attachment["fields"]) == 2
        assert attachment["fields"][0]["title"] == "symbol"
        assert attachment["fields"][0]["value"] == "AAPL"
    
    def test_format_slack_message_truncates_long_message(self):
        """Test _format_slack_message truncates messages over 1000 chars."""
        aggregator = LogAggregator()
        
        long_message = "A" * 1500
        payload = aggregator._format_slack_message("INFO", long_message)
        
        attachment = payload["attachments"][0]
        assert len(attachment["text"]) == 1000  # 997 + "..."
        assert attachment["text"].endswith("...")
    
    @patch('backend.monitoring.log_aggregator.requests.post')
    def test_send_to_slack_success(self, mock_post):
        """Test successful Slack message send."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        aggregator = LogAggregator()
        aggregator.webhooks["testing"] = "https://hooks.slack.com/test"
        
        payload = {"test": "payload"}
        result = aggregator._send_to_slack("testing", payload)
        
        assert result is True
        mock_post.assert_called_once()
    
    @patch('backend.monitoring.log_aggregator.requests.post')
    def test_send_to_slack_failure(self, mock_post):
        """Test failed Slack message send."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        aggregator = LogAggregator()
        aggregator.webhooks["testing"] = "https://hooks.slack.com/test"
        
        payload = {"test": "payload"}
        result = aggregator._send_to_slack("testing", payload)
        
        assert result is False
    
    @patch('backend.monitoring.log_aggregator.requests.post')
    def test_send_to_slack_no_webhook(self, mock_post):
        """Test Slack send with no webhook configured."""
        aggregator = LogAggregator()
        aggregator.webhooks["testing"] = ""
        
        payload = {"test": "payload"}
        result = aggregator._send_to_slack("testing", payload)
        
        assert result is False
        mock_post.assert_not_called()
    
    @patch.object(LogAggregator, '_send_to_slack')
    def test_forward_log_error(self, mock_send):
        """Test forward_log for error level."""
        aggregator = LogAggregator()
        aggregator.enabled = True
        
        aggregator.forward_log("ERROR", "Test error", "test_component")
        
        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        assert args[0] == "error"  # Channel
    
    @patch.object(LogAggregator, '_send_to_slack')
    def test_forward_log_warning(self, mock_send):
        """Test forward_log for warning level."""
        aggregator = LogAggregator()
        aggregator.enabled = True
        
        aggregator.forward_log("WARNING", "Test warning", "test_component")
        
        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        assert args[0] == "warning"  # Channel
    
    @patch.object(LogAggregator, '_send_to_slack')
    def test_forward_log_disabled(self, mock_send):
        """Test forward_log when disabled."""
        aggregator = LogAggregator()
        aggregator.enabled = False
        
        aggregator.forward_log("ERROR", "Test error", "test_component")
        
        mock_send.assert_not_called()
    
    @patch.object(LogAggregator, '_send_to_slack')
    def test_forward_test_result_pass(self, mock_send, monkeypatch):
        """Test forward_test_result for passing test."""
        monkeypatch.setenv("TEST_INCLUDE_PASS", "1")
        
        aggregator = LogAggregator()
        aggregator.enabled = True
        
        result = TestResult(
            test_name="test_something",
            status="PASS",
            duration_ms=15.5
        )
        
        aggregator.forward_test_result(result)
        
        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        assert args[0] == "testing"  # Channel
    
    @patch.object(LogAggregator, '_send_to_slack')
    def test_forward_test_result_fail(self, mock_send, monkeypatch):
        """Test forward_test_result for failing test."""
        monkeypatch.setenv("TEST_INCLUDE_FAIL", "1")
        
        aggregator = LogAggregator()
        aggregator.enabled = True
        
        result = TestResult(
            test_name="test_something",
            status="FAIL",
            duration_ms=15.5,
            error_message="AssertionError: test failed"
        )
        
        aggregator.forward_test_result(result)
        
        mock_send.assert_called_once()
    
    @patch.object(LogAggregator, '_send_to_slack')
    def test_forward_test_result_skip_pass(self, mock_send, monkeypatch):
        """Test forward_test_result skips passing tests when configured."""
        monkeypatch.setenv("TEST_INCLUDE_PASS", "0")
        
        aggregator = LogAggregator()
        aggregator.enabled = True
        
        result = TestResult(
            test_name="test_something",
            status="PASS",
            duration_ms=15.5
        )
        
        aggregator.forward_test_result(result)
        
        mock_send.assert_not_called()
    
    @patch.object(LogAggregator, '_send_to_slack')
    def test_forward_trade(self, mock_send):
        """Test forward_trade."""
        aggregator = LogAggregator()
        aggregator.enabled = True
        
        trade_event = {
            "symbol": "AAPL",
            "action": "BUY",
            "confidence": 0.75,
            "size": 0.05,
            "phit": 0.68,
            "hold_time_check": True
        }
        
        aggregator.forward_trade(trade_event)
        
        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        assert args[0] == "trading"  # Channel
    
    @patch.object(LogAggregator, '_send_to_slack')
    def test_forward_health(self, mock_send):
        """Test forward_health."""
        aggregator = LogAggregator()
        aggregator.enabled = True
        
        health_data = {
            "cycles_total": 100,
            "cycles_completed": 98,
            "errors": 2,
            "warnings": 5,
            "uptime": "24h 15m",
            "last_cycle": "2m ago"
        }
        
        aggregator.forward_health(health_data)
        
        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        assert args[0] == "health"  # Channel
    
    def test_get_aggregator_singleton(self):
        """Test get_aggregator returns singleton instance."""
        agg1 = get_aggregator()
        agg2 = get_aggregator()
        
        assert agg1 is agg2
        assert isinstance(agg1, LogAggregator)


class TestLogEntry:
    """Test LogEntry dataclass."""
    
    def test_log_entry_creation(self):
        """Test LogEntry creation with required fields."""
        entry = LogEntry(
            level="ERROR",
            message="Test message",
            timestamp=datetime.now(timezone.utc)
        )
        
        assert entry.level == "ERROR"
        assert entry.message == "Test message"
        assert entry.component == ""
        assert isinstance(entry.context, dict)
        assert len(entry.context) == 0
    
    def test_log_entry_with_context(self):
        """Test LogEntry creation with context."""
        context = {"key": "value"}
        entry = LogEntry(
            level="INFO",
            message="Test",
            timestamp=datetime.now(timezone.utc),
            component="test_component",
            context=context
        )
        
        assert entry.component == "test_component"
        assert entry.context == context


class TestTestResult:
    """Test TestResult dataclass."""
    
    def test_test_result_creation(self):
        """Test TestResult creation."""
        result = TestResult(
            test_name="test_something",
            status="PASS",
            duration_ms=15.5
        )
        
        assert result.test_name == "test_something"
        assert result.status == "PASS"
        assert result.duration_ms == 15.5
        assert result.error_message is None
        assert isinstance(result.timestamp, datetime)
    
    def test_test_result_with_error(self):
        """Test TestResult with error message."""
        result = TestResult(
            test_name="test_failing",
            status="FAIL",
            duration_ms=10.0,
            error_message="Test failed"
        )
        
        assert result.status == "FAIL"
        assert result.error_message == "Test failed"
