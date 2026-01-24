"""Unit tests for Slack-aware logger."""

import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from utils.logger_slack import SlackLogger, create_slack_logger


class TestSlackLogger:
    """Test SlackLogger functionality."""
    
    def test_init_default(self):
        """Test SlackLogger initialization with defaults."""
        logger = SlackLogger()
        
        assert logger.name == "aion"
        assert logger.source == "backend"
        assert logger.slack_enabled is True  # Default from env
        assert logger.dt_brain is None
    
    def test_init_custom(self):
        """Test SlackLogger initialization with custom values."""
        logger = SlackLogger(
            name="test_logger",
            source="dt",
            slack_enabled=False
        )
        
        assert logger.name == "test_logger"
        assert logger.source == "dt"
        assert logger.slack_enabled is False
    
    def test_init_slack_disabled_env(self, monkeypatch):
        """Test SlackLogger respects LOG_SLACK_ENABLED env var."""
        monkeypatch.setenv("LOG_SLACK_ENABLED", "0")
        
        logger = SlackLogger(slack_enabled=True)
        
        # Should be False due to env override
        assert logger.slack_enabled is False
    
    def test_get_aggregator_success(self):
        """Test _get_aggregator successfully loads aggregator."""
        with patch('backend.monitoring.log_aggregator.get_aggregator') as mock_get_aggregator:
            mock_aggregator = Mock()
            mock_get_aggregator.return_value = mock_aggregator
            
            logger = SlackLogger(slack_enabled=True)
            aggregator = logger._get_aggregator()
            
            assert aggregator is mock_aggregator
            assert logger._aggregator is mock_aggregator
    
    def test_get_aggregator_failure(self):
        """Test _get_aggregator handles import errors gracefully."""
        with patch('backend.monitoring.log_aggregator.get_aggregator', side_effect=Exception("Import error")):
            logger = SlackLogger(slack_enabled=True)
            aggregator = logger._get_aggregator()
            
            assert aggregator is None
            assert logger.slack_enabled is False  # Should disable Slack
    
    def test_get_aggregator_disabled(self):
        """Test _get_aggregator when Slack disabled."""
        logger = SlackLogger(slack_enabled=False)
        aggregator = logger._get_aggregator()
        
        assert aggregator is None
    
    def test_get_aggregator_caching(self):
        """Test _get_aggregator caches the aggregator instance."""
        logger = SlackLogger(slack_enabled=True)
        
        with patch('backend.monitoring.log_aggregator.get_aggregator') as mock_get:
            mock_aggregator = Mock()
            mock_get.return_value = mock_aggregator
            
            # First call should load
            agg1 = logger._get_aggregator()
            # Second call should use cache
            agg2 = logger._get_aggregator()
            
            assert agg1 is agg2
            assert mock_get.call_count == 1  # Only called once
    
    @patch.object(SlackLogger, '_get_aggregator')
    @patch.object(SlackLogger, '_write_log')
    def test_error_forwards_to_slack(self, mock_write, mock_get_agg):
        """Test error() forwards to Slack."""
        mock_aggregator = Mock()
        mock_get_agg.return_value = mock_aggregator
        
        logger = SlackLogger(name="test", slack_enabled=True)
        logger.error("Test error", key="value")
        
        # Should write log normally
        mock_write.assert_called_once()
        
        # Should forward to Slack (includes component name)
        mock_aggregator.forward_log.assert_called_once_with(
            "ERROR",
            "Test error",
            "test",
            {"key": "value"}
        )
    
    @patch.object(SlackLogger, '_get_aggregator')
    @patch.object(SlackLogger, '_write_log')
    def test_error_with_exception(self, mock_write, mock_get_agg):
        """Test error() with exception forwarding."""
        mock_aggregator = Mock()
        mock_get_agg.return_value = mock_aggregator
        
        logger = SlackLogger(name="test", slack_enabled=True)
        
        try:
            raise ValueError("Test exception")
        except Exception as e:
            logger.error("Error occurred", exc=e)
        
        # Should still forward to Slack
        mock_aggregator.forward_log.assert_called_once()
    
    @patch.object(SlackLogger, '_get_aggregator')
    @patch.object(SlackLogger, '_write_log')
    def test_warn_forwards_to_slack(self, mock_write, mock_get_agg, monkeypatch):
        """Test warn() forwards to Slack when not in errors-only mode."""
        monkeypatch.setenv("LOG_SLACK_ERRORS_ONLY", "0")
        
        mock_aggregator = Mock()
        mock_get_agg.return_value = mock_aggregator
        
        logger = SlackLogger(name="test", slack_enabled=True)
        logger.warn("Test warning", key="value")
        
        # Should write log normally
        mock_write.assert_called_once()
        
        # Should forward to Slack (includes component name)
        mock_aggregator.forward_log.assert_called_once_with(
            "WARNING",
            "Test warning",
            "test",
            {"key": "value"}
        )
    
    @patch.object(SlackLogger, '_get_aggregator')
    @patch.object(SlackLogger, '_write_log')
    def test_warn_skips_slack_errors_only(self, mock_write, mock_get_agg, monkeypatch):
        """Test warn() skips Slack in errors-only mode."""
        monkeypatch.setenv("LOG_SLACK_ERRORS_ONLY", "1")
        
        mock_aggregator = Mock()
        mock_get_agg.return_value = mock_aggregator
        
        logger = SlackLogger(name="test", slack_enabled=True)
        logger.warn("Test warning")
        
        # Should write log normally
        mock_write.assert_called_once()
        
        # Should NOT forward to Slack
        mock_aggregator.forward_log.assert_not_called()
    
    @patch.object(SlackLogger, '_get_aggregator')
    @patch.object(SlackLogger, '_should_log')
    @patch.object(SlackLogger, '_write_log')
    def test_info_forwards_to_slack(self, mock_write, mock_should_log, mock_get_agg, monkeypatch):
        """Test info() forwards to Slack when not in errors-only mode."""
        monkeypatch.setenv("LOG_SLACK_ERRORS_ONLY", "0")
        mock_should_log.return_value = True  # Enable logging
        
        mock_aggregator = Mock()
        mock_get_agg.return_value = mock_aggregator
        
        logger = SlackLogger(name="test", slack_enabled=True)
        logger.info("Test info", key="value")
        
        # Should write log normally
        mock_write.assert_called_once()
        
        # Should forward to Slack
        mock_aggregator.forward_log.assert_called_once()
    
    @patch.object(SlackLogger, '_get_aggregator')
    @patch.object(SlackLogger, '_should_log')
    @patch.object(SlackLogger, '_write_log')
    def test_info_skips_slack_errors_only(self, mock_write, mock_should_log, mock_get_agg, monkeypatch):
        """Test info() skips Slack in errors-only mode."""
        monkeypatch.setenv("LOG_SLACK_ERRORS_ONLY", "1")
        mock_should_log.return_value = True  # Enable logging
        
        mock_aggregator = Mock()
        mock_get_agg.return_value = mock_aggregator
        
        logger = SlackLogger(name="test", slack_enabled=True)
        logger.info("Test info")
        
        # Should write log normally
        mock_write.assert_called_once()
        
        # Should NOT forward to Slack
        mock_aggregator.forward_log.assert_not_called()
    
    @patch.object(SlackLogger, '_get_aggregator')
    @patch.object(SlackLogger, '_should_log')
    @patch.object(SlackLogger, '_write_log')
    def test_debug_never_forwards_to_slack(self, mock_write, mock_should_log, mock_get_agg):
        """Test debug() never forwards to Slack."""
        mock_should_log.return_value = True  # Enable logging
        mock_aggregator = Mock()
        mock_get_agg.return_value = mock_aggregator
        
        logger = SlackLogger(name="test", slack_enabled=True)
        logger.debug("Test debug")
        
        # Should write log normally
        mock_write.assert_called_once()
        
        # Should NOT forward to Slack
        mock_aggregator.forward_log.assert_not_called()
    
    @patch.object(SlackLogger, '_should_log')
    @patch.object(SlackLogger, '_write_log')
    def test_slack_disabled_no_forwarding(self, mock_write, mock_should_log):
        """Test that Slack disabled prevents any forwarding."""
        mock_should_log.return_value = True  # Enable logging at log level
        logger = SlackLogger(name="test", slack_enabled=False)
        
        logger.error("Test error")
        logger.warn("Test warning")
        # Don't call info - it might be filtered by log level
        
        # Should write logs normally (at least 2 calls for error and warn)
        assert mock_write.call_count >= 2
        
        # But _get_aggregator should never be called
        assert logger._aggregator is None
    
    @patch.object(SlackLogger, '_get_aggregator')
    @patch.object(SlackLogger, '_write_log')
    def test_slack_forward_exception_silently_fails(self, mock_write, mock_get_agg):
        """Test that Slack forwarding exceptions don't break logging."""
        mock_aggregator = Mock()
        mock_aggregator.forward_log.side_effect = Exception("Slack error")
        mock_get_agg.return_value = mock_aggregator
        
        logger = SlackLogger(name="test", slack_enabled=True)
        
        # Should not raise exception
        logger.error("Test error")
        
        # Should still write log normally
        mock_write.assert_called_once()
    
    def test_inherits_from_logger(self):
        """Test SlackLogger inherits from base Logger."""
        from utils.logger import Logger
        
        logger = SlackLogger()
        assert isinstance(logger, Logger)
    
    def test_all_logger_methods_available(self):
        """Test SlackLogger has all expected methods."""
        logger = SlackLogger()
        
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'log')  # Alias
        assert hasattr(logger, 'warn')
        assert hasattr(logger, 'warning')  # Alias
        assert hasattr(logger, 'error')
        assert callable(logger.debug)
        assert callable(logger.error)


class TestCreateSlackLogger:
    """Test create_slack_logger factory function."""
    
    def test_creates_slack_logger(self):
        """Test create_slack_logger creates SlackLogger instance."""
        logger = create_slack_logger("test_component")
        
        assert isinstance(logger, SlackLogger)
        assert logger.name == "test_component"
        assert logger.source == "backend"
    
    def test_creates_with_custom_source(self):
        """Test create_slack_logger with custom source."""
        logger = create_slack_logger("test", source="dt")
        
        assert logger.name == "test"
        assert logger.source == "dt"
    
    def test_creates_with_slack_disabled(self):
        """Test create_slack_logger with Slack disabled."""
        logger = create_slack_logger("test", slack_enabled=False)
        
        assert logger.slack_enabled is False
