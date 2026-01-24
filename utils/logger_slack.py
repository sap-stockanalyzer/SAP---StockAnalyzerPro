"""Slack-aware logging enhancement for AION Analytics.

Extends the unified logger to automatically forward logs to Slack channels
based on log level and configuration.

Features:
- Automatic forwarding of errors to Slack
- Automatic forwarding of warnings to Slack
- Configurable filtering (errors only, all logs, etc.)
- Graceful degradation if Slack unavailable
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logger import Logger


class SlackLogger(Logger):
    """Logger that also sends messages to Slack.
    
    This class extends the unified Logger to automatically forward
    important logs to Slack channels via the log aggregator.
    """
    
    def __init__(
        self,
        name: str = "aion",
        source: str = "backend",
        dt_brain: Optional[Any] = None,
        log_dir: Optional[Path] = None,
        log_level: Optional[int] = None,
        slack_enabled: bool = True,
    ):
        """Initialize Slack-aware logger.
        
        Args:
            name: Component name
            source: System source (swing/dt/backend)
            dt_brain: Optional DT brain instance
            log_dir: Optional log directory override
            log_level: Optional log level override
            slack_enabled: Enable Slack forwarding (default: True)
        """
        super().__init__(name, source, dt_brain, log_dir, log_level)
        
        # Check if Slack forwarding is enabled
        self.slack_enabled = slack_enabled and os.getenv("LOG_SLACK_ENABLED", "1") == "1"
        
        # Lazy-load log aggregator to avoid circular imports
        self._aggregator = None
    
    def _get_aggregator(self):
        """Get log aggregator instance (lazy-loaded)."""
        if self._aggregator is None and self.slack_enabled:
            try:
                from backend.monitoring.log_aggregator import get_aggregator
                self._aggregator = get_aggregator()
            except Exception:
                # If aggregator unavailable, disable Slack forwarding
                self.slack_enabled = False
        return self._aggregator
    
    def _send_to_slack(self, level: str, message: str, context: Dict[str, Any]) -> None:
        """Send log entry to Slack via aggregator.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            message: Log message
            context: Context dictionary
        """
        if not self.slack_enabled:
            return
        
        try:
            aggregator = self._get_aggregator()
            if aggregator:
                aggregator.forward_log(level, message, self.name, context)
        except Exception:
            # Silently fail - Slack forwarding should never break normal logging
            pass
    
    def error(self, message: str, exc: Optional[BaseException] = None, **context) -> None:
        """Log error and send to Slack.
        
        Args:
            message: Error message
            exc: Optional exception
            **context: Additional context
        """
        # Log normally first
        super().error(message, exc=exc, **context)
        
        # Forward to Slack
        self._send_to_slack("ERROR", message, context)
    
    def warn(self, message: str, **context) -> None:
        """Log warning and send to Slack.
        
        Args:
            message: Warning message
            **context: Additional context
        """
        # Log normally first
        super().warn(message, **context)
        
        # Forward to Slack (if not in errors-only mode)
        errors_only = os.getenv("LOG_SLACK_ERRORS_ONLY", "0") == "1"
        if not errors_only:
            self._send_to_slack("WARNING", message, context)
    
    def info(self, message: str, **context) -> None:
        """Log info message.
        
        Info messages are not forwarded to Slack by default unless
        LOG_SLACK_ERRORS_ONLY=0.
        
        Args:
            message: Info message
            **context: Additional context
        """
        # Log normally first
        super().info(message, **context)
        
        # Forward to Slack only if all logs should be sent
        errors_only = os.getenv("LOG_SLACK_ERRORS_ONLY", "0") == "1"
        if not errors_only:
            self._send_to_slack("INFO", message, context)
    
    def debug(self, message: str, **context) -> None:
        """Log debug message.
        
        Debug messages are never forwarded to Slack.
        
        Args:
            message: Debug message
            **context: Additional context
        """
        # Log normally - debug never goes to Slack
        super().debug(message, **context)


def create_slack_logger(
    name: str,
    source: str = "backend",
    slack_enabled: bool = True,
) -> SlackLogger:
    """Create a Slack-aware logger instance.
    
    Args:
        name: Component name
        source: System source (swing/dt/backend)
        slack_enabled: Enable Slack forwarding
        
    Returns:
        SlackLogger instance
    """
    return SlackLogger(name=name, source=source, slack_enabled=slack_enabled)
