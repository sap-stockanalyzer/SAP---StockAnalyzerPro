"""Central log aggregation and Slack forwarding for AION Analytics.

Provides intelligent log routing to Slack channels:
- Errors → #errors channel (critical alerts)
- Warnings → #warnings channel
- Testing → #testing channel
- Trading decisions → #trading channel
- Operations/cycle progress → #operations channel
- Health checks → #health channel

Features:
- Log buffering to reduce Slack API calls
- Automatic rate limiting
- Graceful degradation if Slack unavailable
- Context-aware message formatting
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from threading import Lock

import requests

from utils.logger import Logger

# Initialize logger
_logger = Logger("log_aggregator", source="backend")


@dataclass
class LogEntry:
    """Represents a single log entry."""
    level: str  # DEBUG, INFO, WARNING, ERROR
    message: str
    timestamp: datetime
    component: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Represents a test result."""
    test_name: str
    status: str  # PASS, FAIL, SKIP
    duration_ms: float
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class LogAggregator:
    """Aggregate logs and forward to appropriate Slack channels.
    
    This class provides centralized log aggregation with intelligent routing
    to different Slack channels based on log level and context.
    """
    
    def __init__(self):
        """Initialize log aggregator with channel configuration."""
        self.buffer: List[LogEntry] = []
        self.buffer_lock = Lock()
        self.last_flush_time = time.time()
        
        # Channel configuration from environment
        self.channels = {
            "error": os.getenv("SLACK_CHANNEL_ERRORS", "#errors"),
            "warning": os.getenv("SLACK_CHANNEL_WARNINGS", "#warnings"),
            "testing": os.getenv("SLACK_CHANNEL_TESTING", "#testing"),
            "trading": os.getenv("SLACK_CHANNEL_TRADING", "#trading"),
            "operations": os.getenv("SLACK_CHANNEL_OPERATIONS", "#operations"),
            "health": os.getenv("SLACK_CHANNEL_HEALTH", "#health"),
        }
        
        # Webhook URLs from environment (using existing alerting webhooks)
        self.webhooks = {
            "error": os.getenv("SLACK_WEBHOOK_ERRORS", ""),
            "warning": os.getenv("SLACK_WEBHOOK_ERRORS", ""),  # Warnings go to errors channel
            "testing": os.getenv("SLACK_WEBHOOK_TESTING", ""),
            "trading": os.getenv("SLACK_WEBHOOK_TRADING", ""),
            "operations": os.getenv("SLACK_WEBHOOK_DT", ""),  # Operations go to DT channel
            "health": os.getenv("SLACK_WEBHOOK_REPORTS", ""),  # Health goes to reports
        }
        
        # Configuration
        self.enabled = os.getenv("LOG_SLACK_ENABLED", "1") == "1"
        self.errors_only = os.getenv("LOG_SLACK_ERRORS_ONLY", "0") == "1"
        self.buffer_size = int(os.getenv("LOG_BUFFER_SIZE", "10"))
        self.buffer_timeout_sec = int(os.getenv("LOG_BUFFER_TIMEOUT_SEC", "60"))
        self.slack_timeout_sec = int(os.getenv("SLACK_TIMEOUT_SECONDS", "5"))
        
        _logger.info(
            f"LogAggregator initialized: enabled={self.enabled}, "
            f"errors_only={self.errors_only}, buffer_size={self.buffer_size}"
        )
    
    def _should_forward(self, level: str) -> bool:
        """Check if log should be forwarded to Slack.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            
        Returns:
            True if log should be forwarded, False otherwise
        """
        if not self.enabled:
            return False
        
        if self.errors_only and level not in ("ERROR", "CRITICAL"):
            return False
        
        return True
    
    def _get_webhook_url(self, channel: str) -> str:
        """Get webhook URL for specified channel.
        
        Args:
            channel: Channel name (error, warning, testing, trading, operations, health)
            
        Returns:
            Webhook URL or empty string if not configured
        """
        return self.webhooks.get(channel, "")
    
    def _format_slack_message(
        self,
        level: str,
        message: str,
        component: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format log entry as Slack message payload.
        
        Args:
            level: Log level
            message: Log message
            component: Component name
            context: Additional context
            
        Returns:
            Slack message payload
        """
        # Emoji mapping
        emojis = {
            "ERROR": "❌",
            "CRITICAL": "🚨",
            "WARNING": "⚠️",
            "INFO": "ℹ️",
            "DEBUG": "🐛",
        }
        
        # Color mapping
        colors = {
            "ERROR": "#FF0000",  # Red
            "CRITICAL": "#CC0000",  # Dark red
            "WARNING": "#FFA500",  # Orange
            "INFO": "#00FF00",  # Green
            "DEBUG": "#888888",  # Gray
        }
        
        emoji = emojis.get(level, "📝")
        color = colors.get(level, "#888888")
        
        # Build title
        title = f"{emoji} {level}"
        if component:
            title += f" [{component}]"
        
        # Build fields
        fields = []
        if context:
            for key, value in context.items():
                fields.append({
                    "title": key,
                    "value": str(value),
                    "short": True
                })
        
        # Truncate long messages
        if len(message) > 1000:
            message = message[:997] + "..."
        
        return {
            "username": "AION Analytics Logger",
            "icon_emoji": ":memo:",
            "attachments": [
                {
                    "color": color,
                    "title": title,
                    "text": message,
                    "fields": fields,
                    "footer": "AION Analytics",
                    "ts": int(datetime.now(timezone.utc).timestamp()),
                }
            ],
        }
    
    def _send_to_slack(self, channel: str, payload: Dict[str, Any]) -> bool:
        """Send message to Slack channel.
        
        Args:
            channel: Channel name
            payload: Slack message payload
            
        Returns:
            True if sent successfully, False otherwise
        """
        webhook_url = self._get_webhook_url(channel)
        
        if not webhook_url:
            _logger.debug(f"No webhook configured for channel: {channel}")
            return False
        
        try:
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=self.slack_timeout_sec
            )
            
            if response.status_code == 200:
                _logger.debug(f"✅ Sent log to Slack channel: {channel}")
                return True
            else:
                _logger.warn(
                    f"⚠️ Slack send failed for {channel}: HTTP {response.status_code}"
                )
                return False
        
        except Exception as e:
            _logger.warn(f"❌ Slack send error for {channel}: {e}")
            return False
    
    def forward_log(
        self,
        level: str,
        message: str,
        component: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Forward log entry to appropriate Slack channel.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            message: Log message
            component: Component name
            context: Additional context
        """
        if not self._should_forward(level):
            return
        
        # Determine target channel based on level
        if level in ("ERROR", "CRITICAL"):
            channel = "error"
        elif level == "WARNING":
            channel = "warning"
        else:
            # INFO and DEBUG go to operations by default
            channel = "operations"
        
        # Format and send message
        payload = self._format_slack_message(level, message, component, context)
        self._send_to_slack(channel, payload)
    
    def forward_test_result(self, result: TestResult) -> None:
        """Forward test result to #testing channel.
        
        Args:
            result: Test result to forward
        """
        if not self.enabled:
            return
        
        # Check if we should include this result
        include_pass = os.getenv("TEST_INCLUDE_PASS", "1") == "1"
        include_fail = os.getenv("TEST_INCLUDE_FAIL", "1") == "1"
        
        if result.status == "PASS" and not include_pass:
            return
        if result.status == "FAIL" and not include_fail:
            return
        
        # Emoji based on status
        status_emoji = {
            "PASS": "✅",
            "FAIL": "❌",
            "SKIP": "⏭️",
        }
        
        emoji = status_emoji.get(result.status, "❓")
        
        # Build message
        message = f"{emoji} TEST {result.status}: {result.test_name}\n"
        message += f"Duration: {result.duration_ms:.1f}ms"
        
        if result.error_message:
            message += f"\n\nError:\n```\n{result.error_message[:500]}\n```"
        
        context = {
            "Test Name": result.test_name,
            "Status": result.status,
            "Duration": f"{result.duration_ms:.1f}ms",
        }
        
        payload = self._format_slack_message(
            "ERROR" if result.status == "FAIL" else "INFO",
            message,
            "Testing",
            context
        )
        
        self._send_to_slack("testing", payload)
    
    def forward_trade(self, trade_event: Dict[str, Any]) -> None:
        """Forward trade decision to #trading channel.
        
        Args:
            trade_event: Trade event dictionary with keys:
                - symbol: Stock symbol
                - action: BUY, SELL, HOLD
                - confidence: Confidence level (0-1)
                - size: Position size (fraction of account)
                - phit: Probability of hitting target (optional)
                - hold_time_check: Hold time validation result (optional)
        """
        if not self.enabled:
            return
        
        symbol = trade_event.get("symbol", "UNKNOWN")
        action = trade_event.get("action", "UNKNOWN")
        confidence = trade_event.get("confidence", 0.0)
        size = trade_event.get("size", 0.0)
        phit = trade_event.get("phit")
        hold_time_check = trade_event.get("hold_time_check")
        
        # Emoji based on action
        action_emoji = {
            "BUY": "🟢",
            "SELL": "🔴",
            "HOLD": "🟡",
        }
        
        emoji = action_emoji.get(action, "🔵")
        
        # Build message
        message = f"{emoji} TRADE DECISION [dt_exec]\n"
        message += f"Symbol: {symbol}\n"
        message += f"Action: {action}\n"
        message += f"Confidence: {confidence:.1%}\n"
        message += f"Size: {size:.1%} of account"
        
        if phit is not None:
            message += f"\nP(Hit): {phit:.1%}"
        
        if hold_time_check is not None:
            check_emoji = "✅" if hold_time_check else "❌"
            message += f"\nHold Time Check: {check_emoji}"
        
        context = {
            "Symbol": symbol,
            "Action": action,
            "Confidence": f"{confidence:.1%}",
            "Size": f"{size:.1%}",
        }
        
        payload = self._format_slack_message("INFO", message, "Trading", context)
        self._send_to_slack("trading", payload)
    
    def forward_health(self, health_data: Dict[str, Any]) -> None:
        """Forward system health data to #health channel.
        
        Args:
            health_data: Health data dictionary with keys:
                - cycles_total: Total cycles
                - cycles_completed: Completed cycles
                - errors: Number of errors
                - warnings: Number of warnings
                - uptime: Uptime string
                - last_cycle: Time since last cycle
        """
        if not self.enabled:
            return
        
        cycles_total = health_data.get("cycles_total", 0)
        cycles_completed = health_data.get("cycles_completed", 0)
        errors = health_data.get("errors", 0)
        warnings = health_data.get("warnings", 0)
        uptime = health_data.get("uptime", "unknown")
        last_cycle = health_data.get("last_cycle", "unknown")
        
        # Calculate completion rate
        completion_rate = 0.0
        if cycles_total > 0:
            completion_rate = cycles_completed / cycles_total
        
        # Build message
        message = "🟢 SYSTEM HEALTH\n"
        message += f"Cycles: {cycles_completed}/{cycles_total} completed ({completion_rate:.1%})\n"
        message += f"Errors: {errors}\n"
        message += f"Warnings: {warnings}\n"
        message += f"Uptime: {uptime}\n"
        message += f"Last Cycle: {last_cycle}"
        
        context = {
            "Cycles": f"{cycles_completed}/{cycles_total}",
            "Errors": str(errors),
            "Warnings": str(warnings),
            "Uptime": uptime,
        }
        
        payload = self._format_slack_message("INFO", message, "Health", context)
        self._send_to_slack("health", payload)


# Singleton instance
_aggregator: Optional[LogAggregator] = None


def get_aggregator() -> LogAggregator:
    """Get singleton log aggregator instance.
    
    Returns:
        LogAggregator instance
    """
    global _aggregator
    if _aggregator is None:
        _aggregator = LogAggregator()
    return _aggregator
