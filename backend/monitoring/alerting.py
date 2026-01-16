"""Multi-channel Slack alerting system for AION Analytics.

Provides intelligent routing of alerts to different Slack channels:
- #errors-tracebacks: Critical errors, exceptions, crashes
- #trading-alerts: Trading system alerts (emergency stops)
- #day_trading: DT-specific alerts
- #swing_trading: Swing/EOD bot alerts
- #nightly-logs-summary: Nightly job summaries
- #daily-pnl: PnL updates, equity tracking
- #reports: Insights, model metrics, regime changes
- #testing: Test alerts only
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

import requests

from utils.logger import log

# Alert levels
AlertLevel = Literal["critical", "warning", "info"]

# Channel configuration - maps channel names to environment variables
CHANNELS = {
    "errors": os.getenv("SLACK_WEBHOOK_ERRORS", ""),
    "trading": os.getenv("SLACK_WEBHOOK_TRADING", ""),
    "dt": os.getenv("SLACK_WEBHOOK_DT", ""),
    "swing": os.getenv("SLACK_WEBHOOK_SWING", ""),
    "nightly": os.getenv("SLACK_WEBHOOK_NIGHTLY", ""),
    "pnl": os.getenv("SLACK_WEBHOOK_PNL", ""),
    "reports": os.getenv("SLACK_WEBHOOK_REPORTS", ""),
    "testing": os.getenv("SLACK_WEBHOOK_TESTING", ""),
}

# Fallback to trading channel if specific channel not configured
DEFAULT_CHANNEL = "trading"

# Rate limiting
_last_alert_times: Dict[str, float] = {}
RATE_LIMIT_SECONDS = int(os.getenv("ALERT_RATE_LIMIT_SECONDS", "300"))


def _send_slack_alert(
    level: AlertLevel,
    title: str,
    message: str,
    channel: str = "trading",
    mention_channel: bool = False,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Send alert to specific Slack channel.
    
    Args:
        level: Alert level (critical, warning, info)
        title: Alert title
        message: Alert message
        channel: Target channel name (errors, trading, dt, swing, nightly, pnl, reports, testing)
        mention_channel: If True, adds @channel mention for critical alerts
        context: Additional context fields to display in the alert
    """
    # Get webhook URL for channel
    webhook_url = CHANNELS.get(channel, CHANNELS.get(DEFAULT_CHANNEL, ""))
    
    if not webhook_url:
        log(f"[alerting] No webhook configured for channel: {channel}")
        return
    
    # Add @channel mention if critical
    title_text = f"<!channel> {title}" if mention_channel else title
    
    # Add context fields
    fields = []
    if context:
        for key, value in context.items():
            fields.append({"title": key, "value": str(value), "short": True})
    
    # Color mapping
    colors = {
        "critical": "#FF0000",  # Red
        "warning": "#FFA500",   # Orange
        "info": "#00FF00",      # Green
    }
    
    # Emoji mapping
    emojis = {
        "critical": "🚨",
        "warning": "⚠️",
        "info": "ℹ️",
    }
    
    payload = {
        "username": "AION Analytics",
        "icon_emoji": ":robot_face:",
        "attachments": [
            {
                "color": colors[level],
                "title": f"{emojis[level]} {title_text}",
                "text": message,
                "fields": fields,
                "footer": f"AION Analytics → #{channel}",
                "ts": int(datetime.now(timezone.utc).timestamp()),
            }
        ],
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        
        if response.status_code == 200:
            log(f"[alerting] ✅ Slack alert sent to #{channel}: {title}")
        else:
            log(f"[alerting] ⚠️ Slack alert failed for #{channel}: HTTP {response.status_code}")
    
    except Exception as e:
        log(f"[alerting] ❌ Slack alert error for #{channel}: {e}")


def _should_send_alert(alert_key: str) -> bool:
    """Check if alert should be sent based on rate limiting.
    
    Args:
        alert_key: Unique key for the alert (typically title)
        
    Returns:
        True if alert should be sent, False if rate-limited
    """
    if RATE_LIMIT_SECONDS <= 0:
        return True
    
    now = time.time()
    last_time = _last_alert_times.get(alert_key, 0)
    
    if now - last_time >= RATE_LIMIT_SECONDS:
        _last_alert_times[alert_key] = now
        return True
    
    return False


def send_alert(
    level: AlertLevel,
    title: str,
    message: str,
    channel: str = "trading",
    mention_channel: bool = False,
    context: Optional[Dict[str, Any]] = None,
    skip_rate_limit: bool = False,
    **kwargs,
) -> None:
    """Send an alert to the specified Slack channel.
    
    Args:
        level: Alert level (critical, warning, info)
        title: Alert title
        message: Alert message
        channel: Target channel name
        mention_channel: If True, adds @channel mention
        context: Additional context fields
        skip_rate_limit: If True, bypasses rate limiting
        **kwargs: Additional context fields (merged with context dict)
    """
    # Merge kwargs into context
    if context is None:
        context = {}
    context.update(kwargs)
    
    # Check rate limiting
    if not skip_rate_limit:
        alert_key = f"{channel}:{title}"
        if not _should_send_alert(alert_key):
            log(f"[alerting] Rate-limited alert skipped: {alert_key}")
            return
    
    # Send to Slack
    _send_slack_alert(level, title, message, channel, mention_channel, context)


def alert_critical(
    title: str,
    message: str,
    channel: str = "trading",
    mention_channel: bool = False,
    context: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """Send critical alert to specified channel.
    
    Args:
        title: Alert title
        message: Alert message
        channel: Target channel name (default: trading)
        mention_channel: If True, adds @channel mention
        context: Additional context fields
        **kwargs: Additional context fields
    """
    send_alert("critical", title, message, channel=channel, mention_channel=mention_channel, context=context, **kwargs)


def alert_warning(
    title: str,
    message: str,
    channel: str = "trading",
    context: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """Send warning alert to specified channel.
    
    Args:
        title: Alert title
        message: Alert message
        channel: Target channel name (default: trading)
        context: Additional context fields
        **kwargs: Additional context fields
    """
    send_alert("warning", title, message, channel=channel, context=context, **kwargs)


def alert_info(
    title: str,
    message: str,
    channel: str = "trading",
    context: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """Send info alert to specified channel.
    
    Args:
        title: Alert title
        message: Alert message
        channel: Target channel name (default: trading)
        context: Additional context fields
        **kwargs: Additional context fields
    """
    send_alert("info", title, message, channel=channel, context=context, **kwargs)


# Smart routing helpers


def alert_error(title: str, message: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> None:
    """Send to #errors-tracebacks with @channel mention.
    
    Args:
        title: Alert title
        message: Alert message
        context: Additional context fields
        **kwargs: Additional context fields
    """
    alert_critical(title, message, channel="errors", mention_channel=True, context=context, **kwargs)


def alert_dt(
    title: str,
    message: str,
    level: AlertLevel = "info",
    context: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """Send to #day_trading.
    
    Args:
        title: Alert title
        message: Alert message
        level: Alert level (default: info)
        context: Additional context fields
        **kwargs: Additional context fields
    """
    send_alert(level, title, message, channel="dt", context=context, **kwargs)


def alert_swing(
    title: str,
    message: str,
    level: AlertLevel = "info",
    context: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """Send to #swing_trading.
    
    Args:
        title: Alert title
        message: Alert message
        level: Alert level (default: info)
        context: Additional context fields
        **kwargs: Additional context fields
    """
    send_alert(level, title, message, channel="swing", context=context, **kwargs)


def alert_nightly(
    title: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """Send to #nightly-logs-summary.
    
    Args:
        title: Alert title
        message: Alert message
        context: Additional context fields
        **kwargs: Additional context fields
    """
    alert_info(title, message, channel="nightly", context=context, **kwargs)


def alert_pnl(
    title: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """Send to #daily-pnl.
    
    Args:
        title: Alert title
        message: Alert message
        context: Additional context fields
        **kwargs: Additional context fields
    """
    alert_info(title, message, channel="pnl", context=context, **kwargs)


def alert_report(
    title: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """Send to #reports.
    
    Args:
        title: Alert title
        message: Alert message
        context: Additional context fields
        **kwargs: Additional context fields
    """
    alert_info(title, message, channel="reports", context=context, **kwargs)
