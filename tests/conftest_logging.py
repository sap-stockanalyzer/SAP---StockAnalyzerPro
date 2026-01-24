"""Pytest logging configuration for test result tracking and Slack integration.

Provides fixtures and hooks for:
- Capturing test logs
- Tracking test results
- Sending test results to Slack
- Integration with log aggregator
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import pytest


@dataclass
class TestLogEntry:
    """Represents a test log entry."""
    test_name: str
    message: str
    level: str  # DEBUG, INFO, WARNING, ERROR
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TestLogHandler:
    """Capture test logs and failures for Slack integration.
    
    This class is used as a pytest fixture to track test execution,
    capture logs, and optionally send results to Slack.
    """
    
    def __init__(self):
        """Initialize test log handler."""
        self.logs: List[TestLogEntry] = []
        self.failures: List[Dict[str, Any]] = []
        self.passes: List[Dict[str, Any]] = []
        self.current_test: Optional[str] = None
        self.test_start_time: Optional[float] = None
        
        # Check if Slack integration is enabled
        self.slack_enabled = os.getenv("TEST_LOG_TO_SLACK", "0") == "1"
    
    def on_test_start(self, test_name: str) -> None:
        """Log test start.
        
        Args:
            test_name: Name of the test
        """
        self.current_test = test_name
        self.test_start_time = time.time()
        
        log_entry = TestLogEntry(
            test_name=test_name,
            message=f"Starting test: {test_name}",
            level="INFO"
        )
        self.logs.append(log_entry)
    
    def on_test_end(
        self,
        test_name: str,
        status: str,
        duration_ms: float,
        error_message: Optional[str] = None
    ) -> None:
        """Log test end with result.
        
        Args:
            test_name: Name of the test
            status: Test status (PASS, FAIL, SKIP)
            duration_ms: Test duration in milliseconds
            error_message: Optional error message for failures
        """
        result = {
            "test_name": test_name,
            "status": status,
            "duration_ms": duration_ms,
            "error_message": error_message,
            "timestamp": datetime.now(timezone.utc),
        }
        
        if status == "FAIL":
            self.failures.append(result)
        elif status == "PASS":
            self.passes.append(result)
        
        # Log the result
        log_entry = TestLogEntry(
            test_name=test_name,
            message=f"Test {status}: {test_name} ({duration_ms:.1f}ms)",
            level="ERROR" if status == "FAIL" else "INFO"
        )
        self.logs.append(log_entry)
        
        # Reset current test
        self.current_test = None
        self.test_start_time = None
    
    def send_to_slack(self) -> None:
        """Send test results to Slack.
        
        This is typically called at the end of a test session to send
        a summary of all test results.
        """
        if not self.slack_enabled:
            return
        
        try:
            from backend.monitoring.log_aggregator import get_aggregator, TestResult
            
            aggregator = get_aggregator()
            
            # Send each failure
            for failure in self.failures:
                result = TestResult(
                    test_name=failure["test_name"],
                    status=failure["status"],
                    duration_ms=failure["duration_ms"],
                    error_message=failure.get("error_message"),
                    timestamp=failure["timestamp"],
                )
                aggregator.forward_test_result(result)
            
            # Send passes if configured
            include_pass = os.getenv("TEST_INCLUDE_PASS", "1") == "1"
            if include_pass:
                for passed in self.passes:
                    result = TestResult(
                        test_name=passed["test_name"],
                        status=passed["status"],
                        duration_ms=passed["duration_ms"],
                        timestamp=passed["timestamp"],
                    )
                    aggregator.forward_test_result(result)
        
        except Exception as e:
            # Don't fail tests if Slack integration fails
            print(f"Warning: Failed to send test results to Slack: {e}")


@pytest.fixture
def log_capture():
    """Pytest fixture to capture logs for test.
    
    Usage:
        def test_something(log_capture):
            log_capture.on_test_start("test_something")
            # Test logic
            log_capture.on_test_end("test_something", "PASS", 15.0)
    
    Yields:
        TestLogHandler instance
    """
    handler = TestLogHandler()
    yield handler
    
    # Send to Slack if any failures occurred
    if handler.failures and handler.slack_enabled:
        handler.send_to_slack()


# Pytest hooks for automatic test tracking


def pytest_runtest_setup(item):
    """Hook called before each test runs."""
    # Store test start time in item
    item._test_start_time = time.time()


def pytest_runtest_makereport(item, call):
    """Hook called after each test phase (setup, call, teardown)."""
    # Only process after test execution (not setup/teardown)
    if call.when == "call":
        duration_ms = (time.time() - getattr(item, "_test_start_time", time.time())) * 1000
        
        # Determine status
        if call.excinfo is None:
            status = "PASS"
            error_msg = None
        elif call.excinfo.typename == "Skipped":
            status = "SKIP"
            error_msg = str(call.excinfo.value)
        else:
            status = "FAIL"
            error_msg = str(call.excinfo.value)
        
        # Send to Slack if enabled
        slack_enabled = os.getenv("TEST_LOG_TO_SLACK", "0") == "1"
        if slack_enabled and status in ("FAIL", "PASS"):
            try:
                from backend.monitoring.log_aggregator import get_aggregator, TestResult
                
                # Check if we should send this result
                include_pass = os.getenv("TEST_INCLUDE_PASS", "1") == "1"
                include_fail = os.getenv("TEST_INCLUDE_FAIL", "1") == "1"
                
                should_send = (status == "FAIL" and include_fail) or \
                              (status == "PASS" and include_pass)
                
                if should_send:
                    aggregator = get_aggregator()
                    result = TestResult(
                        test_name=item.nodeid,
                        status=status,
                        duration_ms=duration_ms,
                        error_message=error_msg,
                    )
                    aggregator.forward_test_result(result)
            
            except Exception:
                # Don't fail tests if Slack integration fails
                pass
