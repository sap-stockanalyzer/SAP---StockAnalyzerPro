"""Testing router for backend endpoint verification.

Provides individual test endpoints that can be called from the frontend
testing page. Each endpoint tests a specific backend function and returns
detailed results including response time and status.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.monitoring.alerting import send_alert

# Import router functions at module level for better performance
from backend.routers.health_router import health_check
from backend.routers.eod_bots_router import eod_status
from backend.routers.intraday_router import api_intraday_snapshot
from backend.routers.live_prices_router import api_live_prices
from backend.routers.dashboard_router import dashboard_metrics

router = APIRouter(prefix="/api/testing", tags=["testing"])


class TestResult(BaseModel):
    """Result of a test endpoint call."""
    test_name: str
    status: str  # "pass" or "fail"
    http_status: int
    response_time_ms: float
    timestamp: str
    message: str
    error: str | None = None


def send_test_alert(result: TestResult) -> None:
    """Send Slack alert for test result."""
    level = "info" if result.status == "pass" else "critical"
    emoji = "✅" if result.status == "pass" else "❌"
    
    title = f"Backend Test: {result.test_name}"
    message = f"""Status: {result.status.upper()} ({result.http_status})
Response Time: {result.response_time_ms:.0f}ms
Timestamp: {result.timestamp}
{f"Error: {result.error}" if result.error else ""}"""
    
    context = {
        "Test Name": result.test_name,
        "Status": f"{emoji} {result.status.upper()}",
        "HTTP Status": result.http_status,
        "Response Time": f"{result.response_time_ms:.0f}ms",
    }
    
    if result.error:
        context["Error"] = result.error
    
    # Send to testing channel with rate limiting disabled for tests
    send_alert(
        level=level,
        title=title,
        message=message,
        channel="testing",
        context=context,
        skip_rate_limit=True,
    )


@router.post("/test-health")
def test_health_endpoint() -> TestResult:
    """Test main health endpoint."""
    start_time = time.time()
    test_name = "Health Check"
    
    try:
        result = health_check()
        response_time_ms = (time.time() - start_time) * 1000
        
        # Verify response structure
        if "status" not in result:
            raise ValueError("Health check missing 'status' field")
        
        test_result = TestResult(
            test_name=test_name,
            status="pass",
            http_status=200,
            response_time_ms=response_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message=f"Health status: {result['status']}",
        )
        
        send_test_alert(test_result)
        return test_result
        
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        test_result = TestResult(
            test_name=test_name,
            status="fail",
            http_status=500,
            response_time_ms=response_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="Health check failed",
            error=str(e),
        )
        
        send_test_alert(test_result)
        return test_result


@router.post("/test-eod-status")
def test_eod_status_endpoint() -> TestResult:
    """Test EOD bot status endpoint."""
    start_time = time.time()
    test_name = "EOD Bot Status"
    
    try:
        # Run async function
        result = asyncio.run(eod_status())
        response_time_ms = (time.time() - start_time) * 1000
        
        test_result = TestResult(
            test_name=test_name,
            status="pass",
            http_status=200,
            response_time_ms=response_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="EOD bot status retrieved successfully",
        )
        
        send_test_alert(test_result)
        return test_result
        
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        test_result = TestResult(
            test_name=test_name,
            status="fail",
            http_status=500,
            response_time_ms=response_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="EOD bot status check failed",
            error=str(e),
        )
        
        send_test_alert(test_result)
        return test_result


@router.post("/test-intraday-snapshot")
def test_intraday_snapshot_endpoint() -> TestResult:
    """Test intraday snapshot endpoint."""
    start_time = time.time()
    test_name = "Intraday Snapshot"
    
    try:
        result = api_intraday_snapshot(limit=10)
        response_time_ms = (time.time() - start_time) * 1000
        
        test_result = TestResult(
            test_name=test_name,
            status="pass",
            http_status=200,
            response_time_ms=response_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="Intraday snapshot retrieved successfully",
        )
        
        send_test_alert(test_result)
        return test_result
        
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        test_result = TestResult(
            test_name=test_name,
            status="fail",
            http_status=500,
            response_time_ms=response_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="Intraday snapshot check failed",
            error=str(e),
        )
        
        send_test_alert(test_result)
        return test_result


@router.post("/test-live-prices")
def test_live_prices_endpoint() -> TestResult:
    """Test live prices endpoint."""
    start_time = time.time()
    test_name = "Live Prices"
    
    try:
        result = asyncio.run(api_live_prices(symbols=None, limit=10, include_intraday=False))
        response_time_ms = (time.time() - start_time) * 1000
        
        test_result = TestResult(
            test_name=test_name,
            status="pass",
            http_status=200,
            response_time_ms=response_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="Live prices retrieved successfully",
        )
        
        send_test_alert(test_result)
        return test_result
        
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        test_result = TestResult(
            test_name=test_name,
            status="fail",
            http_status=500,
            response_time_ms=response_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="Live prices check failed",
            error=str(e),
        )
        
        send_test_alert(test_result)
        return test_result


@router.post("/test-dashboard")
def test_dashboard_endpoint() -> TestResult:
    """Test dashboard metrics endpoint."""
    start_time = time.time()
    test_name = "Dashboard Metrics"
    
    try:
        result = dashboard_metrics()
        response_time_ms = (time.time() - start_time) * 1000
        
        test_result = TestResult(
            test_name=test_name,
            status="pass",
            http_status=200,
            response_time_ms=response_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="Dashboard metrics retrieved successfully",
        )
        
        send_test_alert(test_result)
        return test_result
        
    except Exception as e:
        response_time_ms = (time.time() - start_time) * 1000
        test_result = TestResult(
            test_name=test_name,
            status="fail",
            http_status=500,
            response_time_ms=response_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="Dashboard metrics check failed",
            error=str(e),
        )
        
        send_test_alert(test_result)
        return test_result
