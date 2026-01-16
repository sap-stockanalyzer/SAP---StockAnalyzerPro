"""Global error handling and circuit breaker for AION Analytics.

Provides:
- Global exception handler for FastAPI
- Circuit breaker pattern for external service calls
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from utils.logger import log


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors.
    
    Args:
        request: FastAPI request object
        exc: Exception that was raised
        
    Returns:
        JSONResponse with error details
    """
    # Import here to avoid circular dependency
    try:
        from backend.monitoring.alerting import alert_error
        
        # Send alert to #errors-tracebacks
        alert_error(
            "Unhandled Exception",
            f"Path: {request.url.path}\nError: {exc}",
            context={
                "Path": request.url.path,
                "Method": request.method,
                "User-Agent": request.headers.get("user-agent", "unknown"),
            },
        )
    except Exception as alert_exc:
        log(f"[error_handler] Failed to send alert: {alert_exc}")
    
    # Log the exception
    log(f"[error_handler] Unhandled exception on {request.url.path}: {exc}")
    
    # Return error response
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url.path),
        },
    )


class CircuitBreaker:
    """Circuit breaker pattern for external service calls.
    
    Tracks failures and opens circuit after threshold is reached.
    When circuit is open, calls fail immediately without hitting the service.
    Circuit automatically closes after cooldown period.
    
    States:
    - CLOSED: Normal operation, calls go through
    - OPEN: Too many failures, calls fail immediately
    - HALF_OPEN: Testing if service recovered
    
    Usage:
        circuit_breaker = CircuitBreaker("external_api", failure_threshold=5, cooldown_seconds=60)
        
        if circuit_breaker.can_execute():
            try:
                result = external_api.call()
                circuit_breaker.record_success()
            except Exception as e:
                circuit_breaker.record_failure()
                raise
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        cooldown_seconds: int = 60,
        alert_on_open: bool = True,
    ):
        """Initialize circuit breaker.
        
        Args:
            name: Name of the service/circuit
            failure_threshold: Number of failures before opening circuit
            cooldown_seconds: Seconds to wait before attempting recovery
            alert_on_open: If True, sends alert when circuit opens
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.alert_on_open = alert_on_open
        
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Check if a call can be executed.
        
        Returns:
            True if circuit allows execution, False if circuit is open
        """
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            # Check if cooldown period has passed
            if self.last_failure_time and time.time() - self.last_failure_time >= self.cooldown_seconds:
                log(f"[circuit_breaker] {self.name}: Entering HALF_OPEN state")
                self.state = "HALF_OPEN"
                return True
            return False
        
        if self.state == "HALF_OPEN":
            return True
        
        return False
    
    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == "HALF_OPEN":
            log(f"[circuit_breaker] {self.name}: Closing circuit after successful recovery")
            self.state = "CLOSED"
            self.failures = 0
            self.last_failure_time = None
        elif self.state == "CLOSED":
            # Reset failure count on success
            self.failures = 0
            self.last_failure_time = None
    
    def record_failure(self) -> None:
        """Record a failed call."""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.state == "HALF_OPEN":
            log(f"[circuit_breaker] {self.name}: Re-opening circuit after failed recovery")
            self.state = "OPEN"
            self._send_alert()
        elif self.failures >= self.failure_threshold:
            log(f"[circuit_breaker] {self.name}: Opening circuit after {self.failures} failures")
            self.state = "OPEN"
            self._send_alert()
    
    def _send_alert(self) -> None:
        """Send alert when circuit opens."""
        if not self.alert_on_open:
            return
        
        try:
            from backend.monitoring.alerting import alert_error
            
            alert_error(
                f"Circuit Breaker Opened: {self.name}",
                f"Service has failed {self.failures} times.",
                context={
                    "Service": self.name,
                    "Failures": self.failures,
                    "State": self.state,
                },
            )
        except Exception as e:
            log(f"[circuit_breaker] Failed to send alert: {e}")
    
    def get_state(self) -> dict:
        """Get current circuit breaker state.
        
        Returns:
            Dict with state information
        """
        return {
            "name": self.name,
            "state": self.state,
            "failures": self.failures,
            "threshold": self.failure_threshold,
            "last_failure": (
                datetime.fromtimestamp(self.last_failure_time, tz=timezone.utc).isoformat()
                if self.last_failure_time
                else None
            ),
        }
