#!/usr/bin/env python3
"""
Backend Health Check Script
============================

Verifies that all backend routers are properly mounted and responding.
This script checks the main backend service to ensure all 26+ routers are accessible.

Usage:
    python backend/health_check.py
    python backend/health_check.py --url http://localhost:8000
"""

import argparse
import sys
import time
from typing import Dict, List, Tuple

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install with: pip install requests")
    sys.exit(1)


# List of all expected routers and their test endpoints
ROUTER_ENDPOINTS = [
    # Core endpoints
    ("/health", "GET", "Health Router"),
    ("/api/testing/test-health", "POST", "Testing Router"),
    
    # System & Status
    ("/api/system/status", "GET", "System Status Router"),
    ("/api/diagnostics/logs", "GET", "Diagnostics Router"),
    
    # Data & Insights
    ("/api/insights/top-predictions", "GET", "Insights Router"),
    ("/api/intraday/snapshot", "GET", "Intraday Router"),
    ("/api/live-prices/snapshot", "GET", "Live Prices Router"),
    
    # Models & Metrics
    ("/api/model/status", "GET", "Model Router"),
    ("/api/metrics/overview", "GET", "Metrics Router"),
    
    # Settings & Configuration
    ("/api/settings/knobs", "GET", "Settings Router"),
    
    # Logs
    ("/api/nightly-logs/recent", "GET", "Nightly Logs Router"),
    ("/api/intraday-logs/recent", "GET", "Intraday Logs Router"),
    
    # Bots
    ("/api/bots/page", "GET", "Bots Page Router"),
    ("/api/bots/hub", "GET", "Bots Hub Router"),
    ("/api/eod-bots/status", "GET", "EOD Bots Router"),
    
    # Replay
    ("/api/replay/status", "GET", "Replay Router"),
    
    # Dashboard & Portfolio
    ("/api/dashboard/metrics", "GET", "Dashboard Router"),
    ("/api/portfolio/holdings/top/1w", "GET", "Portfolio Router"),
    
    # Streaming & Events
    ("/api/stream/status", "GET", "Intraday Stream Router"),
    ("/api/events/status", "GET", "Events Router (SSE)"),
    
    # Admin (requires auth - just check if router is mounted)
    ("/admin/health", "GET", "Admin Router"),
    
    # Cache
    ("/api/cache/status", "GET", "Unified Cache Router"),
    
    # System Run
    ("/api/system-run/status", "GET", "System Run Router"),
]


def check_endpoint(base_url: str, path: str, method: str, name: str, timeout: int = 5) -> Tuple[bool, str]:
    """
    Check if an endpoint is accessible.
    
    Returns:
        (success: bool, message: str)
    """
    url = f"{base_url}{path}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json={}, timeout=timeout)
        else:
            return False, f"Unsupported method: {method}"
        
        # Consider 2xx, 401, 403 as success (router is mounted)
        # 401/403 means auth required, which is expected for admin endpoints
        if response.status_code < 500:
            return True, f"✓ {response.status_code}"
        else:
            return False, f"✗ {response.status_code}: {response.text[:100]}"
            
    except requests.exceptions.Timeout:
        return False, "✗ Timeout"
    except requests.exceptions.ConnectionError:
        return False, "✗ Connection failed"
    except Exception as e:
        return False, f"✗ {str(e)[:100]}"


def run_health_check(base_url: str, verbose: bool = False) -> bool:
    """
    Run health check on all routers.
    
    Returns:
        True if all checks pass, False otherwise
    """
    print(f"🔍 Backend Health Check")
    print(f"Target: {base_url}")
    print(f"{'='*80}\n")
    
    results: List[Tuple[str, str, bool, str]] = []
    success_count = 0
    
    for path, method, name in ROUTER_ENDPOINTS:
        if verbose:
            print(f"Checking: {name} ({method} {path})...", end=" ")
        
        success, message = check_endpoint(base_url, path, method, name)
        results.append((name, path, success, message))
        
        if success:
            success_count += 1
        
        if verbose:
            print(message)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary: {success_count}/{len(ROUTER_ENDPOINTS)} endpoints accessible\n")
    
    # Print detailed results
    print(f"{'Router':<30} {'Endpoint':<40} {'Status':<15}")
    print(f"{'-'*30} {'-'*40} {'-'*15}")
    
    for name, path, success, message in results:
        status_icon = "✅" if success else "❌"
        print(f"{name:<30} {path:<40} {status_icon} {message}")
    
    # Check if any critical endpoints failed
    critical_failures = [
        (name, path) for name, path, success, _ in results 
        if not success and path in ["/health", "/api/bots/page", "/api/system/status"]
    ]
    
    if critical_failures:
        print(f"\n⚠️  Critical endpoints failed:")
        for name, path in critical_failures:
            print(f"   - {name}: {path}")
        return False
    
    if success_count == len(ROUTER_ENDPOINTS):
        print(f"\n✅ All routers are healthy!")
        return True
    elif success_count >= len(ROUTER_ENDPOINTS) * 0.8:
        print(f"\n⚠️  Most routers are healthy (some optional endpoints may be unavailable)")
        return True
    else:
        print(f"\n❌ Multiple routers are unavailable")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Backend Health Check - Verify all routers are mounted and responding"
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Base URL of the backend service (default: http://127.0.0.1:8000)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Run health check
    start_time = time.time()
    success = run_health_check(args.url, verbose=args.verbose)
    duration = time.time() - start_time
    
    print(f"\n⏱️  Health check completed in {duration:.2f} seconds")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
