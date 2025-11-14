"""
backend_extension.py — v1.3 (Import-Safe Drift Report Wrapper)
Author: AION Analytics / StockAnalyzerPro

Purpose:
- Provides a unified and stable call interface for drift reporting.
- Safely imports `run_drift_report` from its possible locations.
- Tries both 2-arg and 3-arg signatures automatically.
- Never raises import errors if drift monitoring is unavailable.
"""

from __future__ import annotations
from typing import Optional


def run_latest_drift(previous: str, latest: str) -> Optional[str]:
    """
    Stable wrapper for running the latest drift comparison.
    Returns:
        str | None — path to generated drift report, or None if unavailable.
    """
    run_drift_report = None

    # Try modern location first
    try:
        from backend.drift_monitor import run_drift_report  # type: ignore
    except Exception:
        # Try legacy fallback
        try:
            from backend.backenddrift_monitor import run_drift_report  # type: ignore
        except Exception:
            # No drift module available at all
            print(ℹ️ Drift monitor module not found — skipping drift check.")
            return None

    # Attempt 2-argument signature
    try:
        result = run_drift_report(previous, latest)  # type: ignore[arg-type]
        return result
    except TypeError:
        pass

    # Attempt legacy 3-argument variant
    try:
        result = run_drift_report(previous, latest, "latest_drift")  # type: ignore[arg-type]
        return result
    except TypeError:
        print("⚠️ Drift function signature mismatch — skipping drift run.")
        return None
    except Exception as e:
        print(f"⚠️ Drift run failed: {e}")
        return None
