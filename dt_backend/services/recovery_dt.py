"""dt_backend/services/recovery_dt.py — v1.0

Recovery service for crash recovery and startup validation.

Runs on system startup to:
1. Replay execution ledger to detect incomplete executions
2. Alert on pending/stuck executions that need manual review
3. Validate state consistency between ledger and position files
4. Optionally auto-reconcile simple discrepancies

This prevents silent data loss and ensures system can resume cleanly after crashes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dt_backend.core.logger_dt import log
from dt_backend.services import execution_ledger

# Optional: Slack alerting
try:
    from backend.monitoring.alerting import alert_dt, alert_error
except ImportError:
    alert_dt = None  # type: ignore
    alert_error = None  # type: ignore


def run_recovery_check(*, alert_on_issues: bool = True) -> Dict[str, Any]:
    """Run startup recovery check.
    
    Analyzes execution ledger and position state to detect:
    - Incomplete executions (pending/confirmed but not recorded)
    - State inconsistencies
    - Need for manual intervention
    
    Args:
        alert_on_issues: If True, send Slack alerts for issues
        
    Returns:
        Dict with recovery status and any issues found
    """
    log("[recovery] 🔍 Running startup recovery check...")
    
    # Replay ledger to get state
    replay_result = execution_ledger.replay_ledger()
    
    status = replay_result.get("status", "unknown")
    total = replay_result.get("total", 0)
    completed = replay_result.get("completed", 0)
    pending = replay_result.get("pending", 0)
    failed = replay_result.get("failed", 0)
    incomplete = replay_result.get("incomplete", [])
    
    # Build summary
    issues: List[Dict[str, Any]] = []
    needs_attention = False
    
    # Check for incomplete executions
    if pending > 0:
        needs_attention = True
        for rec in incomplete:
            issue = {
                "type": "incomplete_execution",
                "execution_id": rec.get("execution_id"),
                "symbol": rec.get("symbol"),
                "side": rec.get("side"),
                "qty": rec.get("qty"),
                "status": rec.get("status"),
                "ts": rec.get("ts"),
                "broker_order_id": rec.get("broker_order_id"),
            }
            issues.append(issue)
            log(f"[recovery] ⚠️ Incomplete: {issue['execution_id']} {issue['status']} - {issue['symbol']}")
    
    # Alert if needed
    if needs_attention and alert_on_issues:
        _send_recovery_alert(incomplete, pending, failed)
    
    result = {
        "status": "needs_attention" if needs_attention else "ok",
        "total_executions": total,
        "completed": completed,
        "pending": pending,
        "failed": failed,
        "issues": issues,
        "needs_manual_review": needs_attention,
    }
    
    if needs_attention:
        log(f"[recovery] ⚠️ {pending} incomplete executions need manual review")
    else:
        log(f"[recovery] ✅ Recovery check passed - all executions complete")
    
    return result


def validate_position_consistency() -> Dict[str, Any]:
    """Validate consistency between ledger and position state.
    
    Compares:
    - Completed executions in ledger
    - Current positions in positions_dt.json
    
    Returns:
        Dict with validation results
    """
    log("[recovery] 🔍 Validating position consistency...")
    
    try:
        # Get completed executions from today
        recent = execution_ledger.get_recent(limit=500)
        completed_today = [
            r for r in recent
            if r.get("status") == "recorded"
            and r.get("ts", "")[:10] == _today_date()
        ]
        
        # Read current positions
        from dt_backend.services.position_manager_dt import read_positions_state
        positions = read_positions_state()
        
        # Build expected positions from ledger
        ledger_positions = {}
        for rec in completed_today:
            sym = rec.get("symbol")
            side = rec.get("side")
            qty = rec.get("qty", 0.0)
            
            if sym not in ledger_positions:
                ledger_positions[sym] = {"qty": 0.0, "entries": 0, "exits": 0}
            
            if side == "BUY":
                ledger_positions[sym]["qty"] += qty
                ledger_positions[sym]["entries"] += 1
            elif side == "SELL":
                ledger_positions[sym]["qty"] -= qty
                ledger_positions[sym]["exits"] += 1
        
        # Compare with actual positions
        discrepancies = []
        for sym, expected in ledger_positions.items():
            actual_pos = positions.get(sym, {})
            actual_qty = actual_pos.get("qty", 0.0) if isinstance(actual_pos, dict) else 0.0
            expected_qty = expected["qty"]
            
            # Allow small floating point differences
            if abs(actual_qty - expected_qty) > 0.001:
                discrepancies.append({
                    "symbol": sym,
                    "expected_qty": expected_qty,
                    "actual_qty": actual_qty,
                    "difference": actual_qty - expected_qty,
                })
        
        result = {
            "status": "ok" if not discrepancies else "discrepancies_found",
            "ledger_positions": len(ledger_positions),
            "actual_positions": len([p for p in positions.values() if isinstance(p, dict) and p.get("status") == "OPEN"]),
            "discrepancies": discrepancies,
        }
        
        if discrepancies:
            log(f"[recovery] ⚠️ Found {len(discrepancies)} position discrepancies")
            for disc in discrepancies:
                log(f"[recovery]   {disc['symbol']}: expected={disc['expected_qty']}, actual={disc['actual_qty']}")
        else:
            log("[recovery] ✅ Position consistency validated")
        
        return result
        
    except Exception as e:
        log(f"[recovery] ⚠️ Error validating positions: {e}")
        return {
            "status": "error",
            "error": str(e),
            "ledger_positions": 0,
            "actual_positions": 0,
            "discrepancies": [],
        }


def _today_date() -> str:
    """Get today's date in YYYY-MM-DD format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _send_recovery_alert(incomplete: List[Dict[str, Any]], pending: int, failed: int) -> None:
    """Send Slack alert about recovery issues."""
    if alert_error is None:
        return
    
    try:
        # Build context for alert
        context = {
            "Pending Executions": pending,
            "Failed Executions": failed,
        }
        
        # Add details for each incomplete execution
        for i, rec in enumerate(incomplete[:5]):  # Limit to 5 for brevity
            key = f"Incomplete #{i+1}"
            context[key] = f"{rec.get('symbol')} {rec.get('side')} {rec.get('qty')} ({rec.get('status')})"
        
        if len(incomplete) > 5:
            context["..."] = f"({len(incomplete) - 5} more)"
        
        alert_error(
            "Recovery Check: Incomplete Executions",
            f"Found {pending} incomplete executions that need manual review",
            context=context,
        )
        
    except Exception as e:
        log(f"[recovery] ⚠️ Failed to send recovery alert: {e}")


def reconcile_execution(execution_id: str, *, action: str = "mark_failed") -> bool:
    """Manually reconcile a stuck execution.
    
    Actions:
    - "mark_failed": Mark execution as failed (safe default)
    - "mark_recorded": Mark as complete (if you manually verified state)
    
    Args:
        execution_id: Execution ID to reconcile
        action: Reconciliation action
        
    Returns:
        bool: True if reconciled successfully
    """
    log(f"[recovery] 🔧 Reconciling {execution_id} with action={action}")
    
    if action == "mark_failed":
        return execution_ledger.record_failed(
            execution_id,
            error_msg="Manually marked as failed during recovery",
        )
    elif action == "mark_recorded":
        # This is risky - should only be done after manual verification
        log(f"[recovery] ⚠️ Manually marking {execution_id} as recorded - ensure state is correct!")
        return execution_ledger.record_recorded(
            execution_id,
            position_state={"manual_reconciliation": True},
        )
    else:
        log(f"[recovery] ⚠️ Unknown reconciliation action: {action}")
        return False


def get_recovery_status() -> Dict[str, Any]:
    """Get current recovery status (for API/monitoring).
    
    Returns quick summary without performing full check.
    """
    pending = execution_ledger.get_pending()
    
    return {
        "pending_executions": len(pending),
        "needs_attention": len(pending) > 0,
        "pending_details": [
            {
                "execution_id": p.get("execution_id"),
                "symbol": p.get("symbol"),
                "side": p.get("side"),
                "status": p.get("status"),
                "ts": p.get("ts"),
            }
            for p in pending
        ],
    }
