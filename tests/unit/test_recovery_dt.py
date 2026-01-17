"""Tests for dt_backend/services/recovery_dt.py

Tests the crash recovery and startup validation service.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dt_backend.services import execution_ledger, recovery_dt


@pytest.fixture
def temp_recovery_dir(tmp_path, monkeypatch):
    """Set up temporary directory for recovery tests."""
    intraday_dir = tmp_path / "intraday"
    intraday_dir.mkdir(parents=True, exist_ok=True)
    
    positions_dir = intraday_dir / "positions"
    positions_dir.mkdir(parents=True, exist_ok=True)
    
    # Override paths
    monkeypatch.setenv("DT_TRUTH_DIR", str(tmp_path))
    
    yield tmp_path


class TestRecoveryCheck:
    """Test basic recovery check operations."""
    
    def test_recovery_check_no_issues(self, temp_recovery_dir):
        """Test recovery check when everything is clean."""
        # Create some completed executions
        exec1 = execution_ledger.record_pending("AAPL", "BUY", 10, 180.0)
        execution_ledger.record_confirmed(exec1, "b1", 180.0)
        execution_ledger.record_recorded(exec1, {"symbol": "AAPL"})
        
        exec2 = execution_ledger.record_pending("MSFT", "BUY", 5, 380.0)
        execution_ledger.record_confirmed(exec2, "b2", 380.0)
        execution_ledger.record_recorded(exec2, {"symbol": "MSFT"})
        
        # Run recovery check
        result = recovery_dt.run_recovery_check(alert_on_issues=False)
        
        assert result["status"] == "ok"
        assert result["total_executions"] == 2
        assert result["completed"] == 2
        assert result["pending"] == 0
        assert result["needs_manual_review"] is False
        assert len(result["issues"]) == 0
    
    def test_recovery_check_with_pending(self, temp_recovery_dir):
        """Test recovery check detects pending executions."""
        # Create some executions, leave some incomplete
        exec1 = execution_ledger.record_pending("AAPL", "BUY", 10, 180.0)
        execution_ledger.record_confirmed(exec1, "b1", 180.0)
        execution_ledger.record_recorded(exec1, {"symbol": "AAPL"})
        
        exec2 = execution_ledger.record_pending("MSFT", "BUY", 5, 380.0)
        # Leave pending - simulates crash after broker submit
        
        exec3 = execution_ledger.record_pending("GOOGL", "BUY", 2, 140.0)
        execution_ledger.record_confirmed(exec3, "b3", 140.0)
        # Leave confirmed - simulates crash before position update
        
        # Run recovery check
        result = recovery_dt.run_recovery_check(alert_on_issues=False)
        
        assert result["status"] == "needs_attention"
        assert result["total_executions"] == 3
        assert result["completed"] == 1
        assert result["pending"] == 2
        assert result["needs_manual_review"] is True
        assert len(result["issues"]) == 2
        
        # Check issue details
        issues = result["issues"]
        symbols = {issue["symbol"] for issue in issues}
        assert "MSFT" in symbols
        assert "GOOGL" in symbols
        
        # Verify different statuses
        statuses = {issue["status"] for issue in issues}
        assert "pending" in statuses
        assert "confirmed" in statuses
    
    def test_recovery_check_with_failures(self, temp_recovery_dir):
        """Test recovery check counts failed executions."""
        # Create some failed executions
        exec1 = execution_ledger.record_pending("AAPL", "BUY", 10, 180.0)
        execution_ledger.record_failed(exec1, "Broker API timeout")
        
        exec2 = execution_ledger.record_pending("MSFT", "BUY", 5, 380.0)
        execution_ledger.record_failed(exec2, "Insufficient buying power")
        
        # Run recovery check
        result = recovery_dt.run_recovery_check(alert_on_issues=False)
        
        # Failures don't need attention (already handled)
        assert result["status"] == "ok"
        assert result["failed"] == 2
        assert result["needs_manual_review"] is False


class TestPositionConsistency:
    """Test position consistency validation."""
    
    def test_validate_consistency_no_positions(self, temp_recovery_dir):
        """Test validation when no positions exist."""
        result = recovery_dt.validate_position_consistency()
        
        assert result["status"] == "ok"
        assert result["ledger_positions"] == 0
        assert result["actual_positions"] == 0
        assert len(result["discrepancies"]) == 0
    
    def test_validate_consistency_matching(self, temp_recovery_dir):
        """Test validation when positions match ledger."""
        # Create position file
        positions_file = temp_recovery_dir / "intraday" / "positions" / "dt_positions.json"
        positions_file.parent.mkdir(parents=True, exist_ok=True)
        
        positions = {
            "AAPL": {
                "status": "OPEN",
                "qty": 10.0,
                "entry_price": 180.0,
            }
        }
        positions_file.write_text(json.dumps(positions))
        
        # Create matching ledger entry (for today)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        exec1 = execution_ledger.record_pending("AAPL", "BUY", 10, 180.0)
        execution_ledger.record_confirmed(exec1, "b1", 180.0)
        execution_ledger.record_recorded(exec1, {"symbol": "AAPL"})
        
        # Validate
        result = recovery_dt.validate_position_consistency()
        
        # Should match
        assert result["status"] == "ok"
        assert len(result["discrepancies"]) == 0
    
    def test_validate_consistency_discrepancy(self, temp_recovery_dir):
        """Test validation detects position discrepancies."""
        # Create position file with one position
        positions_file = temp_recovery_dir / "intraday" / "positions" / "dt_positions.json"
        positions_file.parent.mkdir(parents=True, exist_ok=True)
        
        positions = {
            "AAPL": {
                "status": "OPEN",
                "qty": 15.0,  # Actual position
                "entry_price": 180.0,
            }
        }
        positions_file.write_text(json.dumps(positions))
        
        # Create ledger entry for today with different qty
        exec1 = execution_ledger.record_pending("AAPL", "BUY", 10, 180.0)
        execution_ledger.record_confirmed(exec1, "b1", 180.0)
        execution_ledger.record_recorded(exec1, {"symbol": "AAPL"})
        
        # Validate
        result = recovery_dt.validate_position_consistency()
        
        # Should detect discrepancy
        assert result["status"] == "discrepancies_found"
        assert len(result["discrepancies"]) == 1
        
        disc = result["discrepancies"][0]
        assert disc["symbol"] == "AAPL"
        assert disc["expected_qty"] == 10.0
        assert disc["actual_qty"] == 15.0
        assert disc["difference"] == 5.0


class TestReconciliation:
    """Test manual reconciliation operations."""
    
    def test_reconcile_mark_failed(self, temp_recovery_dir):
        """Test reconciling by marking as failed."""
        # Create a stuck execution
        exec_id = execution_ledger.record_pending("AAPL", "BUY", 10, 180.0)
        
        # Reconcile by marking as failed
        success = recovery_dt.reconcile_execution(exec_id, action="mark_failed")
        
        assert success is True
        
        # Verify it's no longer pending
        pending = execution_ledger.get_pending()
        assert len(pending) == 0
        
        # Verify it's marked as failed
        recent = execution_ledger.get_recent(limit=10)
        assert recent[0]["status"] == "failed"
        assert "Manually marked as failed" in recent[0]["error_msg"]
    
    def test_reconcile_mark_recorded(self, temp_recovery_dir):
        """Test reconciling by marking as recorded."""
        # Create a stuck execution (confirmed but not recorded)
        exec_id = execution_ledger.record_pending("AAPL", "BUY", 10, 180.0)
        execution_ledger.record_confirmed(exec_id, "b1", 180.0)
        
        # Reconcile by marking as recorded
        success = recovery_dt.reconcile_execution(exec_id, action="mark_recorded")
        
        assert success is True
        
        # Verify it's no longer pending
        pending = execution_ledger.get_pending()
        assert len(pending) == 0
        
        # Verify it's marked as recorded
        recent = execution_ledger.get_recent(limit=10)
        assert recent[0]["status"] == "recorded"
    
    def test_reconcile_invalid_action(self, temp_recovery_dir):
        """Test reconciling with invalid action."""
        exec_id = execution_ledger.record_pending("AAPL", "BUY", 10, 180.0)
        
        success = recovery_dt.reconcile_execution(exec_id, action="invalid_action")
        
        assert success is False


class TestRecoveryStatus:
    """Test recovery status monitoring."""
    
    def test_get_recovery_status_clean(self, temp_recovery_dir):
        """Test getting status when system is clean."""
        # Create completed executions
        exec1 = execution_ledger.record_pending("AAPL", "BUY", 10, 180.0)
        execution_ledger.record_confirmed(exec1, "b1", 180.0)
        execution_ledger.record_recorded(exec1, {"symbol": "AAPL"})
        
        # Get status
        status = recovery_dt.get_recovery_status()
        
        assert status["pending_executions"] == 0
        assert status["needs_attention"] is False
        assert len(status["pending_details"]) == 0
    
    def test_get_recovery_status_with_pending(self, temp_recovery_dir):
        """Test getting status when there are pending executions."""
        # Create pending executions
        exec1 = execution_ledger.record_pending("AAPL", "BUY", 10, 180.0)
        exec2 = execution_ledger.record_pending("MSFT", "BUY", 5, 380.0)
        execution_ledger.record_confirmed(exec2, "b2", 380.0)
        
        # Get status
        status = recovery_dt.get_recovery_status()
        
        assert status["pending_executions"] == 2
        assert status["needs_attention"] is True
        assert len(status["pending_details"]) == 2
        
        # Check details
        details = status["pending_details"]
        symbols = {d["symbol"] for d in details}
        assert "AAPL" in symbols
        assert "MSFT" in symbols


class TestRecoveryScenarios:
    """Test realistic recovery scenarios."""
    
    def test_scenario_system_restart_after_crash(self, temp_recovery_dir):
        """Test full recovery scenario after system crash."""
        # Simulate system running, creating executions
        exec1 = execution_ledger.record_pending("AAPL", "BUY", 10, 180.0)
        execution_ledger.record_confirmed(exec1, "b1", 180.0)
        execution_ledger.record_recorded(exec1, {"symbol": "AAPL"})
        
        exec2 = execution_ledger.record_pending("MSFT", "BUY", 5, 380.0)
        execution_ledger.record_confirmed(exec2, "b2", 380.0)
        # CRASH - didn't record!
        
        exec3 = execution_ledger.record_pending("GOOGL", "BUY", 2, 140.0)
        # CRASH - never sent to broker!
        
        # System restarts, run recovery check
        result = recovery_dt.run_recovery_check(alert_on_issues=False)
        
        # Should detect issues
        assert result["status"] == "needs_attention"
        assert result["completed"] == 1
        assert result["pending"] == 2
        
        # Operator reviews and reconciles
        # exec2 was confirmed - might have filled, mark as recorded
        recovery_dt.reconcile_execution(exec2, action="mark_recorded")
        
        # exec3 was only pending - safe to fail
        recovery_dt.reconcile_execution(exec3, action="mark_failed")
        
        # Check status after reconciliation
        status = recovery_dt.get_recovery_status()
        assert status["needs_attention"] is False
    
    def test_scenario_broker_api_failures(self, temp_recovery_dir):
        """Test recovery when broker API had failures."""
        # Simulate multiple broker failures
        for i in range(5):
            exec_id = execution_ledger.record_pending(
                symbol=f"SYM{i}",
                side="BUY",
                qty=10.0,
                price=100.0,
            )
            execution_ledger.record_failed(exec_id, f"Broker error {i}")
        
        # Run recovery
        result = recovery_dt.run_recovery_check(alert_on_issues=False)
        
        # Failures are recorded, no manual action needed
        assert result["status"] == "ok"
        assert result["failed"] == 5
        assert result["needs_manual_review"] is False
    
    def test_scenario_partial_fills(self, temp_recovery_dir):
        """Test recovery handles partial fills correctly."""
        # Simulate partial fill scenario
        exec_id = execution_ledger.record_pending("AAPL", "BUY", 100, 180.0)
        execution_ledger.record_confirmed(exec_id, "b1", 180.0)
        # Only 50 filled, recorded with actual qty
        execution_ledger.record_recorded(
            exec_id,
            position_state={"symbol": "AAPL", "qty": 50.0}
        )
        
        # Recovery should see this as completed (recorded)
        result = recovery_dt.run_recovery_check(alert_on_issues=False)
        
        assert result["status"] == "ok"
        assert result["completed"] == 1
        assert result["pending"] == 0


class TestIntegration:
    """Integration tests for recovery service."""
    
    def test_startup_sequence(self, temp_recovery_dir):
        """Test typical startup sequence."""
        # Simulate previous session left pending executions
        exec1 = execution_ledger.record_pending("AAPL", "BUY", 10, 180.0)
        exec2 = execution_ledger.record_pending("MSFT", "BUY", 5, 380.0)
        execution_ledger.record_confirmed(exec2, "b2", 380.0)
        
        # System starts up
        # Step 1: Run recovery check
        recovery_result = recovery_dt.run_recovery_check(alert_on_issues=False)
        assert recovery_result["needs_manual_review"] is True
        
        # Step 2: Validate position consistency
        consistency_result = recovery_dt.validate_position_consistency()
        # Should complete without error
        assert consistency_result["status"] in ["ok", "discrepancies_found"]
        
        # Step 3: Get detailed status
        status = recovery_dt.get_recovery_status()
        assert status["pending_executions"] == 2
        
        # Step 4: Operator reviews and reconciles
        for detail in status["pending_details"]:
            exec_id = detail["execution_id"]
            if detail["status"] == "confirmed":
                # Might have filled, mark as recorded
                recovery_dt.reconcile_execution(exec_id, action="mark_recorded")
            else:
                # Only pending, safe to fail
                recovery_dt.reconcile_execution(exec_id, action="mark_failed")
        
        # Step 5: Verify system is clean
        final_status = recovery_dt.get_recovery_status()
        assert final_status["needs_attention"] is False
