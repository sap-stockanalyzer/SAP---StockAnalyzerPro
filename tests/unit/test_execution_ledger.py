"""Tests for dt_backend/services/execution_ledger.py

Tests the saga pattern execution ledger (3-phase commit).
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dt_backend.services import execution_ledger


@pytest.fixture
def temp_ledger_dir(tmp_path, monkeypatch):
    """Set up temporary ledger directory."""
    ledger_dir = tmp_path / "intraday"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    
    # Override the ledger path
    monkeypatch.setenv("DT_TRUTH_DIR", str(tmp_path))
    
    yield ledger_dir
    
    # Cleanup is automatic with tmp_path


class TestExecutionLedger:
    """Test basic execution ledger operations."""
    
    def test_record_pending(self, temp_ledger_dir):
        """Test Phase 1: recording pending execution."""
        exec_id = execution_ledger.record_pending(
            symbol="AAPL",
            side="BUY",
            qty=10.0,
            price=180.0,
            bot="ORB",
            confidence=0.75,
        )
        
        assert exec_id is not None
        assert exec_id.startswith("exec_")
        
        # Verify ledger entry
        ledger_file = temp_ledger_dir / "dt_execution_ledger.jsonl"
        assert ledger_file.exists()
        
        lines = ledger_file.read_text().strip().split("\n")
        assert len(lines) == 1
        
        record = json.loads(lines[0])
        assert record["execution_id"] == exec_id
        assert record["symbol"] == "AAPL"
        assert record["side"] == "BUY"
        assert record["qty"] == 10.0
        assert record["price"] == 180.0
        assert record["status"] == "pending"
        assert record["bot"] == "ORB"
        assert record["confidence"] == 0.75
    
    def test_record_confirmed(self, temp_ledger_dir):
        """Test Phase 2: recording confirmed fill."""
        # First, record pending
        exec_id = execution_ledger.record_pending(
            symbol="AAPL",
            side="BUY",
            qty=10.0,
            price=180.0,
        )
        
        # Then confirm
        success = execution_ledger.record_confirmed(
            execution_id=exec_id,
            broker_order_id="broker_12345",
            fill_price=180.25,
        )
        
        assert success is True
        
        # Verify ledger has 2 entries
        ledger_file = temp_ledger_dir / "dt_execution_ledger.jsonl"
        lines = ledger_file.read_text().strip().split("\n")
        assert len(lines) == 2
        
        # Check confirmed record
        confirmed = json.loads(lines[1])
        assert confirmed["execution_id"] == exec_id
        assert confirmed["status"] == "confirmed"
        assert confirmed["broker_order_id"] == "broker_12345"
        assert confirmed["fill_price"] == 180.25
        assert "fill_ts" in confirmed
    
    def test_record_recorded(self, temp_ledger_dir):
        """Test Phase 3: recording final state."""
        # Phase 1: Pending
        exec_id = execution_ledger.record_pending(
            symbol="AAPL",
            side="BUY",
            qty=10.0,
            price=180.0,
        )
        
        # Phase 2: Confirmed
        execution_ledger.record_confirmed(
            execution_id=exec_id,
            broker_order_id="broker_12345",
            fill_price=180.25,
        )
        
        # Phase 3: Recorded
        position_state = {
            "symbol": "AAPL",
            "qty": 10.0,
            "entry_price": 180.25,
            "status": "OPEN",
        }
        success = execution_ledger.record_recorded(
            execution_id=exec_id,
            position_state=position_state,
        )
        
        assert success is True
        
        # Verify ledger has 3 entries
        ledger_file = temp_ledger_dir / "dt_execution_ledger.jsonl"
        lines = ledger_file.read_text().strip().split("\n")
        assert len(lines) == 3
        
        # Check recorded entry
        recorded = json.loads(lines[2])
        assert recorded["execution_id"] == exec_id
        assert recorded["status"] == "recorded"
        assert "recorded_ts" in recorded
        assert recorded["position_state"] == position_state
    
    def test_record_failed(self, temp_ledger_dir):
        """Test recording failed execution."""
        # Record pending
        exec_id = execution_ledger.record_pending(
            symbol="AAPL",
            side="BUY",
            qty=10.0,
            price=180.0,
        )
        
        # Mark as failed
        success = execution_ledger.record_failed(
            execution_id=exec_id,
            error_msg="Broker API timeout",
        )
        
        assert success is True
        
        # Verify ledger
        ledger_file = temp_ledger_dir / "dt_execution_ledger.jsonl"
        lines = ledger_file.read_text().strip().split("\n")
        assert len(lines) == 2
        
        failed = json.loads(lines[1])
        assert failed["execution_id"] == exec_id
        assert failed["status"] == "failed"
        assert failed["error_msg"] == "Broker API timeout"
        assert "error_ts" in failed
    
    def test_full_saga_success(self, temp_ledger_dir):
        """Test complete successful saga flow."""
        # Phase 1
        exec_id = execution_ledger.record_pending(
            symbol="AAPL",
            side="BUY",
            qty=10.0,
            price=180.0,
            bot="ORB",
            confidence=0.8,
            meta={"regime": "bull", "time_window": "morning"},
        )
        
        # Phase 2
        execution_ledger.record_confirmed(
            execution_id=exec_id,
            broker_order_id="broker_abc123",
            fill_price=180.15,
        )
        
        # Phase 3
        execution_ledger.record_recorded(
            execution_id=exec_id,
            position_state={
                "symbol": "AAPL",
                "qty": 10.0,
                "entry_price": 180.15,
            },
        )
        
        # Verify complete saga
        ledger_file = temp_ledger_dir / "dt_execution_ledger.jsonl"
        lines = ledger_file.read_text().strip().split("\n")
        assert len(lines) == 3
        
        statuses = [json.loads(line)["status"] for line in lines]
        assert statuses == ["pending", "confirmed", "recorded"]


class TestLedgerQueries:
    """Test ledger query functions."""
    
    def test_get_pending(self, temp_ledger_dir):
        """Test getting pending executions."""
        # Create some executions
        exec1 = execution_ledger.record_pending("AAPL", "BUY", 10, 180.0)
        exec2 = execution_ledger.record_pending("MSFT", "BUY", 5, 380.0)
        exec3 = execution_ledger.record_pending("GOOGL", "BUY", 2, 140.0)
        
        # Complete one
        execution_ledger.record_confirmed(exec1, "broker_1", 180.0)
        execution_ledger.record_recorded(exec1, {"symbol": "AAPL"})
        
        # Fail one
        execution_ledger.record_failed(exec2, "Timeout")
        
        # Get pending (should only be exec3)
        pending = execution_ledger.get_pending()
        
        assert len(pending) == 1
        assert pending[0]["execution_id"] == exec3
        assert pending[0]["status"] == "pending"
    
    def test_get_recent(self, temp_ledger_dir):
        """Test getting recent executions."""
        # Create several executions
        for i in range(5):
            exec_id = execution_ledger.record_pending(
                symbol=f"SYM{i}",
                side="BUY",
                qty=10.0,
                price=100.0 + i,
            )
            
            if i % 2 == 0:
                # Complete even-numbered ones
                execution_ledger.record_confirmed(exec_id, f"broker_{i}", 100.0 + i)
                execution_ledger.record_recorded(exec_id, {"symbol": f"SYM{i}"})
        
        # Get recent
        recent = execution_ledger.get_recent(limit=10)
        
        # Should return latest state for each execution
        assert len(recent) == 5
        
        # Check that we have the right mix of statuses
        statuses = [r["status"] for r in recent]
        assert "recorded" in statuses
        assert "pending" in statuses
    
    def test_replay_ledger(self, temp_ledger_dir):
        """Test ledger replay for recovery."""
        # Create some executions in various states
        exec1 = execution_ledger.record_pending("AAPL", "BUY", 10, 180.0)
        execution_ledger.record_confirmed(exec1, "b1", 180.0)
        execution_ledger.record_recorded(exec1, {"symbol": "AAPL"})
        
        exec2 = execution_ledger.record_pending("MSFT", "BUY", 5, 380.0)
        execution_ledger.record_failed(exec2, "Error")
        
        exec3 = execution_ledger.record_pending("GOOGL", "BUY", 2, 140.0)
        # Leave pending
        
        exec4 = execution_ledger.record_pending("TSLA", "BUY", 3, 200.0)
        execution_ledger.record_confirmed(exec4, "b2", 200.0)
        # Leave confirmed (not recorded)
        
        # Replay
        result = execution_ledger.replay_ledger()
        
        assert result["status"] == "ok"
        assert result["total"] == 4
        assert result["completed"] == 1  # Only exec1
        assert result["failed"] == 1  # Only exec2
        assert result["pending"] == 2  # exec3 (pending) and exec4 (confirmed)
        
        # Check incomplete list
        incomplete = result["incomplete"]
        assert len(incomplete) == 2
        
        incomplete_ids = {r["execution_id"] for r in incomplete}
        assert exec3 in incomplete_ids
        assert exec4 in incomplete_ids


class TestErrorHandling:
    """Test error handling in execution ledger."""
    
    def test_confirm_unknown_execution(self, temp_ledger_dir):
        """Test confirming an unknown execution ID."""
        success = execution_ledger.record_confirmed(
            execution_id="exec_nonexistent",
            broker_order_id="broker_123",
            fill_price=100.0,
        )
        
        assert success is False
    
    def test_record_unknown_execution(self, temp_ledger_dir):
        """Test recording an unknown execution ID."""
        success = execution_ledger.record_recorded(
            execution_id="exec_nonexistent",
            position_state={"symbol": "AAPL"},
        )
        
        assert success is False
    
    def test_fail_unknown_execution(self, temp_ledger_dir):
        """Test failing an unknown execution ID."""
        success = execution_ledger.record_failed(
            execution_id="exec_nonexistent",
            error_msg="Error",
        )
        
        assert success is False
    
    def test_ledger_file_locking(self, temp_ledger_dir):
        """Test that ledger operations use file locking."""
        # Create multiple concurrent pending records
        from concurrent.futures import ThreadPoolExecutor
        
        def create_pending(idx):
            return execution_ledger.record_pending(
                symbol=f"SYM{idx}",
                side="BUY",
                qty=10.0,
                price=100.0,
            )
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            exec_ids = list(executor.map(create_pending, range(10)))
        
        # All should succeed
        assert len(exec_ids) == 10
        assert all(exec_id is not None for exec_id in exec_ids)
        
        # Verify ledger integrity
        ledger_file = temp_ledger_dir / "dt_execution_ledger.jsonl"
        lines = ledger_file.read_text().strip().split("\n")
        assert len(lines) == 10
        
        # All should be valid JSON
        for line in lines:
            record = json.loads(line)
            assert "execution_id" in record


class TestSagaScenarios:
    """Test realistic saga pattern scenarios."""
    
    def test_broker_timeout_scenario(self, temp_ledger_dir):
        """Test handling broker API timeout."""
        # Phase 1: Record intent
        exec_id = execution_ledger.record_pending(
            symbol="AAPL",
            side="BUY",
            qty=10.0,
            price=180.0,
        )
        
        # Broker API times out (no phase 2)
        # Record failure
        execution_ledger.record_failed(exec_id, "Broker API timeout")
        
        # Verify state
        pending = execution_ledger.get_pending()
        assert len(pending) == 0  # Not pending anymore
        
        recent = execution_ledger.get_recent(limit=10)
        failed = [r for r in recent if r["status"] == "failed"]
        assert len(failed) == 1
        assert failed[0]["execution_id"] == exec_id
    
    def test_partial_fill_scenario(self, temp_ledger_dir):
        """Test handling partial fill."""
        # Phase 1: Request 100 shares
        exec_id = execution_ledger.record_pending(
            symbol="AAPL",
            side="BUY",
            qty=100.0,
            price=180.0,
        )
        
        # Phase 2: Only 50 filled
        execution_ledger.record_confirmed(
            execution_id=exec_id,
            broker_order_id="broker_123",
            fill_price=180.15,
        )
        
        # Phase 3: Record what we got
        execution_ledger.record_recorded(
            execution_id=exec_id,
            position_state={"symbol": "AAPL", "qty": 50.0},  # Actual fill
        )
        
        # Verify
        recent = execution_ledger.get_recent(limit=10)
        assert len(recent) == 1
        assert recent[0]["status"] == "recorded"
        assert recent[0]["qty"] == 100.0  # Original intent
        assert recent[0]["position_state"]["qty"] == 50.0  # Actual fill
    
    def test_system_crash_recovery(self, temp_ledger_dir):
        """Test recovery after system crash."""
        # Simulate: system starts executing
        exec1 = execution_ledger.record_pending("AAPL", "BUY", 10, 180.0)
        exec2 = execution_ledger.record_pending("MSFT", "BUY", 5, 380.0)
        
        # exec1 gets confirmed
        execution_ledger.record_confirmed(exec1, "b1", 180.0)
        
        # CRASH! (system goes down before recording)
        # ...
        
        # System restarts, replay ledger
        replay = execution_ledger.replay_ledger()
        
        # Should detect 2 incomplete executions
        assert replay["pending"] == 2
        assert len(replay["incomplete"]) == 2
        
        # Operator/system can now review and reconcile
        incomplete = replay["incomplete"]
        
        # exec1 was confirmed, might have position
        confirmed_exec = [e for e in incomplete if e["status"] == "confirmed"][0]
        assert confirmed_exec["execution_id"] == exec1
        
        # exec2 was only pending, safe to fail
        pending_exec = [e for e in incomplete if e["status"] == "pending"][0]
        assert pending_exec["execution_id"] == exec2
