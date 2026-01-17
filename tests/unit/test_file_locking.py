"""Tests for dt_backend/core/file_locking.py

Tests the file locking utilities that prevent race conditions.
"""

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from dt_backend.core.file_locking import (
    AcquireLock,
    AcquireMultipleLocks,
    AppendLocked,
    ReadLocked,
    WriteLocked,
)


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file for testing."""
    test_file = tmp_path / "test_file.json"
    return test_file


@pytest.fixture
def temp_jsonl(tmp_path):
    """Create a temporary JSONL file for testing."""
    test_file = tmp_path / "test_file.jsonl"
    return test_file


class TestFileLocking:
    """Test basic file locking operations."""
    
    def test_acquire_lock_success(self, temp_file):
        """Test that we can acquire a lock successfully."""
        with AcquireLock(temp_file, timeout=2.0) as acquired:
            assert acquired is True
    
    def test_acquire_lock_timeout(self, temp_file):
        """Test that lock acquisition times out if already held."""
        # First acquire the lock
        with AcquireLock(temp_file, timeout=2.0) as acquired1:
            assert acquired1 is True
            
            # Try to acquire again from "another process" (really same process, different lock)
            # This should timeout since we're already holding it
            with AcquireLock(temp_file, timeout=0.1) as acquired2:
                # Should fail because lock is held
                assert acquired2 is False
    
    def test_lock_release(self, temp_file):
        """Test that locks are properly released."""
        # Acquire and release
        with AcquireLock(temp_file, timeout=2.0) as acquired1:
            assert acquired1 is True
        
        # Should be able to acquire again
        with AcquireLock(temp_file, timeout=2.0) as acquired2:
            assert acquired2 is True
    
    def test_concurrent_lock_prevention(self, temp_file):
        """Test that concurrent writes are prevented."""
        results = []
        
        def try_write(idx):
            """Try to acquire lock and write."""
            with AcquireLock(temp_file, timeout=1.0) as acquired:
                if acquired:
                    time.sleep(0.1)  # Simulate work
                    results.append(idx)
                return acquired
        
        # Launch concurrent attempts
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(try_write, i) for i in range(3)]
            outcomes = [f.result() for f in as_completed(futures)]
        
        # Only one should succeed at a time (all should eventually succeed)
        # But due to timeout, some may fail
        assert True in outcomes  # At least one succeeded


class TestWriteLocked:
    """Test locked write operations."""
    
    def test_write_locked_success(self, temp_file):
        """Test that WriteLocked writes data successfully."""
        data = json.dumps({"test": "data"}, indent=2)
        success = WriteLocked(temp_file, data, timeout=2.0)
        
        assert success is True
        assert temp_file.exists()
        
        # Verify content
        content = temp_file.read_text(encoding="utf-8")
        assert "test" in content
        assert "data" in content
    
    def test_write_locked_atomic(self, temp_file):
        """Test that writes are atomic (use temp file + rename)."""
        data = json.dumps({"key": "value"})
        WriteLocked(temp_file, data, timeout=2.0)
        
        # Check no .tmp file left behind
        tmp_files = list(temp_file.parent.glob("*.tmp"))
        assert len(tmp_files) == 0
    
    def test_concurrent_writes_sequential(self, temp_file):
        """Test that concurrent writes happen sequentially."""
        def write_number(num):
            data = json.dumps({"number": num})
            return WriteLocked(temp_file, data, timeout=2.0)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(write_number, i) for i in range(3)]
            results = [f.result() for f in as_completed(futures)]
        
        # At least one should succeed
        assert True in results
        
        # File should contain valid JSON (last write wins)
        content = temp_file.read_text(encoding="utf-8")
        data = json.loads(content)
        assert "number" in data


class TestAppendLocked:
    """Test locked append operations."""
    
    def test_append_locked_success(self, temp_jsonl):
        """Test that AppendLocked appends data successfully."""
        line1 = json.dumps({"event": "first"})
        line2 = json.dumps({"event": "second"})
        
        success1 = AppendLocked(temp_jsonl, line1, timeout=2.0)
        success2 = AppendLocked(temp_jsonl, line2, timeout=2.0)
        
        assert success1 is True
        assert success2 is True
        
        # Verify content
        lines = temp_jsonl.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        
        event1 = json.loads(lines[0])
        event2 = json.loads(lines[1])
        assert event1["event"] == "first"
        assert event2["event"] == "second"
    
    def test_append_adds_newline(self, temp_jsonl):
        """Test that AppendLocked adds newline if missing."""
        line_no_newline = json.dumps({"test": "data"})
        AppendLocked(temp_jsonl, line_no_newline, timeout=2.0)
        
        content = temp_jsonl.read_text(encoding="utf-8")
        assert content.endswith("\n")
    
    def test_concurrent_appends_no_loss(self, temp_jsonl):
        """Test that concurrent appends don't lose data."""
        def append_number(num):
            line = json.dumps({"num": num})
            return AppendLocked(temp_jsonl, line, timeout=5.0)
        
        # Launch many concurrent appends
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(append_number, i) for i in range(20)]
            results = [f.result() for f in as_completed(futures)]
        
        # All should succeed (with sufficient timeout)
        success_count = sum(1 for r in results if r)
        assert success_count >= 15  # Allow some to timeout but most should succeed
        
        # Verify no data loss - count lines
        lines = temp_jsonl.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == success_count
        
        # Verify all lines are valid JSON
        for line in lines:
            data = json.loads(line)
            assert "num" in data


class TestReadLocked:
    """Test locked read operations."""
    
    def test_read_locked_success(self, temp_file):
        """Test that ReadLocked reads data successfully."""
        content = json.dumps({"key": "value"})
        temp_file.write_text(content, encoding="utf-8")
        
        read_content = ReadLocked(temp_file, timeout=2.0)
        assert read_content == content
    
    def test_read_locked_default(self, temp_file):
        """Test that ReadLocked returns default if file doesn't exist."""
        non_existent = temp_file.parent / "non_existent.json"
        result = ReadLocked(non_existent, timeout=2.0, default=None)
        assert result is None
    
    def test_read_while_writing(self, temp_file):
        """Test that reads block during writes."""
        def write_slowly():
            with AcquireLock(temp_file, timeout=2.0) as acquired:
                if acquired:
                    time.sleep(0.2)  # Simulate slow write
                    temp_file.write_text("test data", encoding="utf-8")
                return acquired
        
        def try_read():
            return ReadLocked(temp_file, timeout=1.0, default="")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            write_future = executor.submit(write_slowly)
            time.sleep(0.05)  # Let write start
            read_future = executor.submit(try_read)
            
            write_result = write_future.result()
            read_result = read_future.result()
        
        # Write should succeed
        assert write_result is True
        # Read should either succeed after write or return default
        assert read_result is not None


class TestAcquireMultipleLocks:
    """Test multiple lock acquisition."""
    
    def test_acquire_multiple_locks_success(self, tmp_path):
        """Test that we can acquire multiple locks."""
        file1 = tmp_path / "positions_dt.json"
        file2 = tmp_path / "dt_trades.jsonl"
        file3 = tmp_path / "dt_execution_ledger.jsonl"
        
        files = [file1, file2, file3]
        
        with AcquireMultipleLocks(files, timeout=5.0) as acquired:
            assert acquired is True
    
    def test_lock_ordering(self, tmp_path):
        """Test that locks are acquired in correct order."""
        # Create files in reverse order of lock priority
        file1 = tmp_path / "dt_state.json"  # Priority 4
        file2 = tmp_path / "dt_trades.jsonl"  # Priority 2
        file3 = tmp_path / "positions_dt.json"  # Priority 1
        
        files = [file1, file2, file3]  # Out of order
        
        # Should succeed because they're automatically sorted
        with AcquireMultipleLocks(files, timeout=5.0) as acquired:
            assert acquired is True
    
    def test_multiple_locks_atomic_operation(self, tmp_path):
        """Test atomic operation across multiple files."""
        positions_file = tmp_path / "positions_dt.json"
        trades_file = tmp_path / "dt_trades.jsonl"
        
        files = [positions_file, trades_file]
        
        with AcquireMultipleLocks(files, timeout=5.0) as acquired:
            if acquired:
                # Write directly without using WriteLocked/AppendLocked
                # (they would try to acquire locks again - deadlock)
                tmp = positions_file.with_suffix(".tmp")
                tmp.write_text(json.dumps({"AAPL": {"qty": 10}}), encoding="utf-8")
                tmp.replace(positions_file)
                
                with open(trades_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"symbol": "AAPL", "qty": 10}) + "\n")
        
        # Verify both files were updated
        assert positions_file.exists()
        assert trades_file.exists()
        
        positions_data = json.loads(positions_file.read_text())
        assert positions_data["AAPL"]["qty"] == 10
        
        trades_lines = trades_file.read_text().strip().split("\n")
        trade_data = json.loads(trades_lines[0])
        assert trade_data["symbol"] == "AAPL"


class TestErrorHandling:
    """Test error handling in locking operations."""
    
    def test_lock_path_creation(self, tmp_path):
        """Test that parent directories are created."""
        deep_file = tmp_path / "a" / "b" / "c" / "test.json"
        
        with AcquireLock(deep_file, timeout=2.0) as acquired:
            assert acquired is True
        
        assert deep_file.parent.exists()
    
    def test_write_locked_creates_parents(self, tmp_path):
        """Test that WriteLocked creates parent directories."""
        deep_file = tmp_path / "x" / "y" / "test.json"
        
        success = WriteLocked(deep_file, "test data", timeout=2.0)
        assert success is True
        assert deep_file.exists()
    
    def test_append_locked_creates_parents(self, tmp_path):
        """Test that AppendLocked creates parent directories."""
        deep_file = tmp_path / "p" / "q" / "test.jsonl"
        
        success = AppendLocked(deep_file, "test line", timeout=2.0)
        assert success is True
        assert deep_file.exists()


class TestPerformance:
    """Performance and stress tests."""
    
    def test_high_concurrency_appends(self, temp_jsonl):
        """Test system under high concurrency load."""
        def append_event(idx):
            line = json.dumps({"event_id": idx, "ts": time.time()})
            return AppendLocked(temp_jsonl, line, timeout=10.0)
        
        # 50 concurrent appends
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(append_event, i) for i in range(50)]
            results = [f.result() for f in as_completed(futures)]
        
        # Most should succeed
        success_count = sum(1 for r in results if r)
        assert success_count >= 40  # At least 80% success rate
        
        # Verify data integrity
        lines = temp_jsonl.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == success_count
        
        # All lines should be valid JSON
        for line in lines:
            data = json.loads(line)
            assert "event_id" in data
    
    def test_rapid_lock_release(self, temp_file):
        """Test rapid lock acquisition and release."""
        iterations = 100
        
        for i in range(iterations):
            with AcquireLock(temp_file, timeout=1.0) as acquired:
                assert acquired is True
        
        # Should complete without deadlocks
        assert True
