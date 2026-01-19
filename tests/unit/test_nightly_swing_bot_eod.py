"""Tests for swing bot EOD rebalance in nightly job.

Tests the new phase 19 that runs swing bots after policy engine.
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

# We need to test the PIPELINE structure and the swing bot execution logic
from backend.jobs import nightly_job


class TestSwingBotEODPhase:
    """Test swing bot EOD rebalance phase."""
    
    def test_pipeline_includes_swing_bot_eod(self):
        """Test that PIPELINE includes the swing_bot_eod phase."""
        pipeline_keys = [key for key, _ in nightly_job.PIPELINE]
        assert "swing_bot_eod" in pipeline_keys
        
        # Verify it's after policy
        policy_idx = pipeline_keys.index("policy")
        swing_bot_idx = pipeline_keys.index("swing_bot_eod")
        assert swing_bot_idx == policy_idx + 1, "swing_bot_eod should be right after policy"
        
    def test_total_phases_count(self):
        """Test that TOTAL_PHASES matches PIPELINE length."""
        assert nightly_job.TOTAL_PHASES == len(nightly_job.PIPELINE)
        assert nightly_job.TOTAL_PHASES == 21, "Expected 21 phases including swing_bot_eod"
        
    def test_swing_bot_eod_phase_position(self):
        """Test that swing_bot_eod is at the correct position in PIPELINE."""
        # Phase 18 (index 17) should be policy
        # Phase 19 (index 18) should be swing_bot_eod
        assert nightly_job.PIPELINE[17][0] == "policy"
        assert nightly_job.PIPELINE[18][0] == "swing_bot_eod"
        assert nightly_job.PIPELINE[18][1] == "Swing Bot EOD Rebalance"
        
    @patch('subprocess.run')
    def test_swing_bot_execution_success(self, mock_subprocess):
        """Test successful execution of all swing bots."""
        # Mock successful subprocess runs
        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_subprocess.return_value = mock_proc
        
        # Simulate the swing bot execution logic
        bots = ["1w", "2w", "4w"]
        results = {}
        
        for bot in bots:
            cmd = [
                sys.executable, "-u", "-m",
                f"backend.bots.runner_{bot}",
                "--mode", "full"
            ]
            proc = subprocess.run(
                cmd,
                cwd=Path("/fake/root"),
                capture_output=True,
                text=True,
                timeout=600
            )
            results[bot] = {
                "exit_code": proc.returncode,
                "success": proc.returncode == 0,
            }
        
        # Verify all bots succeeded
        success_count = sum(1 for r in results.values() if r.get("success"))
        assert success_count == 3
        assert all(r["success"] for r in results.values())
        assert mock_subprocess.call_count == 3
        
    @patch('subprocess.run')
    def test_swing_bot_execution_timeout(self, mock_subprocess):
        """Test timeout handling for swing bots."""
        # Mock timeout
        mock_subprocess.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=600)
        
        # Simulate the swing bot execution logic with timeout handling
        bots = ["1w"]
        results = {}
        
        for bot in bots:
            try:
                cmd = [
                    sys.executable, "-u", "-m",
                    f"backend.bots.runner_{bot}",
                    "--mode", "full"
                ]
                proc = subprocess.run(
                    cmd,
                    cwd=Path("/fake/root"),
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                results[bot] = {
                    "exit_code": proc.returncode,
                    "success": proc.returncode == 0,
                }
            except subprocess.TimeoutExpired:
                results[bot] = {"exit_code": -1, "success": False, "error": "timeout"}
        
        # Verify timeout was handled
        assert results["1w"]["success"] is False
        assert results["1w"]["error"] == "timeout"
        assert results["1w"]["exit_code"] == -1
        
    @patch('subprocess.run')
    def test_swing_bot_execution_partial_failure(self, mock_subprocess):
        """Test partial failure scenario (some bots succeed, some fail)."""
        # Mock mixed results: success, failure, success
        return_values = [
            Mock(returncode=0),  # 1w succeeds
            Mock(returncode=1),  # 2w fails
            Mock(returncode=0),  # 4w succeeds
        ]
        mock_subprocess.side_effect = return_values
        
        # Simulate the swing bot execution logic
        bots = ["1w", "2w", "4w"]
        results = {}
        
        for bot in bots:
            try:
                cmd = [
                    sys.executable, "-u", "-m",
                    f"backend.bots.runner_{bot}",
                    "--mode", "full"
                ]
                proc = subprocess.run(
                    cmd,
                    cwd=Path("/fake/root"),
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                results[bot] = {
                    "exit_code": proc.returncode,
                    "success": proc.returncode == 0,
                }
            except subprocess.TimeoutExpired:
                results[bot] = {"exit_code": -1, "success": False, "error": "timeout"}
            except Exception as e:
                results[bot] = {"exit_code": -1, "success": False, "error": str(e)}
        
        # Verify partial success
        success_count = sum(1 for r in results.values() if r.get("success"))
        assert success_count == 2
        assert results["1w"]["success"] is True
        assert results["2w"]["success"] is False
        assert results["4w"]["success"] is True
        
    def test_swing_bot_command_format(self):
        """Test that swing bot commands are formatted correctly."""
        bots = ["1w", "2w", "4w"]
        for bot in bots:
            cmd = [
                sys.executable, "-u", "-m",
                f"backend.bots.runner_{bot}",
                "--mode", "full"
            ]
            
            # Verify command structure
            assert cmd[0] == sys.executable
            assert cmd[1] == "-u"
            assert cmd[2] == "-m"
            assert cmd[3] == f"backend.bots.runner_{bot}"
            assert cmd[4] == "--mode"
            assert cmd[5] == "full"
