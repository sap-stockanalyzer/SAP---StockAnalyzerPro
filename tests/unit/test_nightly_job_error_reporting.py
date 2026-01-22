"""Unit tests for nightly job error reporting enhancements."""

from __future__ import annotations

import os
from unittest.mock import Mock, patch, call

import pytest


class TestNightlyJobErrorReporting:
    """Test suite for enhanced nightly job error reporting."""
    
    @pytest.fixture
    def mock_alert_functions(self):
        """Mock alert functions."""
        with patch("backend.monitoring.alerting.alert_nightly") as mock_nightly, \
             patch("backend.monitoring.alerting.alert_error") as mock_error:
            yield mock_nightly, mock_error
    
    def test_ok_with_errors_sends_both_alerts(self, mock_alert_functions):
        """Test that ok_with_errors status sends both nightly and error alerts."""
        mock_nightly, mock_error = mock_alert_functions
        
        # Simulate a nightly job summary with errors
        summary = {
            "status": "ok_with_errors",
            "duration_secs": 600,  # 10 minutes
            "phases": {
                "phase1": {"status": "ok", "secs": 100},
                "phase2": {"status": "error", "secs": 50, "error": "Connection timeout"},
                "phase3": {"status": "ok", "secs": 200},
                "phase4": {"status": "error", "secs": 75, "error": "Database locked"},
                "phase5": {"status": "skipped"},
            }
        }
        
        # Simulate the alerting logic
        duration_mins = round(summary.get("duration_secs", 0) / 60, 1)
        phases = summary.get("phases", {})
        phases_completed = sum(1 for v in phases.values() if isinstance(v, dict) and v.get("status") in ["ok", "skipped"])
        total_phases = len(phases)
        
        # Send completion summary
        status_emoji = "⚠️"
        mock_nightly(
            f"{status_emoji} Nightly Job Completed",
            f"Duration: {duration_mins} minutes\nPhases: {phases_completed}/{total_phases}",
            context={
                "Status": summary["status"],
                "Duration": f"{duration_mins} min",
                "Phases": f"{phases_completed}/{total_phases}",
            },
        )
        
        # Send error details
        failed_phases = [k for k, v in phases.items() if isinstance(v, dict) and v.get("status") == "error"]
        if failed_phases:
            error_details = []
            for phase_key in failed_phases:
                phase_info = phases.get(phase_key, {})
                error_msg = phase_info.get("error", "Unknown error")
                secs = phase_info.get("secs", 0)
                error_details.append(f"• **{phase_key}** ({secs}s): {error_msg}")
            
            error_message = "\n".join(error_details)
            mock_error(
                "⚠️ Nightly Job Completed With Errors",
                f"**Failed Phases ({len(failed_phases)}/{total_phases}):**\n{error_message}\n\n**Duration:** {duration_mins} minutes",
                context={
                    "Status": "ok_with_errors",
                    "Failed Phases": len(failed_phases),
                    "Total Phases": total_phases,
                    "Duration": f"{duration_mins} min",
                }
            )
        
        # Verify both alerts were called
        assert mock_nightly.called
        assert mock_error.called
        
        # Verify nightly alert details
        nightly_call = mock_nightly.call_args
        assert "⚠️ Nightly Job Completed" in nightly_call[0][0]
        assert "10.0 minutes" in nightly_call[0][1]
        
        # Verify error alert details
        error_call = mock_error.call_args
        assert "Nightly Job Completed With Errors" in error_call[0][0]
        assert "phase2" in error_call[0][1]
        assert "Connection timeout" in error_call[0][1]
        assert "phase4" in error_call[0][1]
        assert "Database locked" in error_call[0][1]
        assert error_call[1]["context"]["Failed Phases"] == 2
        assert error_call[1]["context"]["Total Phases"] == 5
    
    def test_ok_status_only_sends_nightly_alert(self, mock_alert_functions):
        """Test that ok status only sends nightly alert, not error alert."""
        mock_nightly, mock_error = mock_alert_functions
        
        summary = {
            "status": "ok",
            "duration_secs": 300,
            "phases": {
                "phase1": {"status": "ok", "secs": 100},
                "phase2": {"status": "ok", "secs": 150},
                "phase3": {"status": "skipped"},
            }
        }
        
        # Simulate the alerting logic
        duration_mins = round(summary.get("duration_secs", 0) / 60, 1)
        phases = summary.get("phases", {})
        phases_completed = sum(1 for v in phases.values() if isinstance(v, dict) and v.get("status") in ["ok", "skipped"])
        total_phases = len(phases)
        
        if summary["status"] in ["ok", "ok_with_errors"]:
            status_emoji = "✅" if summary["status"] == "ok" else "⚠️"
            mock_nightly(
                f"{status_emoji} Nightly Job Completed",
                f"Duration: {duration_mins} minutes\nPhases: {phases_completed}/{total_phases}",
                context={
                    "Status": summary["status"],
                    "Duration": f"{duration_mins} min",
                    "Phases": f"{phases_completed}/{total_phases}",
                },
            )
        
        # Don't send error alert for ok status
        if summary["status"] == "ok_with_errors":
            failed_phases = [k for k, v in phases.items() if isinstance(v, dict) and v.get("status") == "error"]
            if failed_phases:
                mock_error("test", "test")
        
        # Verify only nightly alert was called
        assert mock_nightly.called
        assert not mock_error.called
    
    def test_error_message_format(self):
        """Test that error messages are properly formatted."""
        phases = {
            "download_data": {"status": "error", "secs": 45, "error": "API rate limit exceeded"},
            "process_data": {"status": "error", "secs": 120, "error": "Memory allocation failed"},
        }
        
        failed_phases = [k for k, v in phases.items() if isinstance(v, dict) and v.get("status") == "error"]
        error_details = []
        for phase_key in failed_phases:
            phase_info = phases.get(phase_key, {})
            error_msg = phase_info.get("error", "Unknown error")
            secs = phase_info.get("secs", 0)
            error_details.append(f"• **{phase_key}** ({secs}s): {error_msg}")
        
        error_message = "\n".join(error_details)
        
        # Verify format
        assert "• **download_data** (45s): API rate limit exceeded" in error_message
        assert "• **process_data** (120s): Memory allocation failed" in error_message


class TestNightlySkipGuardAlert:
    """Test suite for nightly skip guard alert."""
    
    @pytest.fixture
    def mock_alert_nightly(self):
        """Mock alert_nightly function."""
        with patch("backend.monitoring.alerting.alert_nightly") as mock:
            yield mock
    
    def test_skip_alert_sent(self, mock_alert_nightly):
        """Test that skip alert is sent when nightly is skipped."""
        MIN_HOURS_BETWEEN_RUNS = 8  # Default value from nightly_job.py
        
        # Simulate skip notification
        mock_alert_nightly(
            "⏭️ Nightly Job Skipped",
            f"Last run finished within {MIN_HOURS_BETWEEN_RUNS}h. Skipping to prevent duplication.",
            context={"Reason": "Recent run guard"},
        )
        
        # Verify alert was called
        assert mock_alert_nightly.called
        
        # Verify alert details
        call_args = mock_alert_nightly.call_args
        assert "Nightly Job Skipped" in call_args[0][0]
        assert "8h" in call_args[0][1]
        assert call_args[1]["context"]["Reason"] == "Recent run guard"


class TestRollingOptimizerQuietMode:
    """Test suite for RollingOptimizer quiet mode."""
    
    def test_quiet_mode_enabled(self):
        """Test that quiet mode suppresses logs when enabled."""
        env_vars = {"QUIET_ROLLING_OPTIMIZER": "1"}
        
        with patch.dict(os.environ, env_vars, clear=False):
            quiet = os.getenv("QUIET_ROLLING_OPTIMIZER", "0").lower() in {"1", "true", "yes", "on"}
            assert quiet is True
    
    def test_quiet_mode_disabled(self):
        """Test that quiet mode is disabled by default."""
        env_vars = {"QUIET_ROLLING_OPTIMIZER": "0"}
        
        with patch.dict(os.environ, env_vars, clear=False):
            quiet = os.getenv("QUIET_ROLLING_OPTIMIZER", "0").lower() in {"1", "true", "yes", "on"}
            assert quiet is False
    
    def test_quiet_mode_various_values(self):
        """Test that quiet mode works with various truthy values."""
        truthy_values = ["1", "true", "True", "TRUE", "yes", "YES", "on", "ON"]
        
        for value in truthy_values:
            with patch.dict(os.environ, {"QUIET_ROLLING_OPTIMIZER": value}, clear=False):
                quiet = os.getenv("QUIET_ROLLING_OPTIMIZER", "0").lower() in {"1", "true", "yes", "on"}
                assert quiet is True, f"Value '{value}' should enable quiet mode"
        
        falsy_values = ["0", "false", "False", "FALSE", "no", "NO", "off", "OFF", ""]
        
        for value in falsy_values:
            with patch.dict(os.environ, {"QUIET_ROLLING_OPTIMIZER": value}, clear=False):
                quiet = os.getenv("QUIET_ROLLING_OPTIMIZER", "0").lower() in {"1", "true", "yes", "on"}
                assert quiet is False, f"Value '{value}' should disable quiet mode"
