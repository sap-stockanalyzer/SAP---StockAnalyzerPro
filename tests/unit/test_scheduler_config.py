"""
Test scheduler configuration to ensure all swing bot jobs are properly configured.
"""

import pytest
from backend.scheduler_config import SCHEDULE


def test_scheduler_total_job_count():
    """Test that scheduler has the expected total number of jobs (25)."""
    assert len(SCHEDULE) == 25, f"Expected 25 jobs, got {len(SCHEDULE)}"


def test_afternoon_loop_jobs_exist():
    """Test that 2:30 PM (14:30) loop jobs exist for all 3 bots."""
    expected_jobs = [
        "bot_loop_1w_1430",
        "bot_loop_2w_1430",
        "bot_loop_4w_1430",
    ]
    job_names = [job["name"] for job in SCHEDULE]
    
    for expected_name in expected_jobs:
        assert expected_name in job_names, f"Missing job: {expected_name}"


def test_market_close_jobs_exist():
    """Test that 4:15 PM (16:15) market close jobs exist for all 3 bots."""
    expected_jobs = [
        "eod_1w_close",
        "eod_2w_close",
        "eod_4w_close",
    ]
    job_names = [job["name"] for job in SCHEDULE]
    
    for expected_name in expected_jobs:
        assert expected_name in job_names, f"Missing job: {expected_name}"


def test_afternoon_loop_jobs_configuration():
    """Test that 2:30 PM loop jobs have correct time and mode."""
    afternoon_loop_jobs = [
        job for job in SCHEDULE 
        if job["name"] in ["bot_loop_1w_1430", "bot_loop_2w_1430", "bot_loop_4w_1430"]
    ]
    
    assert len(afternoon_loop_jobs) == 3, "Expected 3 afternoon loop jobs"
    
    for job in afternoon_loop_jobs:
        assert job["time"] == "14:30", f"Job {job['name']} has wrong time: {job['time']}"
        assert job["args"] == ["--mode", "loop"], f"Job {job['name']} has wrong mode"
        assert "afternoon loop" in job["description"].lower(), (
            f"Job {job['name']} has unexpected description: {job['description']}"
        )


def test_market_close_jobs_configuration():
    """Test that 4:15 PM market close jobs have correct time and mode."""
    market_close_jobs = [
        job for job in SCHEDULE 
        if job["name"] in ["eod_1w_close", "eod_2w_close", "eod_4w_close"]
    ]
    
    assert len(market_close_jobs) == 3, "Expected 3 market close jobs"
    
    for job in market_close_jobs:
        assert job["time"] == "16:15", f"Job {job['name']} has wrong time: {job['time']}"
        assert job["args"] == ["--mode", "full"], f"Job {job['name']} has wrong mode"
        assert "market close" in job["description"].lower(), (
            f"Job {job['name']} has unexpected description: {job['description']}"
        )


def test_swing_bot_job_count():
    """Test that there are exactly 12 swing bot jobs (4 times per day * 3 bots)."""
    swing_bot_jobs = [
        job for job in SCHEDULE
        if any(w in job["name"] for w in ["_1w_", "_2w_", "_4w_", "_1w", "_2w", "_4w"])
        and job["name"].startswith(("eod_", "bot_"))
    ]
    
    # 6:00 AM (3 jobs) + 11:35 AM (3 jobs) + 2:30 PM (3 jobs) + 4:15 PM (3 jobs) = 12 jobs
    assert len(swing_bot_jobs) == 12, f"Expected 12 swing bot jobs, got {len(swing_bot_jobs)}"


def test_all_swing_bot_times():
    """Test that all 4 time slots are covered for each bot."""
    expected_times = {
        "06:00": ["eod_1w_full", "eod_2w_full", "eod_4w_full"],
        "11:35": ["bot_loop_1w_1135", "bot_loop_2w_1135", "bot_loop_4w_1135"],
        "14:30": ["bot_loop_1w_1430", "bot_loop_2w_1430", "bot_loop_4w_1430"],
        "16:15": ["eod_1w_close", "eod_2w_close", "eod_4w_close"],
    }
    
    for time_str, expected_names in expected_times.items():
        jobs_at_time = [job for job in SCHEDULE if job["time"] == time_str]
        job_names = [job["name"] for job in jobs_at_time]
        
        for expected_name in expected_names:
            assert expected_name in job_names, (
                f"Missing job {expected_name} at time {time_str}"
            )
