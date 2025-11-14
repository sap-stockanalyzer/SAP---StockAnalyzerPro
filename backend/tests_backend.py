"""
tests_backend.py
Quick direct-call smoke tests for backend modules.
Run AFTER one full nightly_job.py execution for best results.
"""

from __future__ import annotations
import os, glob, json
from typing import List

from . import ml_data_builder as mldb
from . import prediction_logger as predlog
from . import outcomes_harvester as harv
from . import online_trainer as online
from . import drift_monitor as drift

def assert_file(path: str):
    if not os.path.exists(path):
        raise AssertionError(f"Expected file not found: {path}")

def test_build_dataset():
    print("→ Building weekly dataset...")
    mldb.build_ml_dataset("weekly")
    # Expect a weekly parquet
    cands = sorted(glob.glob("ml_data/training_data_weekly.parquet"))
    assert cands, "No weekly parquet produced"
    print("  OK")

def test_prediction_logging():
    print("→ Logging dummy prediction...")
    recs = [{"symbol": "AAPL", "horizon": "1w", "y_pred": 0.01, "proba": 0.6}]
    path = predlog.log_predictions(recs, "test", feature_names=[])
    assert os.path.exists(path), "Prediction log not created"
    print("  OK")

def test_harvest_and_online():
    print("→ Harvesting outcomes + online trainer")
    files = sorted(glob.glob("ml_data/prediction_logs/*"))
    if not files:
        raise AssertionError("No prediction logs were found for harvesting")
    out = harv.harvest_latest(files[-1])
    assert_file(out)
    res = online.train_incremental(feature_cols=[])
    assert isinstance(res, dict), "Online trainer did not return a dict"
    print("  OK")

def test_drift_report():
    print("→ Drift report")
    cands = sorted(glob.glob("ml_data/training_data_*.parquet"))
    if len(cands) < 2:
        print("  (skipping — need at least 2 parquet files)")
        return
    out = drift.run_drift_report(cands[-2], cands[-1], "test_drift")
    assert_file(out)
    print("  OK")

def run_all():
    test_build_dataset()
    test_prediction_logging()
    test_harvest_and_online()
    test_drift_report()
    print("✅ All smoke tests passed (or safely skipped).")

if __name__ == "__main__":
    run_all()
