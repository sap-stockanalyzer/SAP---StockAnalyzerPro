"""Integration tests for feature importance tracking system."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_ml_data_dir():
    """Create temporary ML data directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_feature_importance_tracker_initialization(temp_ml_data_dir):
    """Test feature importance tracker initializes correctly."""
    from dt_backend.ml.feature_importance_tracker import FeatureImportanceTracker
    
    tracker = FeatureImportanceTracker(ml_data_dir=temp_ml_data_dir)
    
    assert tracker.ml_data_dir == temp_ml_data_dir
    assert tracker.importance_log_path.parent == temp_ml_data_dir
    assert tracker.stats_path.parent == temp_ml_data_dir
    assert tracker.drift_alerts_path.parent == temp_ml_data_dir


def test_feature_importance_logging(temp_ml_data_dir):
    """Test logging predictions with feature importance."""
    from dt_backend.ml.feature_importance_tracker import FeatureImportanceTracker
    
    tracker = FeatureImportanceTracker(ml_data_dir=temp_ml_data_dir)
    
    # Log a prediction
    features = {
        "rsi_14": 65.5,
        "macd": 0.25,
        "volume_ratio": 1.8,
        "atr_14": 2.5,
    }
    
    tracker.log_prediction(
        symbol="AAPL",
        features_dict=features,
        prediction="BUY",
        confidence=0.75
    )
    
    # Check log file was created
    assert tracker.importance_log_path.exists()
    
    # Check we can read it back
    with tracker.importance_log_path.open() as f:
        lines = f.readlines()
        assert len(lines) == 1


def test_feature_importance_top_features(temp_ml_data_dir):
    """Test retrieving top features."""
    from dt_backend.ml.feature_importance_tracker import FeatureImportanceTracker
    
    tracker = FeatureImportanceTracker(ml_data_dir=temp_ml_data_dir)
    
    # Log multiple predictions
    features = {
        "rsi_14": 65.5,
        "macd": 0.25,
        "volume_ratio": 1.8,
    }
    
    for _ in range(5):
        tracker.log_prediction(
            symbol="AAPL",
            features_dict=features,
            prediction="BUY",
            confidence=0.75
        )
    
    # Get top features
    top_features = tracker.get_top_features(symbol="AAPL", top_n=3)
    
    assert len(top_features) > 0
    assert len(top_features) <= 3
    
    # Each entry should be a tuple of (feature_name, importance_score)
    for feature_name, importance in top_features:
        assert isinstance(feature_name, str)
        assert isinstance(importance, float)


def test_feature_importance_drift_detection_no_baseline(temp_ml_data_dir):
    """Test drift detection when no baseline exists."""
    from dt_backend.ml.feature_importance_tracker import FeatureImportanceTracker
    
    tracker = FeatureImportanceTracker(ml_data_dir=temp_ml_data_dir)
    
    # Should not detect drift without baseline
    drift_detected = tracker.detect_drift(threshold=0.15)
    assert drift_detected is False


def test_feature_importance_utils_permutation():
    """Test permutation importance calculation."""
    from dt_backend.core.feature_importance_utils import calculate_permutation_importance
    
    features = {
        "feature1": 10.0,
        "feature2": 5.0,
        "feature3": 2.0,
    }
    
    importance = calculate_permutation_importance(
        features=features,
        baseline_score=0.8
    )
    
    assert isinstance(importance, dict)
    assert len(importance) == 3
    assert all(0 <= v <= 1 for v in importance.values())
    
    # Sum should be approximately 1.0 (normalized)
    total = sum(importance.values())
    assert abs(total - 1.0) < 0.01


def test_feature_importance_utils_top_n():
    """Test getting top N features."""
    from dt_backend.core.feature_importance_utils import get_top_n_features
    
    importance = {
        "feature1": 0.5,
        "feature2": 0.3,
        "feature3": 0.2,
    }
    
    top_2 = get_top_n_features(importance, top_n=2)
    
    assert len(top_2) == 2
    assert top_2[0][0] == "feature1"
    assert top_2[0][1] == 0.5
    assert top_2[1][0] == "feature2"


def test_feature_importance_utils_drift_detection():
    """Test drift detection utility."""
    from dt_backend.core.feature_importance_utils import detect_feature_drift
    
    recent = {
        "feature1": 0.6,
        "feature2": 0.3,
        "feature3": 0.1,
    }
    
    historical = {
        "feature1": 0.5,
        "feature2": 0.3,
        "feature3": 0.2,
    }
    
    # Should not detect drift with low threshold
    drift_detected, drift_score = detect_feature_drift(recent, historical, threshold=0.2)
    assert isinstance(drift_detected, bool)
    assert isinstance(drift_score, float)
    assert drift_score >= 0.0
    
    # Should detect drift with very low threshold
    drift_detected, _ = detect_feature_drift(recent, historical, threshold=0.01)
    assert drift_detected is True


def test_feature_tracker_singleton():
    """Test singleton pattern for feature tracker."""
    from dt_backend.ml.feature_importance_tracker import get_tracker
    
    tracker1 = get_tracker()
    tracker2 = get_tracker()
    
    # Should be the same instance
    assert tracker1 is tracker2


@pytest.mark.integration
def test_feature_importance_integration_with_policy():
    """Integration test: feature importance tracking with policy engine."""
    # This would test the full integration but requires more setup
    # Marking as integration test that can be skipped in unit test runs
    pytest.skip("Requires full policy engine setup")


@pytest.mark.integration  
def test_feature_importance_integration_with_executor():
    """Integration test: feature importance tracking with trade executor."""
    # This would test the full integration but requires more setup
    # Marking as integration test that can be skipped in unit test runs
    pytest.skip("Requires full executor setup")
