"""
Feature importance tracker for ML pipeline.

Tracks and persists feature importance scores across training runs.
"""

from typing import Dict, List
import json
from pathlib import Path


class FeatureImportanceTracker:
    """Track feature importance scores from ML models."""
    
    def __init__(self):
        self.importance_scores: Dict[str, float] = {}
    
    def update(self, feature_name: str, score: float) -> None:
        """Update importance score for a feature."""
        self.importance_scores[feature_name] = float(score)
    
    def get_top_features(self, n: int = 20) -> List[str]:
        """Get top N most important features."""
        sorted_features = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [feat for feat, _ in sorted_features[:n]]
    
    def save(self, path: Path) -> None:
        """Save importance scores to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as f:
            json.dump(self.importance_scores, f, indent=2)
    
    def load(self, path: Path) -> None:
        """Load importance scores from file."""
        if path.exists():
            with path.open('r') as f:
                self.importance_scores = json.load(f)
