"""
Auto-retrain trigger for ML models.

Monitors metrics and determines when model retraining is needed.
"""

from typing import Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json

try:
    from dt_backend.core.data_pipeline_dt import log  # type: ignore
except Exception:
    def log(msg: str) -> None:
        print(msg, flush=True)


class AutoRetrainTrigger:
    """Determine when ML models need retraining."""
    
    def __init__(self, metrics_dir: Path = Path("ml_data_dt/metrics")):
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.last_retrain_file = self.metrics_dir / "last_retrain.json"
        
        # Thresholds for triggering retrain
        self.min_win_rate = 0.45
        self.min_sharpe = 0.5
        self.max_feature_drift = 0.3
        self.min_days_between_retrain = 7
    
    def check_and_trigger(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if retraining should be triggered.
        
        Returns: (should_retrain: bool, reason: str)
        """
        # Check last retrain time
        last_retrain = self._load_last_retrain()
        if last_retrain:
            days_since = (datetime.now(timezone.utc) - last_retrain).days
            if days_since < self.min_days_between_retrain:
                return False, f"retrained_{days_since}d_ago"
        
        # Check win rate
        win_rate = metrics.get("win_rate", 0.5)
        if win_rate < self.min_win_rate:
            return True, f"win_rate_{win_rate:.2f}<{self.min_win_rate}"
        
        # Check Sharpe ratio
        sharpe = metrics.get("sharpe_ratio", 0.0)
        if sharpe < self.min_sharpe:
            return True, f"sharpe_{sharpe:.2f}<{self.min_sharpe}"
        
        # Check feature drift
        drift = metrics.get("feature_drift", 0.0)
        if drift > self.max_feature_drift:
            return True, f"drift_{drift:.2f}>{self.max_feature_drift}"
        
        return False, "metrics_healthy"
    
    def record_retrain(self) -> None:
        """Record that a retrain was performed."""
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self.last_retrain_file.open('w') as f:
            json.dump(data, f, indent=2)
    
    def _load_last_retrain(self) -> datetime | None:
        """Load timestamp of last retrain."""
        if not self.last_retrain_file.exists():
            return None
        
        try:
            with self.last_retrain_file.open('r') as f:
                data = json.load(f)
                ts_str = data.get("timestamp")
                if ts_str:
                    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception as e:
            log(f"[retrain_trigger] ⚠️ Error loading last retrain: {e}")
        
        return None
