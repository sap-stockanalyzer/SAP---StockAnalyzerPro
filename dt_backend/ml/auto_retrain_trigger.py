# dt_backend/ml/auto_retrain_trigger.py
"""
Auto-retraining trigger based on performance metrics.

This module automatically triggers model retraining when performance degrades below
acceptable thresholds or when sufficient time has passed since the last retraining.

Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List

from dt_backend.core.constants_dt import (
    WIN_RATE_RETRAINING_THRESHOLD,
    SHARPE_RETRAINING_THRESHOLD,
    FEATURE_DRIFT_RETRAINING_THRESHOLD,
    MAX_DAYS_WITHOUT_RETRAIN,
)


class AutoRetrainTrigger:
    """Automatically trigger model retraining when performance degrades.
    
    This class monitors trading metrics and triggers retraining when:
    - Win rate falls below threshold
    - Sharpe ratio falls below threshold
    - Feature drift exceeds threshold
    - Maximum days since last retrain exceeded
    
    Attributes:
        metrics_history: List of historical metrics for tracking
        last_retrain_date: Timestamp of last retraining event
        
    Examples:
        >>> trigger = AutoRetrainTrigger()
        >>> metrics = {"win_rate": 0.40, "sharpe_ratio": 0.45, "feature_drift": 0.12}
        >>> should_retrain, reason = trigger.check_and_trigger(metrics)
        >>> if should_retrain:
        ...     print(f"Retraining needed: {reason}")
        ...     # Perform retraining
        ...     trigger.record_retrain()
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize trigger with optional custom config.
        
        Args:
            config_path: Optional path to custom configuration file (reserved for future use)
        """
        self.metrics_history: List[Dict[str, float]] = []
        self.last_retrain_date: Optional[datetime] = None
        self.config_path = config_path
    
    def check_and_trigger(self, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """Check if retraining should be triggered based on current metrics.
        
        Evaluates multiple conditions and returns whether retraining should occur
        along with the reason(s).
        
        Args:
            metrics: Dictionary containing performance metrics with keys:
                - "win_rate": Fraction of winning trades (0.0-1.0)
                - "sharpe_ratio": Risk-adjusted return metric
                - "feature_drift": KL divergence or similar drift metric
                
        Returns:
            Tuple of (should_retrain, reason) where:
                - should_retrain: True if any trigger condition is met
                - reason: Human-readable explanation of trigger(s)
                
        Trigger conditions:
            - win_rate < WIN_RATE_RETRAINING_THRESHOLD (default: 0.45)
            - sharpe_ratio < SHARPE_RETRAINING_THRESHOLD (default: 0.5)
            - feature_drift > FEATURE_DRIFT_RETRAINING_THRESHOLD (default: 0.15)
            - Days since last retrain > MAX_DAYS_WITHOUT_RETRAIN (default: 7)
            
        Examples:
            >>> trigger = AutoRetrainTrigger()
            >>> metrics = {"win_rate": 0.40, "sharpe_ratio": 1.0, "feature_drift": 0.05}
            >>> should_retrain, reason = trigger.check_and_trigger(metrics)
            >>> print(should_retrain)
            True
            >>> print(reason)
            'win_rate=40.0% < 45.0%'
        """
        reasons: List[str] = []
        
        # Check win rate
        win_rate = metrics.get("win_rate", 0.0)
        if win_rate < WIN_RATE_RETRAINING_THRESHOLD:
            reasons.append(
                f"win_rate={win_rate:.1%} < {WIN_RATE_RETRAINING_THRESHOLD:.1%}"
            )
        
        # Check Sharpe ratio
        sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
        if sharpe_ratio < SHARPE_RETRAINING_THRESHOLD:
            reasons.append(
                f"sharpe={sharpe_ratio:.2f} < {SHARPE_RETRAINING_THRESHOLD}"
            )
        
        # Check feature drift
        feature_drift = metrics.get("feature_drift", 0.0)
        if feature_drift > FEATURE_DRIFT_RETRAINING_THRESHOLD:
            reasons.append(
                f"drift={feature_drift:.2f} > {FEATURE_DRIFT_RETRAINING_THRESHOLD}"
            )
        
        # Check time since last retrain
        days_since_retrain = self._days_since_last_retrain()
        if days_since_retrain > MAX_DAYS_WITHOUT_RETRAIN:
            reasons.append(
                f"schedule: {days_since_retrain:.0f}d since last retrain"
            )
        
        # Store metrics in history
        self.metrics_history.append(metrics.copy())
        
        should_retrain = len(reasons) > 0
        reason = " | ".join(reasons) if reasons else "metrics_healthy"
        
        return should_retrain, reason
    
    def record_retrain(self, timestamp: Optional[datetime] = None) -> None:
        """Record that retraining was performed.
        
        Args:
            timestamp: Optional timestamp of retraining event. 
                      If None, uses current UTC time.
                      
        Examples:
            >>> trigger = AutoRetrainTrigger()
            >>> trigger.record_retrain()
            >>> # Or with custom timestamp
            >>> trigger.record_retrain(datetime(2024, 1, 15, tzinfo=timezone.utc))
        """
        self.last_retrain_date = timestamp or datetime.now(timezone.utc)
    
    def _days_since_last_retrain(self) -> float:
        """Calculate days since last retraining.
        
        Returns:
            Number of days since last retrain, or infinity if never retrained.
            
        Examples:
            >>> trigger = AutoRetrainTrigger()
            >>> trigger._days_since_last_retrain()
            inf
            >>> trigger.record_retrain()
            >>> trigger._days_since_last_retrain()  # doctest: +SKIP
            0.0
        """
        if not self.last_retrain_date:
            return float('inf')
        
        delta = datetime.now(timezone.utc) - self.last_retrain_date
        return float(delta.days) + (delta.seconds / 86400.0)
    
    def get_metrics_summary(self) -> Dict[str, any]:
        """Get summary of metrics history and retraining status.
        
        Returns:
            Dictionary with summary statistics including:
                - last_retrain_date: ISO timestamp or None
                - days_since_retrain: Days since last retrain
                - metrics_count: Number of metrics recorded
                
        Examples:
            >>> trigger = AutoRetrainTrigger()
            >>> summary = trigger.get_metrics_summary()
            >>> print(summary['days_since_retrain'])
            inf
        """
        return {
            "last_retrain_date": (
                self.last_retrain_date.isoformat() 
                if self.last_retrain_date else None
            ),
            "days_since_retrain": self._days_since_last_retrain(),
            "metrics_count": len(self.metrics_history),
        }
