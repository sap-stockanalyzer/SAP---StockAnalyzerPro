
"""Continuous learning / meta-ensemble updater for intraday models.

This is intentionally conservative:
  - If metrics are missing or malformed, it logs and exits.
  - It only nudges ensemble weights based on recent accuracies.
  - Integrates AutoRetrainTrigger for automatic retraining based on performance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timezone, timedelta

import numpy as np

try:
    from dt_backend.core.config_dt import DT_PATHS  # type: ignore
except Exception:
    DT_PATHS: Dict[str, Path] = {
        "dtml_data": Path("ml_data_dt"),
        "dtmodels": Path("dt_backend") / "models",
    }

from dt_backend.models.ensemble.intraday_hybrid_ensemble import EnsembleConfig
from dt_backend.ml.auto_retrain_trigger import AutoRetrainTrigger
from dt_backend.core.constants_dt import (
    WIN_RATE_RETRAINING_THRESHOLD,
    SHARPE_RETRAINING_THRESHOLD,
    FEATURE_DRIFT_RETRAINING_THRESHOLD,
)

try:
    from dt_backend.core.data_pipeline_dt import log  # type: ignore
except Exception:
    def log(msg: str) -> None:
        print(msg, flush=True)


# Global auto-retrain trigger instance
_global_trigger = AutoRetrainTrigger()


def _metrics_path() -> Path:
    root = Path(DT_PATHS.get("dtml_data", Path("ml_data_dt")))
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir / "intraday_model_metrics.json"


def _load_metrics() -> Dict[str, Any] | None:
    path = _metrics_path()
    if not path.exists():
        log(f"[continuous_learning_intraday] ℹ️ Metrics file not found at {path} — skipping.")
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("metrics JSON must be an object")
        return data
    except Exception as e:
        log(f"[continuous_learning_intraday] ⚠️ Failed to read metrics: {e}")
        return None


def _derive_weights(metrics: Dict[str, Any]) -> EnsembleConfig:
    """Derive weights from recent per-model accuracies.

    Expected structure:
    {
      "lightgbm": {"accuracy": 0.60},
      "lstm": {"accuracy": 0.55},
      "transformer": {"accuracy": 0.52}
    }
    """
    def acc_of(key: str) -> float:
        return float(((metrics.get(key) or {}).get("accuracy", 0.0)) or 0.0)

    acc_lgb = acc_of("lightgbm")
    acc_lstm = acc_of("lstm")
    acc_transf = acc_of("transformer")

    arr = np.array([acc_lgb, acc_lstm, acc_transf], dtype=float)
    arr[arr < 0.0] = 0.0
    s = float(arr.sum())
    if s <= 0.0:
        log("[continuous_learning_intraday] ℹ️ No positive accuracies, keeping current weights.")
        return EnsembleConfig.load()

    arr = arr / s
    return EnsembleConfig(w_lgb=float(arr[0]), w_lstm=float(arr[1]), w_transf=float(arr[2]))


def run_continuous_learning_intraday() -> None:
    """Legacy ensemble weight updater - kept for backward compatibility.
    
    Now also checks auto-retrain trigger to determine if retraining is needed.
    """
    metrics = _load_metrics()
    if metrics is None:
        return

    old_cfg = EnsembleConfig.load()
    new_cfg = _derive_weights(metrics)

    # Update weights if they have changed significantly
    if (
        abs(new_cfg.w_lgb - old_cfg.w_lgb) >= 1e-6
        or abs(new_cfg.w_lstm - old_cfg.w_lstm) >= 1e-6
        or abs(new_cfg.w_transf - old_cfg.w_transf) >= 1e-6
    ):
        new_cfg.save()
        log(
            "[continuous_learning_intraday] ✅ Updated ensemble weights → "
            f"LGB={new_cfg.w_lgb:.3f}, LSTM={new_cfg.w_lstm:.3f}, TRANSF={new_cfg.w_transf:.3f}"
        )
    else:
        log("[continuous_learning_intraday] ℹ️ Weights unchanged; nothing to update.")

    new_cfg.save()
    log(
        "[continuous_learning_intraday] ✅ Updated ensemble weights → "
        f"LGB={new_cfg.w_lgb:.3f}, LSTM={new_cfg.w_lstm:.3f}, TRANSF={new_cfg.w_transf:.3f}"
    )
    
    # Check for feature importance drift
    try:
        from dt_backend.ml.feature_importance_tracker import get_tracker
        tracker = get_tracker()
        
        # Update feature importance stats
        stats = tracker.update_stats()
        log(f"[continuous_learning_intraday] 📊 Feature importance stats updated: {stats.get('total_predictions', 0)} predictions")
        
        # Detect drift
        if tracker.detect_drift(threshold=0.15):
            log("[continuous_learning_intraday] ⚠️ Feature importance drift detected!")
            log("[continuous_learning_intraday] 💡 Consider retraining models with updated feature distributions")
    except Exception as e:
        log(f"[continuous_learning_intraday] ℹ️ Feature importance check skipped: {e}")


if __name__ == "__main__":
    run_continuous_learning_intraday()


# ============================================================
#  CONTINUOUS LEARNING DT ORCHESTRATOR (Phase 1-3)
# ============================================================

def run_continuous_learning_dt():
    """Main continuous learning orchestrator for day trading.
    
    Coordinates all learning components:
    - Trade outcome analysis
    - Missed opportunity tracking  
    - Performance monitoring
    - Automatic retraining
    - DT Brain updates
    
    This is the primary entry point called post-market.
    """
    try:
        from dt_backend.jobs.post_market_analysis import run_post_market_analysis
        return run_post_market_analysis()
    except Exception as e:
        log(f"[continuous_learning_dt] ❌ Failed: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================
#  ML FEATURE PIPELINE INTEGRATION (PR #5)
# ============================================================

def continuous_learning_cycle():
    """Run continuous learning cycle after trading day.
    
    Integrates ML feature pipeline with auto-retrain trigger.
    This builds features and determines if model retraining is needed.
    """
    try:
        from dt_backend.ml.feature_pipeline_ml import MLFeaturePipeline
        from dt_backend.ml.auto_retrain_trigger import AutoRetrainTrigger
        from dt_backend.core.data_pipeline_dt import _read_rolling
        
        log("[learning] 🔄 Starting continuous learning cycle...")
        
        # Initialize components
        pipeline = MLFeaturePipeline()
        retrain_trigger = AutoRetrainTrigger()
        
        # 1. Build features
        rolling = _read_rolling()
        if not rolling:
            log("[learning] ⚠️ No rolling state available, skipping")
            return {"status": "skipped", "reason": "no_rolling_state"}
        
        # Get recent trades
        trades = _get_recent_trades(days=1)
        symbols = [s for s in rolling.keys() if not str(s).startswith("_")]
        
        features_df = pipeline.build_features(rolling, trades, symbols)
        pipeline.save_features(features_df)
        
        # 2. Calculate metrics
        metrics = {
            "win_rate": _calculate_win_rate(trades),
            "sharpe_ratio": _calculate_sharpe(trades),
            "feature_drift": _calculate_feature_drift(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        log(f"[learning] 📊 Metrics: win_rate={metrics['win_rate']:.2%}, sharpe={metrics['sharpe_ratio']:.2f}")
        
        # 3. Check retrain trigger
        should_retrain, reason = retrain_trigger.check_and_trigger(metrics)
        
        if should_retrain:
            log(f"[learning] 🔄 Triggering retrain: {reason}")
            schedule_model_retraining()
            retrain_trigger.record_retrain()
            return {
                "status": "retrain_triggered",
                "reason": reason,
                "metrics": metrics,
            }
        else:
            log(f"[learning] ✅ Metrics healthy, no retrain needed")
            return {
                "status": "healthy",
                "metrics": metrics,
            }
    
    except Exception as e:
        log(f"[learning] ❌ Continuous learning cycle failed: {e}")
        return {"status": "error", "message": str(e)}


def _get_recent_trades(days: int = 1) -> List[Dict]:
    """Get recent trades from trade events."""
    try:
        from dt_backend.services.dt_truth_store import get_recent_trades
        return get_recent_trades(days=days)
    except Exception as e:
        log(f"[learning] ⚠️ Error getting recent trades: {e}")
        return []


def _calculate_win_rate(trades: List[Dict]) -> float:
    """Calculate win rate from trades."""
    if not trades:
        return 0.5
    
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return wins / len(trades) if len(trades) > 0 else 0.5


def _calculate_sharpe(trades: List[Dict]) -> float:
    """Calculate Sharpe ratio from trades."""
    if not trades or len(trades) < 2:
        return 0.0
    
    pnls = [t.get("pnl", 0) for t in trades]
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls)
    
    if std_pnl == 0:
        return 0.0
    
    # Annualized Sharpe (assuming ~250 trading days)
    return (mean_pnl / std_pnl) * np.sqrt(250)


def _calculate_feature_drift() -> float:
    """Calculate feature drift metric."""
    # Placeholder - would compare current feature distributions to baseline
    return 0.0


def schedule_model_retraining():
    """Schedule model retraining job."""
    log("[learning] 📅 Scheduling model retraining...")
    # This would trigger the retraining pipeline
    # For now, just log the intent
    pass


def run_incremental_learning_dt():
    """Alias for compatibility with existing code."""
    return run_continuous_learning_dt()
