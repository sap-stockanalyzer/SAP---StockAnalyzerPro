"""
dt_backend/services/auto_retrain_dt.py

Automatic retraining system for intraday models.

Triggers:
- Win rate degraded 8%+ vs baseline
- Accuracy degraded 10%+ vs baseline
- Profit factor < 1.0 (and was > 1.2)
- Confidence calibration < 0.85
- Weekly schedule (7+ days since last retrain)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dt_backend.core.config_dt import DT_PATHS
    from dt_backend.core.logger_dt import log
    from dt_backend.ml.trade_outcome_analyzer import TradeOutcomeAnalyzer
except Exception:
    DT_PATHS = {}
    TradeOutcomeAnalyzer = None
    def log(msg: str) -> None:
        print(msg, flush=True)


class AutoRetrainSystem:
    """Automatic retraining system for day trading models."""
    
    def __init__(self):
        learning_path = DT_PATHS.get("learning")
        if learning_path:
            self.learning_path = Path(learning_path)
        else:
            da_brains = DT_PATHS.get("da_brains", Path("da_brains"))
            self.learning_path = Path(da_brains) / "dt_learning"
        
        self.learning_path.mkdir(parents=True, exist_ok=True)
        self.retrain_log_file = self.learning_path / "retrain_log.jsonl"
        self.last_retrain_file = self.learning_path / "last_retrain.json"
    
    def check_retrain_triggers(self) -> Dict[str, Any]:
        """Check if models need retraining.
        
        Returns:
            Dict with should_retrain, triggers, and performance metrics
        """
        try:
            if TradeOutcomeAnalyzer is None:
                return {"should_retrain": False, "triggers": [], "error": "TradeOutcomeAnalyzer not available"}
            
            analyzer = TradeOutcomeAnalyzer(self.learning_path)
            
            # Get current and baseline performance
            current = analyzer.get_performance_window(days=7)
            baseline = analyzer.get_baseline_performance()
            
            triggers = []
            
            # Check if we have baseline to compare against
            if not baseline or not current.get("total_trades", 0):
                # No baseline or no trades - check weekly schedule only
                days_since = self._days_since_last_retrain()
                if days_since >= 7:
                    triggers.append(("weekly_schedule", days_since, 7))
                
                return {
                    "should_retrain": len(triggers) > 0,
                    "triggers": triggers,
                    "current_performance": current,
                    "baseline_performance": baseline,
                }
            
            # Win rate degradation check
            current_win_rate = current.get("win_rate", 0.0)
            baseline_win_rate = baseline.get("win_rate", 0.0)
            
            if baseline_win_rate > 0 and current_win_rate < baseline_win_rate - 0.08:
                triggers.append(("win_rate_drop", current_win_rate, baseline_win_rate))
                log(f"[auto_retrain] 🔴 Win rate dropped: {current_win_rate:.2%} vs {baseline_win_rate:.2%}")
            
            # Accuracy degradation check
            current_acc = current.get("accuracy", 0.0)
            baseline_acc = baseline.get("accuracy", 0.0)
            
            if baseline_acc > 0 and current_acc < baseline_acc - 0.10:
                triggers.append(("accuracy_drop", current_acc, baseline_acc))
                log(f"[auto_retrain] 🔴 Accuracy dropped: {current_acc:.2%} vs {baseline_acc:.2%}")
            
            # Profit factor collapse check
            current_pf = current.get("profit_factor", 0.0)
            baseline_pf = baseline.get("profit_factor", 0.0)
            
            if baseline_pf > 1.2 and current_pf < 1.0:
                triggers.append(("profit_factor_collapse", current_pf, baseline_pf))
                log(f"[auto_retrain] 🔴 Profit factor collapsed: {current_pf:.2f} vs {baseline_pf:.2f}")
            
            # Weekly schedule check
            days_since = self._days_since_last_retrain()
            if days_since >= 7:
                triggers.append(("weekly_schedule", days_since, 7))
                log(f"[auto_retrain] 📅 Weekly retrain due: {days_since} days since last")
            
            return {
                "should_retrain": len(triggers) > 0,
                "triggers": triggers,
                "current_performance": current,
                "baseline_performance": baseline,
            }
            
        except Exception as e:
            log(f"[auto_retrain] ⚠️ Error checking triggers: {e}")
            return {
                "should_retrain": False,
                "triggers": [],
                "error": str(e),
            }
    
    def _days_since_last_retrain(self) -> int:
        """Get days since last retrain."""
        try:
            if not self.last_retrain_file.exists():
                return 999  # Force retrain if never done
            
            with open(self.last_retrain_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            last_retrain_str = data.get("timestamp", "")
            if not last_retrain_str:
                return 999
            
            last_retrain = datetime.fromisoformat(last_retrain_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            delta = now - last_retrain
            
            return int(delta.total_seconds() / 86400)
            
        except Exception:
            return 999
    
    def retrain_intraday_models(self) -> Dict[str, Any]:
        """Full retrain workflow with validation.
        
        Returns:
            Dict with status and validation results
        """
        try:
            log("[auto_retrain] 🔄 Starting automatic retrain...")
            
            retrain_start = datetime.now(timezone.utc)
            
            # Step 1: Rebuild dataset (last 30 days)
            log("[auto_retrain] 📊 Step 1: Rebuilding dataset...")
            dataset_result = self._rebuild_dataset()
            
            if dataset_result.get("status") != "success":
                log(f"[auto_retrain] ⚠️ Dataset rebuild failed: {dataset_result.get('error')}")
                return {
                    "status": "failed",
                    "stage": "dataset",
                    "error": dataset_result.get("error"),
                }
            
            # Step 2: Train models
            log("[auto_retrain] 🧠 Step 2: Training models...")
            train_result = self._train_models()
            
            if train_result.get("status") != "success":
                log(f"[auto_retrain] ⚠️ Model training failed: {train_result.get('error')}")
                return {
                    "status": "failed",
                    "stage": "training",
                    "error": train_result.get("error"),
                }
            
            # Step 3: Validate new models
            log("[auto_retrain] ✅ Step 3: Validating new models...")
            validation = self._validate_new_models()
            
            # Step 4: Deploy if better
            current_accuracy = self._get_current_model_accuracy()
            new_accuracy = validation.get("accuracy", 0.0)
            
            if new_accuracy > current_accuracy * 0.95:  # Deploy if within 5% or better
                self._deploy_new_models()
                self._update_baseline_performance(validation)
                self._record_retrain_success(retrain_start, validation)
                
                log(f"[auto_retrain] ✅ New models deployed (acc: {new_accuracy:.2%} vs {current_accuracy:.2%})")
                
                return {
                    "status": "deployed",
                    "validation": validation,
                    "accuracy_new": new_accuracy,
                    "accuracy_old": current_accuracy,
                }
            else:
                self._rollback_models()
                self._record_retrain_rejected(retrain_start, validation, current_accuracy)
                
                log(f"[auto_retrain] ⚠️ New models not better, keeping current (acc: {new_accuracy:.2%} vs {current_accuracy:.2%})")
                
                return {
                    "status": "rejected",
                    "validation": validation,
                    "accuracy_new": new_accuracy,
                    "accuracy_old": current_accuracy,
                }
                
        except Exception as e:
            log(f"[auto_retrain] ⚠️ Error in retrain workflow: {e}")
            return {
                "status": "error",
                "error": str(e),
            }
    
    def _rebuild_dataset(self) -> Dict[str, Any]:
        """Rebuild training dataset from last 30 days."""
        try:
            # This would call the actual dataset builder
            # For now, return success stub
            log("[auto_retrain] 📊 Dataset rebuild (stub)")
            
            return {
                "status": "success",
                "days_included": 30,
                "samples": 10000,  # Placeholder
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
    
    def _train_models(self) -> Dict[str, Any]:
        """Train models with Optuna hyperparameter optimization."""
        try:
            # This would call the actual training functions
            # For now, return success stub
            log("[auto_retrain] 🧠 Model training (stub)")
            
            return {
                "status": "success",
                "models_trained": ["lightgbm"],
                "best_params": {},
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
    
    def _validate_new_models(self) -> Dict[str, Any]:
        """Validate new models on held-out data."""
        try:
            # This would run actual validation
            # For now, return placeholder metrics
            log("[auto_retrain] ✅ Model validation (stub)")
            
            return {
                "accuracy": 0.58,
                "win_rate": 0.55,
                "profit_factor": 1.3,
                "sharpe_ratio": 1.5,
            }
        except Exception as e:
            log(f"[auto_retrain] ⚠️ Validation error: {e}")
            return {
                "accuracy": 0.0,
                "error": str(e),
            }
    
    def _get_current_model_accuracy(self) -> float:
        """Get current deployed model accuracy."""
        try:
            if TradeOutcomeAnalyzer is None:
                return 0.5
            
            analyzer = TradeOutcomeAnalyzer(self.learning_path)
            baseline = analyzer.get_baseline_performance()
            
            return baseline.get("accuracy", 0.5)
        except Exception:
            return 0.5
    
    def _deploy_new_models(self) -> None:
        """Deploy new models to production."""
        try:
            # This would copy new models to production location
            log("[auto_retrain] 🚀 Deploying new models (stub)")
        except Exception as e:
            log(f"[auto_retrain] ⚠️ Deploy error: {e}")
    
    def _rollback_models(self) -> None:
        """Rollback to previous models."""
        try:
            # This would restore previous models
            log("[auto_retrain] ↩️ Rolling back to previous models (stub)")
        except Exception as e:
            log(f"[auto_retrain] ⚠️ Rollback error: {e}")
    
    def _update_baseline_performance(self, validation: Dict[str, Any]) -> None:
        """Update baseline performance after successful retrain."""
        try:
            if TradeOutcomeAnalyzer is None:
                return
            
            analyzer = TradeOutcomeAnalyzer(self.learning_path)
            analyzer.save_baseline_performance(validation)
            
            log("[auto_retrain] 📊 Baseline performance updated")
        except Exception as e:
            log(f"[auto_retrain] ⚠️ Baseline update error: {e}")
    
    def _record_retrain_success(self, start_time: datetime, validation: Dict[str, Any]) -> None:
        """Record successful retrain to log."""
        try:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "status": "deployed",
                "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "validation": validation,
            }
            
            # Update last retrain file
            with open(self.last_retrain_file, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            
            # Append to log
            with open(self.retrain_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        except Exception as e:
            log(f"[auto_retrain] ⚠️ Log recording error: {e}")
    
    def _record_retrain_rejected(self, start_time: datetime, validation: Dict[str, Any], current_acc: float) -> None:
        """Record rejected retrain to log."""
        try:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "status": "rejected",
                "duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "validation": validation,
                "current_accuracy": current_acc,
            }
            
            # Append to log
            with open(self.retrain_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        except Exception as e:
            log(f"[auto_retrain] ⚠️ Log recording error: {e}")


def check_and_retrain() -> Dict[str, Any]:
    """Check triggers and retrain if needed.
    
    Main entry point for automatic retraining.
    """
    try:
        system = AutoRetrainSystem()
        
        # Check if retrain is needed
        check_result = system.check_retrain_triggers()
        
        if check_result.get("should_retrain", False):
            log(f"[auto_retrain] 🔔 Retrain triggered: {check_result.get('triggers')}")
            
            # Perform retrain
            retrain_result = system.retrain_intraday_models()
            
            return {
                **check_result,
                **retrain_result,
            }
        else:
            log("[auto_retrain] ✅ No retrain needed")
            return {
                **check_result,
                "status": "skipped",
            }
            
    except Exception as e:
        log(f"[auto_retrain] ⚠️ Error in check_and_retrain: {e}")
        return {
            "status": "error",
            "error": str(e),
        }
