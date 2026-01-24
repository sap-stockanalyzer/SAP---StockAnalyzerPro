from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from backend.core.data_pipeline import log

try:
    import lightgbm as lgb  # type: ignore
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def _make_regressor(params: Optional[dict] = None):
    params = params or {}
    return RandomForestRegressor(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", None),
        n_jobs=-1,
        random_state=42,
    )


def _tune_lightgbm_regressor(
    X: np.ndarray,
    y: np.ndarray,
    horizon: str,
    n_trials: int = 100,  # Increased from 20 to 100 for proper hyperparameter tuning
) -> Dict[str, Any]:
    if not (HAS_OPTUNA and HAS_LGBM):
        return {}

    if len(y) < 200:
        log(f"[ai_model] ⚠️ Skipping Optuna for {horizon}: too few samples ({len(y)})")
        return {}

    log(f"[ai_model] 🔍 Optuna regression tuning horizon={horizon}, trials={n_trials}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
        }

        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)

        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=600,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        pred = booster.predict(X_val, num_iteration=booster.best_iteration)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    log(f"[ai_model] 🎯 Best regression params for {horizon}: {best}")
    return best
