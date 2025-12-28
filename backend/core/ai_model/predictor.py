from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
import numpy as np
from joblib import load

from backend.core.data_pipeline import log

from .constants import HORIZONS, MAX_CONF, MIN_CONF, PRED_DIAG_FILE
from .target_builder import _booster_path, _model_path

try:
    import lightgbm as lgb  # type: ignore
    HAS_LGBM = True
except Exception:
    lgb = None  # type: ignore
    HAS_LGBM = False


# ==========================================================
# LOAD REGRESSION MODELS
# ==========================================================
def _load_regressors(model_root: Path | None = None) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    for horizon in HORIZONS:
        pkl = _model_path(horizon, model_root=model_root)
        txt = _booster_path(horizon, model_root=model_root)

        if pkl.exists():
            try:
                models[horizon] = load(pkl)
                continue
            except Exception as e:
                log(f"[ai_model] ⚠️ Failed to load regressor pkl for {horizon}: {e}")

        if HAS_LGBM and txt.exists():
            try:
                models[horizon] = lgb.Booster(model_file=str(txt))
            except Exception as e:
                log(f"[ai_model] ⚠️ Failed to load booster txt for {horizon}: {e}")

    return models


# ==========================================================
# RATING / LABEL / CONFIDENCE HELPERS
# ==========================================================
def _rating_from_return(pred_ret: float, stats: Dict[str, Any], base_conf: float) -> Tuple[str, int, int]:
    std = float(stats.get("std", 0.05)) or 0.05
    t_hold = 0.25 * std
    t_buy = 0.75 * std
    t_strong = 1.5 * std

    if pred_ret >= t_strong and base_conf >= 0.6:
        return "STRONG_BUY", 2, 1
    if pred_ret >= t_buy:
        return "BUY", 1, 1
    if pred_ret <= -t_strong and base_conf >= 0.6:
        return "STRONG_SELL", -2, -1
    if pred_ret <= -t_buy:
        return "SELL", -1, -1
    if abs(pred_ret) <= t_hold:
        return "HOLD", 0, 0

    if pred_ret > 0:
        return "BUY", 1, 1
    if pred_ret < 0:
        return "SELL", -1, -1
    return "HOLD", 0, 0


def _confidence_from_signal(pred_ret: np.ndarray, stats: Dict[str, Any], sector_momo: np.ndarray | None = None) -> np.ndarray:
    std = float(stats.get("std", 0.05)) or 0.05
    eps = 1e-8

    z = np.abs(pred_ret) / (std + eps)
    base_conf = 0.5 + 0.5 * (1.0 - np.exp(-z))

    if sector_momo is not None:
        sec = np.clip(sector_momo, -0.20, 0.20)
        tilt = 1.0 + 0.5 * sec
        base_conf = base_conf * tilt

    return np.clip(base_conf, MIN_CONF, MAX_CONF)


# ==========================================================
# Prediction diagnostics writer
# ==========================================================
def _hist_counts(values: np.ndarray, bins: np.ndarray) -> List[int]:
    try:
        counts, _ = np.histogram(values.astype(float, copy=False), bins=bins)
        return [int(x) for x in counts]
    except Exception:
        return []


def _write_pred_diagnostics(diag: Dict[str, Any]) -> None:
    try:
        PRED_DIAG_FILE.parent.mkdir(parents=True, exist_ok=True)
        PRED_DIAG_FILE.write_text(json.dumps(diag, indent=2), encoding="utf-8")
        log(f"[ai_model] 📈 Prediction diagnostics written → {PRED_DIAG_FILE}")
    except Exception as e:
        log(f"[ai_model] ⚠️ Failed writing prediction diagnostics: {e}")
