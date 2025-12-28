from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .constants import MIN_PRED_STD, MAX_CLIP_SAT_FRAC


# Pulled from v1.7.0
def _post_train_sanity(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    horizon: str,
    clip_limit: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": "ok"}

    if X_val is None or y_val is None or len(y_val) < 2000:
        out["status"] = "insufficient_val"
        out["n_val"] = int(len(y_val) if y_val is not None else 0)
        return out

    try:
        pred = np.asarray(model.predict(X_val), dtype=float)
    except Exception as e:
        return {"status": "error", "error": f"predict_failed: {e}"}

    if pred.size == 0:
        return {"status": "error", "error": "empty_predictions"}

    pred_std = float(np.std(pred))
    pred_zero_frac = float(np.mean(np.isclose(pred, 0.0, atol=1e-12)))

    lim = float(max(1e-9, clip_limit))
    pred_clip = np.clip(pred, -lim, lim)
    sat = float(np.mean((pred_clip <= (-lim + 1e-12)) | (pred_clip >= (lim - 1e-12))))

    corr = None
    try:
        if float(np.std(y_val)) > 0 and float(np.std(pred)) > 0:
            corr = float(np.corrcoef(pred, y_val)[0, 1])
    except Exception:
        corr = None

    out.update(
        {
            "n_val": int(len(y_val)),
            "clip_limit": float(lim),
            "pred_std": float(pred_std),
            "pred_zero_frac": float(pred_zero_frac),
            "clip_saturation_frac": float(sat),
            "pred_mean": float(np.mean(pred)),
            "pred_p05": float(np.percentile(pred, 5)),
            "pred_p50": float(np.percentile(pred, 50)),
            "pred_p95": float(np.percentile(pred, 95)),
            "corr_pred_y": corr,
        }
    )

    if pred_std < MIN_PRED_STD:
        out["status"] = "reject"
        out["reject_reason"] = f"degenerate_predictions(pred_std<{MIN_PRED_STD})"
        return out

    if pred_zero_frac >= 0.995:
        out["status"] = "reject"
        out["reject_reason"] = "degenerate_predictions(all_zero)"
        return out

    if sat >= float(MAX_CLIP_SAT_FRAC):
        out["status"] = "reject"
        out["reject_reason"] = f"unstable_predictions(clip_saturation>={MAX_CLIP_SAT_FRAC:.3f})"
        return out

    return out
