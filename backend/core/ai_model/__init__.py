"""backend.core.ai_model

AI/ML model package.

Some parts of the backend (notably sector inference/training) expect a small
set of symbols to exist on this package namespace (e.g., MODEL_ROOT).

We keep the package import-safe by:
  • Re-exporting lightweight path/constants.
  • Providing thin wrappers for heavier helpers (lazy import on call).
"""

from __future__ import annotations

# Public constants expected by other modules
from .constants import FEATURE_LIST_FILE, METRICS_ROOT, MODEL_ROOT

# Primary public API (these were previously used by callers)
from .core_training import predict_all, train_all_models, train_model


def _load_regressors(*args, **kwargs):
    """Lazy wrapper for predictor._load_regressors."""

    from .predictor import _load_regressors as _impl

    return _impl(*args, **kwargs)


def _load_return_stats(*args, **kwargs):
    """Lazy wrapper for target_builder._load_return_stats."""

    from .target_builder import _load_return_stats as _impl

    return _impl(*args, **kwargs)


__all__ = [
    # constants
    "MODEL_ROOT",
    "METRICS_ROOT",
    "FEATURE_LIST_FILE",
    # core public API
    "train_model",
    "train_all_models",
    "predict_all",
    # helpers expected by sector inference
    "_load_regressors",
    "_load_return_stats",
]
