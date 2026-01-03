# dt_backend/ml/__init__.py
"""dt_backend.ml

Lazy-loaded intraday ML utilities.

Public API (import-safe):
- build_intraday_dataset
- train_intraday_models
- score_intraday_tickers
- build_intraday_signals
- train_incremental_intraday

This module intentionally avoids importing heavy submodules at import time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_intraday_dataset(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Build / refresh the intraday ML dataset parquet."""
    from .ml_data_builder_intraday import build_intraday_dataset as _fn

    return _fn(*args, **kwargs)


def train_intraday_models(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Train and persist intraday models (currently LightGBM 3-class)."""
    from .train_lightgbm_intraday import train_lightgbm_intraday as _fn

    return _fn(*args, **kwargs)


def score_intraday_tickers(
    *args: Any,
    symbols: Optional[List[str]] = None,
    max_symbols: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Score tickers and write predictions_dt into rolling.

    This is a thin compatibility wrapper.

    Preferred implementation lives in:
        dt_backend/ml/predict_intraday_to_rolling.py

    Args:
        symbols: optional explicit universe (fast/slow lane).
        max_symbols: optional cap.
        *args/**kwargs: accepted for backward compatibility; ignored if not needed.

    Returns:
        A small status dict.
    """
    # Keep heavy imports inside.
    from .predict_intraday_to_rolling import attach_intraday_predictions

    # Back-compat: older call sites might pass max_symbols in kwargs.
    if max_symbols is None:
        try:
            ms = kwargs.get("max_symbols")
            max_symbols = int(ms) if ms is not None else None
        except Exception:
            max_symbols = None

    try:
        return attach_intraday_predictions(max_symbols=max_symbols, symbols=symbols)
    except TypeError:
        # Very old attach_intraday_predictions signatures might not accept symbols.
        return attach_intraday_predictions(max_symbols=max_symbols)


def build_intraday_signals(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Build ranked intraday signals from rolling."""
    from .signals_rank_builder import build_intraday_signals as _fn

    return _fn(*args, **kwargs)


def train_incremental_intraday(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Online incremental training (if enabled)."""
    from .continuous_learning_intraday import train_incremental_intraday as _fn

    return _fn(*args, **kwargs)


__all__ = [
    "build_intraday_dataset",
    "train_intraday_models",
    "score_intraday_tickers",
    "build_intraday_signals",
    "train_incremental_intraday",
]
