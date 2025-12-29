"""dt_backend.ml package

High-level intraday ML building blocks:
  • build_intraday_dataset
  • train_intraday_models
  • score_intraday_tickers
  • build_intraday_signals
  • train_incremental_intraday

Notes
-----
This package intentionally keeps imports *lazy*.

Reason: when you run a submodule with `python -m dt_backend.ml.<submodule>`,
Python imports `dt_backend.ml` (this __init__) first. If we eagerly import
that same submodule here, you'll get the classic runpy warning:

    "found in sys.modules ... prior to execution"

Lazy wrappers avoid that and also keep LightGBM / pandas from loading unless
you actually call the function.
"""

from __future__ import annotations

from typing import Any, Dict


# ---------------------------------------------------------------------
# Lazy wrappers
# ---------------------------------------------------------------------

def build_intraday_dataset(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Build the intraday training dataset parquet from rolling."""
    from .ml_data_builder_intraday import build_intraday_dataset as _fn

    return _fn(*args, **kwargs)


def train_intraday_models(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Train and persist the intraday model(s)."""
    # Canonical trainer currently lives in train_lightgbm_intraday.py
    from .train_lightgbm_intraday import train_lightgbm_intraday as _fn

    return _fn(*args, **kwargs)


def score_intraday_tickers(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Score tickers intraday and write predictions_dt back to rolling."""
    from .ai_model_intraday import score_intraday_batch, load_intraday_models
    from dt_backend.core.data_pipeline_dt import _read_rolling, ensure_symbol_node

    try:
        from dt_backend.core.data_pipeline_dt import save_rolling as _save_rolling  # type: ignore
    except Exception:  # pragma: no cover
        _save_rolling = None

    import pandas as pd

    max_symbols = kwargs.get("max_symbols")

    rolling = _read_rolling() or {}
    rows = []
    index = []

    for sym, node in rolling.items():
        if not isinstance(sym, str) or sym.startswith("_"):
            continue

        feats = node.get("features_dt")
        if not isinstance(feats, dict) or not feats:
            continue

        rows.append(feats)
        index.append(sym)

        if max_symbols and len(rows) >= int(max_symbols):
            break

    if not rows:
        return {"status": "empty", "symbols_scored": 0}

    df = pd.DataFrame(rows, index=index)

    models = load_intraday_models()
    proba_df, labels = score_intraday_batch(df, models=models)

    updated = 0
    for sym in proba_df.index:
        node = ensure_symbol_node(rolling, sym)
        node["predictions_dt"] = {
            "label": str(labels.loc[sym]),
            "proba": proba_df.loc[sym].to_dict(),
        }
        rolling[sym] = node
        updated += 1

    # Persist results (so daytrading_job sees them)
    if _save_rolling is not None:
        try:
            _save_rolling(rolling)
        except Exception:
            pass

    return {
        "status": "ok",
        "symbols_scored": updated,
    }


def build_intraday_signals(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Build ranked intraday signals."""
    from .signals_rank_builder import build_intraday_signals as _fn

    return _fn(*args, **kwargs)


def train_incremental_intraday(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Online / incremental intraday training."""
    from .continuous_learning_intraday import train_incremental_intraday as _fn

    return _fn(*args, **kwargs)


__all__ = [
    "build_intraday_dataset",
    "train_intraday_models",
    "score_intraday_tickers",
    "build_intraday_signals",
    "train_incremental_intraday",
]
