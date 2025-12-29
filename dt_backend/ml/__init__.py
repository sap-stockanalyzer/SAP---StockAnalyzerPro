"""dt_backend.ml package

High-level intraday ML building blocks (lazy-loaded to avoid heavy imports):

  • build_intraday_dataset
  • train_lightgbm_intraday
  • score_intraday_tickers
  • build_intraday_signals
  • train_incremental_intraday

Note:
  - Avoid importing dt_backend.ml submodules at package import time.
    Running modules via `python -m dt_backend.ml.some_module` executes
    the package __init__ first; eager imports can cause runpy warnings
    and double-import side effects.
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# Lazy import wrappers (keep package import light + stable)
# ---------------------------------------------------------------------

def build_intraday_dataset(*args, **kwargs):
    from .ml_data_builder_intraday import build_intraday_dataset as _fn
    return _fn(*args, **kwargs)


def train_lightgbm_intraday(*args, **kwargs):
    """Train & persist the intraday LightGBM model."""
    from .train_lightgbm_intraday import train_lightgbm_intraday as _fn
    return _fn(*args, **kwargs)


def score_intraday_tickers(*args, **kwargs):
    """Score tickers and write predictions back into rolling."""
    from .ai_model_intraday import score_intraday_batch, load_intraday_models
    from dt_backend.core.data_pipeline_dt import _read_rolling, ensure_symbol_node, save_rolling

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

    save_rolling(rolling)

    return {
        "status": "ok",
        "symbols_scored": updated,
    }


def build_intraday_signals(*args, **kwargs):
    from .signals_rank_builder import build_intraday_signals as _fn
    return _fn(*args, **kwargs)


def train_incremental_intraday(*args, **kwargs):
    from .continuous_learning_intraday import train_incremental_intraday as _fn
    return _fn(*args, **kwargs)


__all__ = [
    "build_intraday_dataset",
    "train_lightgbm_intraday",
    "score_intraday_tickers",
    "build_intraday_signals",
    "train_incremental_intraday",
]
