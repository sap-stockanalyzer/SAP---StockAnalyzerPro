"""
dt_backend.ml package

High-level intraday ML building blocks:
  • build_intraday_dataset
  • train_intraday_models
  • score_intraday_tickers
  • build_intraday_signals
  • train_incremental_intraday

IMPORTANT:
- Do NOT import submodules at package import-time.
  Running `python -m dt_backend.ml.some_module` imports this package first.
  If we import that same module here, Python will warn that it was already in sys.modules
  before execution (runpy warning).
"""

# ---------------------------------------------------------------------
# Lazy import wrappers (avoid loading modules at package import-time)
# ---------------------------------------------------------------------

def build_intraday_dataset(*args, **kwargs):
    """
    Lazy loader for intraday dataset builder.

    This prevents runpy warnings when executing:
      python -m dt_backend.ml.ml_data_builder_intraday
    """
    from .ml_data_builder_intraday import build_intraday_dataset as _fn
    return _fn(*args, **kwargs)


def train_intraday_models(*args, **kwargs):
    """
    Lazy loader to only import training code when needed.
    Keeps LightGBM and model deps out of import-time.

    Supports either:
      - train_intraday_models() (if you have that wrapper)
      - train_lightgbm_intraday() (your current file’s entrypoint)
    """
    try:
        from .train_lightgbm_intraday import train_intraday_models as _fn  # type: ignore
        return _fn(*args, **kwargs)
    except Exception:
        from .train_lightgbm_intraday import train_lightgbm_intraday as _fn  # type: ignore
        return _fn(*args, **kwargs)


def score_intraday_tickers(*args, **kwargs):
    """
    Lazy loader + compatibility wrapper for intraday scoring.

    Adapts:
        score_intraday_batch(df, models)
    into API used by:
        dt_backend.jobs.daytrading_job
    """
    from .ai_model_intraday import score_intraday_batch, load_intraday_models
    from dt_backend.core.data_pipeline_dt import _read_rolling, ensure_symbol_node

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

        if max_symbols and len(rows) >= max_symbols:
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

    return {"status": "ok", "symbols_scored": updated}


def build_intraday_signals(*args, **kwargs):
    """Lazy loader for intraday signal builder."""
    from .signals_rank_builder import build_intraday_signals as _fn
    return _fn(*args, **kwargs)


def train_incremental_intraday(*args, **kwargs):
    """Lazy loader for online incremental training."""
    from .continuous_learning_intraday import train_incremental_intraday as _fn
    return _fn(*args, **kwargs)


__all__ = [
    "build_intraday_dataset",
    "train_intraday_models",
    "score_intraday_tickers",
    "build_intraday_signals",
    "train_incremental_intraday",
]
