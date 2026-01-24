from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.core.data_pipeline import log
from backend.core.ai_model import core_training
from backend.core.ai_model import target_builder
from backend.core.ai_model import feature_pipeline

from backend.core.sector_training.sector_registry import infer_sector_labels_from_features
from backend.core.sector_training.sector_artifacts import sector_paths, env_max_workers, normalize_sector
from backend.core.sector_training.sector_dataset_builder import (
    default_regime_profile_for_horizon,
    apply_regime_filter,
    build_sector_row_mask,
)

# >>> ADDED: sector feature pruning hook
from backend.core.sector_training.sector_feature_policy import prune_features_for_sector
# <<< END ADDITION

# NOTE: We reuse the global dataset parquet and feature list. Sector training scopes by symbol rows (via sector one-hot in features).


def _load_feature_list_paths():
    # Import inside to avoid circulars if configs move
    from backend.core.ai_model.core_training import FEATURE_LIST_FILE, _load_feature_list, _resolve_dataset_path, HORIZONS
    return FEATURE_LIST_FILE, _load_feature_list, _resolve_dataset_path, HORIZONS


def _load_latest_snapshot_any():
    from backend.core.ai_model.core_training import _read_latest_snapshot_any
    return _read_latest_snapshot_any


def _load_latest_features_df(feature_cols: List[str]) -> pd.DataFrame:
    from backend.core.ai_model.core_training import _load_latest_features_df
    return _load_latest_features_df(feature_cols)


def _stream_sector_validation_sample(
    parquet_path: Path,
    feature_cols: List[str],
    target_col: str,
    sector_mask_fn,
    *,
    max_rows: int,
    batch_size: int,
    y_clip_low: float,
    y_clip_high: float,
) -> Tuple[np.ndarray, np.ndarray]:
    Xv, yv = target_builder._stream_validation_sample(
        parquet_path,
        feature_cols=feature_cols,
        target_col=target_col,
        batch_size=batch_size,
        max_rows=max_rows,
        seed=42,
        y_clip_low=y_clip_low,
        y_clip_high=y_clip_high,
    )
    if Xv.size == 0 or yv.size == 0:
        return Xv, yv
    return Xv, yv


def _resolve_workers(max_workers: Optional[int]) -> int:
    """
    Guardrail: sector training is expensive and can OOM if parallelized too hard.
    - If nightly passes max_workers, we respect it.
    - Default is 1 (safe, stable).
    - env_max_workers() still applies any global env caps, but we never exceed the requested value.
    """
    if max_workers is not None:
        try:
            w = int(max_workers)
        except Exception:
            w = 1
        return max(1, w)

    try:
        # env_max_workers reads env caps; default=1 is safe
        w = int(env_max_workers(1))
    except Exception:
        w = 1
    return max(1, w)


def train_sector_models(
    dataset_name: str = "training_data_daily.parquet",
    use_optuna: bool = True,
    n_trials: int = 100,  # Increased from 10 to 100 for proper hyperparameter tuning
    batch_size: int = 150_000,
    max_workers: Optional[int] = None,   # ✅ accept without crashing
    **kwargs: Any,                       # ✅ swallow any future args safely
) -> Dict[str, Any]:
    """
    Sector training entrypoint.

    Key guardrails:
      - Respects caller-provided max_workers (nightly can force 1).
      - Defaults to 1 worker to avoid RAM blowups on large universes.
      - Logs worker/sectors for visibility.

    NOTE: This does NOT compromise model logic; it only controls concurrency.
    """
    FEATURE_LIST_FILE, _load_feature_list, _resolve_dataset_path, HORIZONS = _load_feature_list_paths()

    feat_info = _load_feature_list()
    feature_cols: List[str] = feat_info.get("feature_columns", [])
    target_cols: List[str] = feat_info.get("target_columns", [])

    if not feature_cols:
        return {"status": "error", "error": "no_features"}

    _ = kwargs

    df_path = _resolve_dataset_path(dataset_name)

    # Infer sector labels from latest feature snapshot
    sector_labels = None
    try:
        X_latest = _load_latest_features_df(feature_cols)
        if "symbol" not in X_latest.columns:
            sector_labels = infer_sector_labels_from_features(
                X_latest.reset_index().rename(columns={"index": "symbol"})
            )
        else:
            sector_labels = infer_sector_labels_from_features(X_latest.reset_index())
    except Exception as e:
        log(f"[sector_training] ⚠️ Could not infer sector labels from latest features: {e}")
        sector_labels = None

    # Build sector list
    sectors: List[str] = []
    if sector_labels is not None:
        try:
            sectors = sorted({str(x).upper().strip() for x in sector_labels.values if x})
        except Exception:
            sectors = []
    if not sectors:
        sectors = ["UNKNOWN"]

    # Prebuild per-sector symbol sets once (saves repeated dataframe building)
    sector_symbol_map: Dict[str, set[str]] = {}
    if sector_labels is not None:
        try:
            df_sector = pd.DataFrame(
                {"symbol": sector_labels.index.astype(str), "sector": sector_labels.values}
            )
            df_sector["symbol"] = df_sector["symbol"].astype(str).str.upper()
            df_sector["sector"] = df_sector["sector"].astype(str).str.upper().str.strip()
            for sec in sectors:
                sec_norm = normalize_sector(sec)
                syms = set(df_sector.loc[df_sector["sector"] == sec_norm, "symbol"].unique())
                sector_symbol_map[sec_norm] = syms
        except Exception:
            sector_symbol_map = {}

    workers = _resolve_workers(max_workers)
    log(f"[sector_training] 🧭 Sector training: sectors={len(sectors)} max_workers={workers}")

    results: Dict[str, Any] = {"status": "ok", "sectors": {}}

    def train_one_sector(sec: str) -> Tuple[str, Dict[str, Any]]:
        sec_norm = normalize_sector(sec)
        sp = sector_paths(sec_norm)

        # Non-destructive pruning hint (kept for future per-sector feature maps)
        pr = prune_features_for_sector(sec_norm, feature_cols)
        _feature_cols_for_sector = pr.kept  # intentionally unused (kept for future wiring)

        symbols = sector_symbol_map.get(sec_norm) or set()

        try:
            res = core_training.train_all_models(
                dataset_name=dataset_name,
                use_optuna=use_optuna,
                n_trials=n_trials,
                batch_size=batch_size,
                symbol_whitelist=symbols if symbols else None,
                model_root=sp.model_dir,
            )
            sector_summary = res if isinstance(res, dict) else {"status": "error", "error": "unexpected_train_result"}
        except Exception as e:
            sector_summary = {"status": "error", "error": str(e)}

        try:
            (sp.metrics_dir / "sector_training_summary.json").write_text(
                json.dumps(sector_summary, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

        return sec_norm, sector_summary

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(train_one_sector, s): s for s in sectors}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                sec, summ = fut.result()
            except Exception as e:
                sec = normalize_sector(s)
                summ = {"status": "error", "error": str(e)}
            results["sectors"][sec] = summ

    return results


def train_all_sector_models(*args, **kwargs) -> Dict[str, Any]:
    """Compatibility alias for older imports.

    Some code paths (e.g., nightly_job) expect
    `backend.core.sector_training.sector_trainer.train_all_sector_models`.
    The canonical entrypoint in this module is `train_sector_models`.
    """
    return train_sector_models(*args, **kwargs)
