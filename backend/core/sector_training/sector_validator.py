# backend/core/sector_training/sector_validator.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from backend.core.config import PATHS
from backend.core.data_pipeline import log


def load_return_stats() -> Dict[str, Any]:
    """
    Load global return statistics produced by ai_model training.

    Expected structure:
      {
        "<horizon>": {
            "std": float,
            "valid_global": bool,
            "invalid_reason": str | None,
            ...
        },
        ...
      }

    Safe to call even if stats do not yet exist.
    """
    try:
        p = PATHS.get("RETURN_STATS")

        # Fallback for older layouts
        if not p:
            ml_root = PATHS.get("ML_DATA_ROOT")
            if ml_root:
                p = Path(ml_root) / "metrics" / "return_stats.json"

        if not p:
            return {}

        p = Path(p)
        if not p.exists():
            return {}

        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    except Exception as e:
        log(f"[sector_validator] ⚠️ load_return_stats failed: {e}")
        return {}


def horizon_valid(stats_horizons: Dict[str, Any], horizon: str) -> Tuple[bool, str | None]:
    """
    Validate a horizon for a given stats map.

    Expected shape:
      {
        "<horizon>": {
            "valid_global": bool,
            "invalid_reason": str | None,
            ...
        }
      }

    Backward compatible:
    - Missing stats → allowed
    - Missing valid_global → allowed
    """
    if not isinstance(stats_horizons, dict):
        return False, "no_stats"

    hstats = (
        stats_horizons.get(horizon)
        or stats_horizons.get(horizon.lower())
        or {}
    )

    if not isinstance(hstats, dict):
        return False, "missing_horizon_stats"

    if hstats.get("valid_global") is False:
        return False, str(hstats.get("invalid_reason") or "invalid")

    return True, None
