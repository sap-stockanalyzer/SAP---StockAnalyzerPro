# dt_backend/calibration/phit_calibrator_dt.py — Phase 7
"""Calibrated P(hit) for intraday strategy bots.

We intentionally keep this *boringly robust*:

* Calibration artifacts are JSON, generated from Phase-6 replay logs.
* Runtime lookup is fast and safe (never raises).
* Default fallback returns None so the rest of the stack can degrade gracefully.

Artifact format
--------------
Stored at:
  <DT_TRUTH_DIR>/intraday/calibration/phit_calib.json
or (live):
  <DT_PATHS['da_brains']>/intraday/calibration/phit_calib.json

Schema:
  {
    "version": "dt_v1",
    "built_at": "...Z",
    "bins": [0.0, 0.1, 0.2, ... , 1.0],
    "tables": {
       "ORB|TREND_UP": [p0, p1, ...],
       "VWAP_MR|RANGE": [...],
       "DEFAULT": [...]
    }
  }

Interpretation:
  input_conf -> find bin index -> return calibrated hit prob.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dt_backend.core import DT_PATHS


_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "data": None}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _calib_path() -> Path:
    # Replay/backtest can override artifact root.
    override = (os.getenv("DT_TRUTH_DIR", "") or "").strip()
    if override:
        p = Path(override) / "intraday" / "calibration" / "phit_calib.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    da = DT_PATHS.get("da_brains")
    base = da / "intraday" if isinstance(da, Path) else Path("da_brains") / "intraday"
    p = base / "calibration" / "phit_calib.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _load() -> Optional[Dict[str, Any]]:
    path = _calib_path()
    try:
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        if _CACHE.get("path") == str(path) and _CACHE.get("mtime") == mtime and isinstance(_CACHE.get("data"), dict):
            return _CACHE["data"]
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        _CACHE.update({"path": str(path), "mtime": mtime, "data": data})
        return data
    except Exception:
        return None


def _bin_index(conf: float, bins: List[float]) -> int:
    # bins are increasing; return index of rightmost <= conf
    if not bins:
        return 0
    if conf <= bins[0]:
        return 0
    for i in range(len(bins) - 1):
        if bins[i] <= conf < bins[i + 1]:
            return i
    return max(0, len(bins) - 2)


def get_phit(
    *,
    bot: str,
    regime_label: str,
    base_conf: float,
    allow_default: bool = True,
) -> Optional[float]:
    """Return calibrated P(hit) or None if unavailable."""

    data = _load()
    if not data:
        return None

    try:
        bins = data.get("bins")
        if not isinstance(bins, list) or not bins:
            return None
        bins_f = [float(x) for x in bins]

        tables = data.get("tables")
        if not isinstance(tables, dict) or not tables:
            return None

        key = f"{str(bot).upper()}|{str(regime_label).upper()}"
        row = tables.get(key)
        if not isinstance(row, list) and allow_default:
            row = tables.get("DEFAULT")
        if not isinstance(row, list) or not row:
            return None

        conf = max(0.0, min(1.0, _safe_float(base_conf, 0.0)))
        idx = _bin_index(conf, bins_f)
        idx = max(0, min(idx, len(row) - 1))
        phit = max(0.0, min(1.0, _safe_float(row[idx], 0.0)))
        return float(phit)
    except Exception:
        return None


def write_default_stub(version: str = "dt_v1") -> Path:
    """Create a conservative default calibrator if none exists.

    This is mainly for first-run ergonomics. It is intentionally *not* optimistic.
    """
    path = _calib_path()
    if path.exists():
        return path

    bins = [i / 10.0 for i in range(0, 11)]
    # Default: slightly pessimistic mapping: conf -> 0.45..0.60
    default_row = [0.45, 0.47, 0.48, 0.50, 0.51, 0.53, 0.55, 0.56, 0.58, 0.59]
    payload = {
        "version": str(version),
        "built_at": _utc_iso(),
        "bins": bins,
        "tables": {"DEFAULT": default_row},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
