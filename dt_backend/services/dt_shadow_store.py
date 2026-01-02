"""dt_backend/services/dt_shadow_store.py — Phase 6.5

Shadow-mode artifacts.

Shadow runs must be *hermetic*: they run alongside live cycles but cannot
overwrite live truth artifacts.

Directory layout
----------------
If DT_TRUTH_DIR is set (recommended for replay/backtests), we write:

    <DT_TRUTH_DIR>/intraday_shadow/
        dt_state.json
        dt_trades.jsonl
        dt_metrics.json
        rolling_intraday_shadow.json.gz

Otherwise we write to:

    <DA_BRAINS>/intraday_shadow/

This mirrors dt_truth_store.py but with a separate root.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from dt_backend.core import DT_PATHS
from dt_backend.core.logger_dt import log
from dt_backend.core.time_override_dt import utc_iso


def _shadow_root_dir() -> Path:
    override = (os.getenv("DT_TRUTH_DIR", "") or "").strip()
    if override:
        base = Path(override) / "intraday_shadow"
    else:
        da = DT_PATHS.get("da_brains")
        if isinstance(da, Path):
            base = da / "intraday_shadow"
        else:
            base = Path("da_brains") / "intraday_shadow"
    base.mkdir(parents=True, exist_ok=True)
    return base


def state_path() -> Path:
    return _shadow_root_dir() / "dt_state.json"


def trades_path() -> Path:
    return _shadow_root_dir() / "dt_trades.jsonl"


def metrics_path() -> Path:
    return _shadow_root_dir() / "dt_metrics.json"


def shadow_rolling_path() -> Path:
    # Keep it in shadow dir so shadow is completely isolated.
    return _shadow_root_dir() / "rolling_intraday_shadow.json.gz"


def read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def atomic_write_json(path: Path, obj: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        log(f"[dt_shadow] ⚠️ failed to write {path.name}: {e}")


def update_shadow_state(patch: Dict[str, Any]) -> Dict[str, Any]:
    state = read_json(state_path(), {})
    if not isinstance(state, dict):
        state = {}
    if not isinstance(patch, dict):
        patch = {}
    state.update(patch)
    state.setdefault("created_at", utc_iso())
    state["updated_at"] = utc_iso()
    atomic_write_json(state_path(), state)
    return state


def append_shadow_event(event: Dict[str, Any]) -> None:
    try:
        if not isinstance(event, dict):
            return
        event.setdefault("ts", utc_iso())
        p = trades_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"[dt_shadow] ⚠️ failed to append shadow event: {e}")
