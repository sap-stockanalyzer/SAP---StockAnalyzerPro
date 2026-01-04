"""backend.services.swing_truth_store — v0.1 (Swing Phase 0)

Truth + safety helpers for *swing* execution.

This mirrors dt_backend's Phase 0 philosophy:
  - Deterministic artifacts
  - Append-only event log
  - Best-effort, never raise in normal use

Artifacts (written under a swing truth dir):
  • swing_state.json       — latest snapshot (risk/bots/regime/etc.)
  • swing_trades.jsonl     — append-only decision + execution + no-trade log
  • swing_metrics.json     — lightweight counters + optional snapshots

Override
--------
Set SWING_TRUTH_DIR to redirect artifacts (useful for replay/backtests):
    SWING_TRUTH_DIR=/abs/path/to/run_dir
Artifacts will be written under: <SWING_TRUTH_DIR>/swing/
"""

from __future__ import annotations

import json
import uuid
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    # Project root config
    from config import PATHS  # type: ignore
except Exception:  # pragma: no cover
    PATHS = {}  # type: ignore

try:
    from backend.core.data_pipeline import log  # type: ignore
except Exception:  # pragma: no cover
    def log(msg: str) -> None:  # type: ignore
        print(msg)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _swing_dir() -> Path:
    override = (os.getenv("SWING_TRUTH_DIR", "") or "").strip()
    if override:
        base = Path(override) / "swing"
        base.mkdir(parents=True, exist_ok=True)
        return base

    # Default: da_brains/swing
    da = PATHS.get("da_brains") if isinstance(PATHS, dict) else None
    base = Path(da) if da else Path("da_brains")
    base = base / "swing"
    base.mkdir(parents=True, exist_ok=True)
    return base


def state_path() -> Path:
    return _swing_dir() / "swing_state.json"


def trades_path() -> Path:
    return _swing_dir() / "swing_trades.jsonl"


def metrics_path() -> Path:
    return _swing_dir() / "swing_metrics.json"


def _read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _atomic_write_json(path: Path, obj: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        log(f"[swing_truth] ⚠️ failed to write {path.name}: {e}")


def read_swing_state() -> Dict[str, Any]:
    s = _read_json(state_path(), {})
    return s if isinstance(s, dict) else {}


def update_swing_state(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge-patch swing_state.json, adding created_at/updated_at."""
    state = read_swing_state()
    if not isinstance(patch, dict):
        patch = {}
    state.update(patch)
    state.setdefault("created_at", _utc_iso())
    state["updated_at"] = _utc_iso()
    _atomic_write_json(state_path(), state)
    return state


def append_swing_event(event: Dict[str, Any]) -> None:
    """Append a single JSONL event line."""
    try:
        if not isinstance(event, dict):
            return
        event.setdefault("ts", _utc_iso())
        p = trades_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"[swing_truth] ⚠️ failed to append swing event: {e}")


def bump_swing_metric(name: str, amount: float = 1.0) -> None:
    """Increment a lightweight counter inside swing_metrics.json."""
    try:
        if not isinstance(name, str) or not name.strip():
            return
        amt = float(amount)
    except Exception:
        return

    m = _read_json(metrics_path(), {})
    if not isinstance(m, dict):
        m = {}
    counters = m.get("counters")
    if not isinstance(counters, dict):
        counters = {}
    try:
        cur = float(counters.get(name) or 0.0)
    except Exception:
        cur = 0.0
    counters[name] = cur + amt
    m["counters"] = counters
    m["ts"] = _utc_iso()
    _atomic_write_json(metrics_path(), m)


# ---------------------------------------------------------------------------
# Optional: tiny "missed opportunities" log
# ---------------------------------------------------------------------------


def missed_path() -> Path:
    return _swing_dir() / "swing_missed.jsonl"


def append_missed_opportunity(event: Dict[str, Any]) -> None:
    """Append a 'missed opportunity candidate' line.

    We don't decide if it *was* missed here; we only record that a trade-like
    setup was rejected (and why). An offline pass can later evaluate outcomes.
    """
    try:
        if not isinstance(event, dict):
            return
        event.setdefault("ts", _utc_iso())
        # Ensure an id exists for later evaluation + attribution joins.
        event.setdefault("event_id", uuid.uuid4().hex)
        p = missed_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"[swing_truth] ⚠️ failed to append missed opportunity: {e}")
