# backend/routers/diagnostics_router.py

from __future__ import annotations

import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter

from config import ROOT, PATHS, DT_PATHS


router = APIRouter(prefix="/api", tags=["diagnostics"])


def _stat(path: Path) -> Dict[str, Any]:
    try:
        st = path.stat()
        return {
            "exists": True,
            "size_bytes": int(st.st_size),
            "mtime_iso": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        }
    except FileNotFoundError:
        return {"exists": False, "size_bytes": None, "mtime_iso": None}
    except Exception as e:
        return {"exists": False, "size_bytes": None, "mtime_iso": None, "error": str(e)}


def _read_json_ts(path: Path, keys: List[str]) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for k in keys:
        v = data.get(k)
        if isinstance(v, str) and v:
            return v
    return None


@router.get("/diagnostics")
def diagnostics() -> Dict[str, Any]:
    # Canonical file set to check (paths + important artifacts)
    check_items: List[Dict[str, Any]] = []

    def add_item(key: str, p: Path):
        s = _stat(p)
        check_items.append(
            {
                "key": key,
                "path": str(p.resolve()),
                **s,
            }
        )

    # Core paths (absolute)
    add_item("root", ROOT)
    for k in [
        "universe",
        "ml_data",
        "ml_models",
        "ml_predictions",
        "ml_datasets",
        "da_brains",
        "rolling_brain",
        "dt_rolling_brain",
        "logs",
        "nightly_logs",
        "intraday_logs",
        "swing_replay_state",
        "nightly_lock",
        "swing_replay_lock",
    ]:
        if k in PATHS:
            add_item(k, Path(PATHS[k]))

    # Universe files (new split files used across repo)
    for k in ["universe_master_file", "universe_swing_file", "universe_dt_file"]:
        # Some builds may not have explicit keys; compute by convention if missing.
        p = PATHS.get(k)
        if p is None:
            # fallback: data/universe/<name>.json
            if k == "universe_master_file":
                p = ROOT / "data" / "universe" / "master_universe.json"
            elif k == "universe_swing_file":
                p = ROOT / "data" / "universe" / "swing_universe.json"
            else:
                p = ROOT / "data" / "universe" / "dt_universe.json"
        add_item(k, Path(p))

    # DT paths (optional)
    for k in ["ml_data_dt", "intraday_ui_store"]:
        if k in DT_PATHS:
            add_item(f"dt:{k}", Path(DT_PATHS[k]))

    missing = [{"key": x["key"], "path": x["path"]} for x in check_items if not x.get("exists")]

    # Last nightly summary (best-effort)
    nightly_summary_path = Path(PATHS.get("logs", ROOT / "logs")) / "nightly" / "last_nightly_summary.json"
    last_nightly_finished = _read_json_ts(nightly_summary_path, ["finished_at", "completed_at", "timestamp"])

    # Replay state
    replay_state_path = Path(PATHS.get("swing_replay_state", ROOT / "data" / "replay" / "swing" / "replay_state.json"))
    last_replay_updated = _read_json_ts(replay_state_path, ["updated_at", "finished_at", "completed_at", "timestamp"])

    # intraday sim summary (optional)
    dt_sim_summary_path = Path(DT_PATHS.get("sim_summary", ROOT / "ml_data_dt" / "sim_logs" / "sim_summary.json"))
    dt_sim_updated = _read_json_ts(dt_sim_summary_path, ["updated_at", "finished_at", "timestamp"])

    return {
        "resolved_root": str(ROOT.resolve()),
        "cwd": os.getcwd(),
        "pythonpath_head": sys.path[:5],
        "paths": check_items,
        "missing": missing,
        "last_nightly": {
            "summary_path": str(nightly_summary_path.resolve()),
            "finished_at": last_nightly_finished,
        },
        "last_replay": {
            "state_path": str(replay_state_path.resolve()),
            "updated_at": last_replay_updated,
        },
        "dt_last_intraday": {
            "sim_summary": str(dt_sim_summary_path.resolve()),
            "updated_at": dt_sim_updated,
        },
    }
