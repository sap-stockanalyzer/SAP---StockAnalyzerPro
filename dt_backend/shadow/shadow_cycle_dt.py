"""dt_backend/shadow/shadow_cycle_dt.py — Phase 6.5

Shadow-mode execution for dt_backend.

This runs an alternate "candidate" configuration on a *copy* of the live
rolling cache, writes its own truth artifacts (dt_shadow_store), and produces
a comparison summary against live.

Design goals
------------
* Zero impact on live artifacts.
* Deterministic and audit-friendly.
* Safe by default: shadow never submits broker orders.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dt_backend.core import (
    build_intraday_context,
    classify_intraday_regime,
    apply_intraday_policy,
)
from dt_backend.core.data_pipeline_dt import _read_rolling, save_rolling
from dt_backend.core.logger_dt import log
from dt_backend.core.time_override_dt import utc_iso
from dt_backend.core.meta_controller_dt import ensure_daily_plan
from dt_backend.engines.feature_engineering import build_intraday_features
from dt_backend.ml import score_intraday_tickers, build_intraday_signals
from dt_backend.core.execution_dt import run_execution_intraday

from dt_backend.services.dt_shadow_store import (
    shadow_rolling_path,
    update_shadow_state,
    append_shadow_event,
)


def _read_gz_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_gz_json(path: Path, obj: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(obj, f)
        tmp.replace(path)
    except Exception as e:
        log(f"[dt_shadow] ⚠️ failed writing {path.name}: {e}")


@contextmanager
def _temp_environ(patch: Dict[str, str]):
    old = dict(os.environ)
    try:
        os.environ.update({k: v for k, v in patch.items() if v is not None})
        yield
    finally:
        # Restore exact env (safe + deterministic).
        os.environ.clear()
        os.environ.update(old)


def _candidate_env() -> Dict[str, str]:
    """Translate DT_SHADOW_* env vars into live DT_* knobs.

    Convention:
      - DT_SHADOW_MIN_CONFIDENCE -> DT_MIN_CONFIDENCE
      - DT_SHADOW_ENABLE_<BOT>  -> DT_ENABLE_<BOT>
      - DT_SHADOW_FORCE_ALL_BOTS -> DT_FORCE_ALL_BOTS

    You can extend this mapping as new knobs appear.
    """
    patch: Dict[str, str] = {}

    # Generic 1:1 mapping for a few common knobs.
    mapping = {
        "DT_SHADOW_MIN_CONFIDENCE": "DT_MIN_CONFIDENCE",
        "DT_SHADOW_MAX_POSITIONS": "DT_MAX_POSITIONS",
        "DT_SHADOW_FORCE_ALL_BOTS": "DT_FORCE_ALL_BOTS",
        "DT_SHADOW_ALLOW_MODEL_FALLBACK": "DT_ALLOW_MODEL_FALLBACK",
        "DT_SHADOW_NEWS_STAND_DOWN": "DT_NEWS_STAND_DOWN",
    }
    for src, dst in mapping.items():
        v = (os.getenv(src, "") or "").strip()
        if v != "":
            patch[dst] = v

    # Per-bot enables
    for bot in ["VWAP_MR", "ORB", "TREND_PULLBACK", "SQUEEZE"]:
        v = (os.getenv(f"DT_SHADOW_ENABLE_{bot}", "") or "").strip()
        if v != "":
            patch[f"DT_ENABLE_{bot}"] = v

    # Tag the version string in artifacts
    ver = (os.getenv("DT_SHADOW_VERSION", "") or "").strip()
    if ver:
        patch["DT_STRATEGY_VERSION"] = ver

    return patch


def _extract_trade_candidates(rolling: Dict[str, Any], top_n: int = 25) -> List[Tuple[str, str, float, float]]:
    """(symbol, action, score, confidence) for trade-gated or top scoring."""
    rows: List[Tuple[str, str, float, float]] = []
    for sym, node in rolling.items():
        if not isinstance(sym, str) or sym.startswith("_"):
            continue
        if not isinstance(node, dict):
            continue
        p = node.get("policy_dt")
        if not isinstance(p, dict):
            continue
        act = str(p.get("action") or "").upper()
        sc = float(p.get("score") or 0.0)
        cf = float(p.get("confidence") or 0.0)
        tg = bool(p.get("trade_gate") is True)
        if tg or act in {"BUY", "SELL", "STAND_DOWN"}:
            rows.append((sym.upper(), act, sc, cf))

    rows.sort(key=lambda r: abs(r[2]), reverse=True)
    return rows[: max(1, int(top_n))]


def compare_live_vs_shadow(*, live: Dict[str, Any], shadow: Dict[str, Any], top_n: int = 25) -> Dict[str, Any]:
    live_rows = _extract_trade_candidates(live, top_n=top_n)
    sh_rows = _extract_trade_candidates(shadow, top_n=top_n)

    live_map = {s: (a, sc, cf) for s, a, sc, cf in live_rows}
    sh_map = {s: (a, sc, cf) for s, a, sc, cf in sh_rows}

    syms = sorted(set(live_map.keys()) | set(sh_map.keys()))
    agree = 0
    div: List[Dict[str, Any]] = []
    for s in syms:
        la, lsc, lcf = live_map.get(s, ("NONE", 0.0, 0.0))
        sa, ssc, scf = sh_map.get(s, ("NONE", 0.0, 0.0))
        if la == sa:
            agree += 1
        else:
            div.append({
                "symbol": s,
                "live": {"action": la, "score": lsc, "confidence": lcf},
                "shadow": {"action": sa, "score": ssc, "confidence": scf},
            })

    div.sort(key=lambda d: abs(float(d["live"]["score"]) - float(d["shadow"]["score"])), reverse=True)

    total = len(syms)
    return {
        "ts": utc_iso(),
        "symbols_compared": total,
        "agreement_rate": (agree / total) if total else 1.0,
        "live_candidates": len(live_rows),
        "shadow_candidates": len(sh_rows),
        "divergences": div[:50],
    }


def run_shadow_cycle(
    *,
    cycle_id: str,
    live_rolling_path: Path,
    max_symbols: int | None = None,
    max_positions: int = 50,
) -> Dict[str, Any]:
    """Run candidate policy on a copy of live rolling.

    Returns a compact comparison dict.
    """
    if not live_rolling_path.exists():
        return {"status": "skipped", "reason": "live_rolling_missing"}

    # 1) Copy live rolling -> shadow rolling (hermetic base state)
    sh_path = shadow_rolling_path()
    try:
        sh_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(live_rolling_path, sh_path)
    except Exception as e:
        return {"status": "error", "reason": f"copy_failed:{e}"}

    # 2) Run the pipeline against the shadow rolling file.
    patch_env = _candidate_env()
    patch_env["DT_ROLLING_PATH"] = str(sh_path)
    patch_env["DT_TRUTH_DIR"] = (os.getenv("DT_TRUTH_DIR") or "").strip()  # keep replay hermetic
    patch_env["DT_SHADOW_ACTIVE"] = "1"

    with _temp_environ(patch_env):
        update_shadow_state({
            "component": "shadow_cycle",
            "cycle_id": cycle_id,
            "version": os.getenv("DT_STRATEGY_VERSION") or os.getenv("DT_SHADOW_VERSION") or "shadow",
        })
        append_shadow_event({"type": "shadow_cycle_start", "cycle_id": cycle_id})

        # NOTE: we do not fetch data here; we use the copied rolling bars.
        # But we recompute derived layers (context/features/scoring/policy/execution/signals)
        # so the candidate can differ by knobs.
        try:
            build_intraday_context()
        except Exception:
            pass
        try:
            build_intraday_features(max_symbols=max_symbols)
        except Exception:
            pass
        try:
            score_intraday_tickers(max_symbols=max_symbols)
        except Exception:
            pass
        try:
            classify_intraday_regime()
        except Exception:
            pass

        try:
            ensure_daily_plan(force=False)
        except Exception:
            pass

        try:
            apply_intraday_policy(max_positions=max_positions)
        except Exception:
            pass
        try:
            run_execution_intraday()
        except Exception:
            pass
        try:
            build_intraday_signals()
        except Exception:
            pass

        append_shadow_event({"type": "shadow_cycle_end", "cycle_id": cycle_id})

    # 3) Load both rollings and compare.
    live_roll = _read_gz_json(live_rolling_path)
    sh_roll = _read_gz_json(sh_path)
    comp = compare_live_vs_shadow(live=live_roll, shadow=sh_roll, top_n=int(os.getenv("DT_SHADOW_TOP_N", "25") or "25"))
    comp["cycle_id"] = cycle_id
    comp["shadow_version"] = (os.getenv("DT_SHADOW_VERSION") or "shadow").strip() or "shadow"
    return comp
