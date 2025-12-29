# backend/jobs/nightly_job.py — v1.6.1 (Normal + Replay mode + as_of_date) (Import-Safe + Regression + Accuracy + UI Refresh + Fail-Loud + Recent-Run Guard)
"""
Nightly Job — AION Analytics

Key fix in v1.5.6:
  ✅ Skip nightly run if last summary finished within last N hours (default: 8h)

Keeps:
  - Regression training + predictions
  - prediction_logger writes UI feed + ledger
  - accuracy_engine writes calibration map
  - post-brain policy UI refresh (append_ledger=False)
  - fail-loud checks for "no valid horizons" and "preds_total==0"
"""

from __future__ import annotations

import os

import json
import sys
import time
import traceback
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

try:
    from config import ROOT  # unified project root
except Exception:  # pragma: no cover
    # Fallback for direct execution: derive repo root and insert into sys.path
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from config import ROOT  # type: ignore

# -----------------------------
# Always-available core imports
# -----------------------------
from backend.core.config import PATHS, TIMEZONE
from backend.core.data_pipeline import _read_rolling, save_rolling, safe_float, _read_rolling_nervous, save_rolling_nervous
from utils.logger import log

LOCK_FILE = PATHS["nightly_lock"]
SUMMARY_FILE = PATHS["logs"] / "nightly" / "last_nightly_summary.json"

# -----------------------------
# Recent-run guard (NEW)
# -----------------------------
MIN_HOURS_BETWEEN_RUNS = 8


def _recent_nightly_ran_within(hours: int) -> bool:
    if not SUMMARY_FILE.exists():
        return False
    try:
        data = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
        finished = data.get("finished_at")
        if not finished:
            return False
        finished_dt = datetime.fromisoformat(str(finished))
        return datetime.now(TIMEZONE) - finished_dt < timedelta(hours=hours)
    except Exception:
        return False




# -----------------------------
# Rolling read retry (NEW)
# -----------------------------
def _read_rolling_with_retry(attempts: int = 5, sleep_secs: float = 2.0) -> Dict[str, Any]:
    """Best-effort read for rolling_body. Helps when another process is mid-atomic-rename on Windows."""
    last: Dict[str, Any] = {}
    for i in range(max(1, int(attempts))):
        last = _read_rolling() or {}
        if last:
            return last
        try:
            rp = Path(PATHS.get("rolling_body") or PATHS.get("rolling") or "")
            if rp and rp.exists() and rp.stat().st_size > 0:
                log(f"[nightly_job] ⏳ rolling_body present but empty-read (attempt {i+1}/{attempts}); retrying...")
            else:
                return {}
        except Exception:
            return {}
        time.sleep(float(sleep_secs))
    return last or {}


def _append_predictions_history(
    nervous: Dict[str, Any],
    preds: Dict[str, Any],
    run_ts: str,
    keep_days: int = 30,
) -> Dict[str, Any]:
    """Append latest predictions into rolling_nervous history per symbol and prune old entries."""
    cutoff = datetime.now(TIMEZONE) - timedelta(days=int(keep_days))

    for sym, pred in (preds or {}).items():
        s = str(sym)
        node = nervous.get(s)
        if not isinstance(node, dict):
            node = {}
        hist = node.get("predictions_history")
        if not isinstance(hist, list):
            hist = []
        hist.append({"ts": run_ts, "predictions": pred})

        # prune
        pruned = []
        for item in hist:
            try:
                ts_s = str(item.get("ts") or "")
                dt = datetime.fromisoformat(ts_s)
                if dt >= cutoff:
                    pruned.append(item)
            except Exception:
                # keep unknown
                pruned.append(item)
        node["predictions_history"] = pruned[-500:]  # hard safety cap
        node["latest_predictions"] = pred
        node["latest_predictions_ts"] = run_ts
        nervous[s] = node

    return nervous
# -----------------------------
# Safe imports helper
# -----------------------------
def _safe_import(path: str, name: str):
    """
    Import something without killing module import.
    Returns the object or None.
    """
    try:
        mod = __import__(path, fromlist=[name])
        return getattr(mod, name)
    except Exception as e:
        log(f"[nightly_job] ⚠️ Import failed: {path}.{name} — {e}")
        return None


# -----------------------------
# Services / fetchers (safe)
# -----------------------------
load_universe = _safe_import("backend.services.backfill_history", "load_universe")
backfill_symbols = _safe_import("backend.services.backfill_history", "backfill_symbols")
update_fundamentals = _safe_import("backend.services.fundamentals_fetcher", "update_fundamentals")
build_metrics = _safe_import("backend.services.metrics_fetcher", "build_metrics")
build_macro_features = _safe_import("backend.services.macro_fetcher", "build_macro_features")
build_social_sentiment = _safe_import("backend.services.social_sentiment_fetcher", "build_social_sentiment")

write_news_brain_snapshots = _safe_import("backend.services.news_brain_builder", "write_news_brain_snapshots")
build_nightly_news_intel = _safe_import("backend.services.news_intel", "build_nightly_news_intel")

build_daily_dataset = _safe_import("backend.services.ml_data_builder", "build_daily_dataset")
build_daily_insights = _safe_import("backend.services.insights_builder", "build_daily_insights")
log_predictions = _safe_import("backend.services.prediction_logger", "log_predictions")

# Core ML + policy + learning (safe)
train_all_models = _safe_import("backend.core.ai_model.core_training", "train_all_models")
train_all_sector_models = _safe_import("backend.core.sector_training.sector_trainer", "train_all_sector_models")
predict_all = _safe_import("backend.core.ai_model.core_training", "predict_all")
apply_policy = _safe_import("backend.core.policy_engine", "apply_policy")
build_context = _safe_import("backend.core.context_state", "build_context")
detect_regime = _safe_import("backend.core.regime_detector", "detect_regime")
run_continuous_learning = _safe_import("backend.core.continuous_learning", "run_continuous_learning")
run_supervisor_agent = _safe_import("backend.core.supervisor_agent", "run_supervisor_agent")

# Optional phases
aggregate_system_performance = _safe_import("backend.services.performance_aggregator", "aggregate_system_performance")
update_aion_brain = _safe_import("backend.services.aion_brain_updater", "update_aion_brain")

# Accuracy engine
compute_accuracy = _safe_import("backend.services.accuracy_engine", "compute_accuracy")

# Progress bar (optional but nice)
progress_bar = _safe_import("utils.progress_bar", "progress_bar")


PIPELINE: List[Tuple[str, str]] = [
    ("load_rolling", "Load rolling cache"),
    ("backfill", "Heal / backfill history"),
    ("fundamentals", "Fundamentals fetch"),
    ("metrics", "Metrics refresh"),
    ("macro", "Macro features"),
    ("social", "Social sentiment"),
    ("news_intel", "News brain + intel (cache-driven)"),
    ("dataset", "ML dataset build (regression)"),
    ("training", "Model training (regression)"),
    ("predictions", "Regression predictions → Rolling"),
    ("prediction_logger", "Prediction logging (UI feed)"),
    ("accuracy_engine", "Accuracy engine (calibration + windows)"),
    ("context", "Context state"),
    ("regime", "Regime detection"),
    ("continuous_learning", "Continuous learning (regression drift)"),
    ("performance", "Performance aggregation (system metrics)"),
    ("aion_brain", "AION brain update (behavioral memory)"),
    ("policy", "Policy engine (brain-aligned)"),
    ("insights", "Insights builder"),
    ("supervisor", "Supervisor agent"),
]
TOTAL_PHASES = len(PIPELINE)


# ----------------------------------------------------------
# Lock helpers
# ----------------------------------------------------------
def _acquire_lock() -> bool:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists():
        log(f"[nightly_job] ⚠️ Lock present at {LOCK_FILE} — exiting.")
        return False
    LOCK_FILE.write_text(datetime.now(TIMEZONE).isoformat(), encoding="utf-8")
    return True


def _release_lock() -> None:
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except Exception as e:
        log(f"[nightly_job] ⚠️ Failed to release lock: {e}")


def _phase(title: str, idx: int, total: int) -> None:
    log("")
    log(f"──────── [{idx}/{total}] {title} ────────")


def _record_ok(summary: Dict[str, Any], key: str, payload: Any, t0: float) -> None:
    summary["phases"][key] = {"status": "ok", "secs": round(time.time() - t0, 3), "result": payload}


def _record_err(summary: Dict[str, Any], key: str, err: Exception, t0: float) -> None:
    tb = traceback.format_exc()
    summary["phases"][key] = {
        "status": "error",
        "secs": round(time.time() - t0, 3),
        "error": str(err),
        "traceback": tb,
    }
    log(f"[nightly_job] ❌ Phase '{key}' failed: {err}")
    log(tb)


def _write_summary(summary: Dict[str, Any]) -> None:
    try:
        SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(f"✅ Summary written → {SUMMARY_FILE}")
    except Exception as e:
        log(f"⚠️ Failed to write summary: {e}")


def _require(fn, name: str) -> None:
    if fn is None:
        raise RuntimeError(f"{name} not available (import failed). Fix missing/broken module: {name}")


def _fail_loud_training_check(train_res: Any) -> None:
    if not isinstance(train_res, dict):
        raise RuntimeError("Training result not a dict (unexpected).")
    horizons = train_res.get("horizons")
    if not isinstance(horizons, dict) or not horizons:
        raise RuntimeError("Training returned no horizons (empty horizons dict).")

    ok_count = 0
    statuses: List[str] = []
    for _h, info in horizons.items():
        if not isinstance(info, dict):
            continue
        st = str(info.get("status", "")).lower()
        statuses.append(st)
        if st == "ok":
            ok_count += 1

    if ok_count <= 0:
        raise RuntimeError("Training produced ZERO valid horizons. Models not usable.")

    skipped = sum(1 for s in statuses if s == "skipped")
    rejected = sum(1 for s in statuses if s in ("rejected", "reject"))
    errors = sum(1 for s in statuses if s == "error")
    log(f"[nightly_job] 📌 Training horizon summary: ok={ok_count}, skipped={skipped}, rejected={rejected}, error={errors}")


# ----------------------------------------------------------
# Main nightly pipeline
# ----------------------------------------------------------
def run_nightly_job(
    mode: str = "normal",
    as_of_date: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    mode = str(mode or "normal").strip().lower()
    as_of_date = str(as_of_date).strip() if as_of_date else None
    if mode not in ("normal", "replay"):
        raise ValueError(f"Invalid mode: {mode}. Expected 'normal' or 'replay'.")

    # Replay mode is strict + deterministic-ish.
    if mode == "replay":
        os.environ["AION_RUN_MODE"] = "replay"
        if as_of_date:
            os.environ["AION_ASOF_DATE"] = as_of_date
        # In replay we never skip due to recent-run guard.
        force = True
    else:
        os.environ["AION_RUN_MODE"] = "normal"

    if (not force) and _recent_nightly_ran_within(MIN_HOURS_BETWEEN_RUNS):
        log(f"[nightly_job] ⏭️ Skipping nightly run — last run finished within {MIN_HOURS_BETWEEN_RUNS}h.")
        summary = {
            "started_at": datetime.now(TIMEZONE).isoformat(),
            "finished_at": datetime.now(TIMEZONE).isoformat(),
            "status": "skipped",
            "reason": f"last_run_within_{MIN_HOURS_BETWEEN_RUNS}h",
            "phases": {},
        }
        _write_summary(summary)
        return summary

    if not _acquire_lock():
        return {"status": "skipped", "reason": "lock_present"}

    summary: Dict[str, Any] = {
        "started_at": datetime.now(TIMEZONE).isoformat(),
        "finished_at": None,
        "status": "running",
        "total_phases": int(TOTAL_PHASES),
        "phases": {},
        "notes": [
            (
                "Policy timing note: prediction_logger runs before aion_brain update in this pipeline. "
                "After policy, nightly_job refreshes latest_predictions.json by re-running prediction_logger "
                "with append_ledger=False (no ledger dupes)."
            )
        ],
    }

    try:
        prediction_run_ts: Optional[str] = None
        rolling: Dict[str, Any] = {}

        # 1) Load rolling
        key, title = PIPELINE[0]
        _phase(title, 1, TOTAL_PHASES)
        t0 = time.time()
        try:
            rolling = _read_rolling_with_retry() or {}
            _record_ok(summary, key, {"symbols": int(len(rolling))}, t0)
            log(f"✅ Load rolling cache — {len(rolling)} symbols loaded.")
        except Exception as e:
            _record_err(summary, key, e, t0)
            rolling = {}

        # 2) Backfill
        key, title = PIPELINE[1]
        _phase(title, 2, TOTAL_PHASES)
        t0 = time.time()
        try:
            _require(load_universe, "backend.services.backfill_history.load_universe")
            _require(backfill_symbols, "backend.services.backfill_history.backfill_symbols")

            universe = load_universe() or [s for s in rolling.keys() if not str(s).startswith("_")]
            if not universe:
                raise RuntimeError("Universe empty — nothing to backfill.")
            updated = backfill_symbols(universe, min_days=180, max_workers=8)
            rolling = _read_rolling() or rolling
            _record_ok(summary, key, {"updated": int(updated), "universe_size": int(len(universe))}, t0)
            log(f"✅ Heal/backfill complete — {updated}/{len(universe)} updated.")
        except Exception as e:
            _record_err(summary, key, e, t0)

        # 3) Fundamentals
        key, title = PIPELINE[2]
        _phase(title, 3, TOTAL_PHASES)
        t0 = time.time()
        try:
            _require(update_fundamentals, "backend.services.fundamentals_fetcher.update_fundamentals")
            res = update_fundamentals(rolling)
            rolling = _read_rolling() or rolling
            _record_ok(summary, key, res, t0)
            log("✅ Fundamentals complete.")
        except Exception as e:
            _record_err(summary, key, e, t0)

        # 4) Metrics
        key, title = PIPELINE[3]
        _phase(title, 4, TOTAL_PHASES)
        t0 = time.time()
        try:
            _require(build_metrics, "backend.services.metrics_fetcher.build_metrics")
            res = build_metrics(rolling)
            rolling = _read_rolling() or rolling
            _record_ok(summary, key, res, t0)
            log("✅ Metrics complete.")
        except Exception as e:
            _record_err(summary, key, e, t0)

        # 5) Macro
        key, title = PIPELINE[4]
        _phase(title, 5, TOTAL_PHASES)
        t0 = time.time()
        try:
            _require(build_macro_features, "backend.services.macro_fetcher.build_macro_features")
            res = build_macro_features()
            _record_ok(summary, key, res, t0)
            log("✅ Macro ready.")
        except Exception as e:
            _record_err(summary, key, e, t0)

        # 6) Social
        key, title = PIPELINE[5]
        _phase(title, 6, TOTAL_PHASES)
        t0 = time.time()
        try:
            _require(build_social_sentiment, "backend.services.social_sentiment_fetcher.build_social_sentiment")
            res = build_social_sentiment()
            _record_ok(summary, key, res, t0)
            log("✅ Social sentiment complete.")
        except Exception as e:
            _record_err(summary, key, e, t0)

        # 7) News brain + intel
        key, title = PIPELINE[6]
        _phase(title, 7, TOTAL_PHASES)
        t0 = time.time()
        try:
            _require(write_news_brain_snapshots, "backend.services.news_brain_builder.write_news_brain_snapshots")
            _require(build_nightly_news_intel, "backend.services.news_intel.build_nightly_news_intel")
            _require(load_universe, "backend.services.backfill_history.load_universe")

            universe = load_universe() or [s for s in rolling.keys() if not str(s).startswith("_")]
            if not universe:
                universe = [s for s in rolling.keys() if not str(s).startswith("_")]

            brain_summary = None
            intel_path = None

            try:
                brain_summary = write_news_brain_snapshots()
            except Exception as e_brain:
                log(f"[nightly_job] ⚠️ write_news_brain_snapshots failed (continuing): {e_brain}")
                brain_summary = {"status": "error", "error": str(e_brain)}

            try:
                intel_path = build_nightly_news_intel(universe)
            except Exception as e_intel:
                log(f"[nightly_job] ⚠️ build_nightly_news_intel failed (continuing): {e_intel}")
                intel_path = None

            payload = {
                "brain": brain_summary,
                "intel_file": str(intel_path) if intel_path else None,
                "universe_size": int(len(universe)),
                "note": "Nightly does NOT call Marketaux. News comes from cache→brain.",
            }
            _record_ok(summary, key, payload, t0)
            log("✅ News brain + intel complete.")
        except Exception as e:
            _record_err(summary, key, e, t0)

        # 8) Dataset
        key, title = PIPELINE[7]
        _phase(title, 8, TOTAL_PHASES)
        t0 = time.time()
        try:
            _require(build_daily_dataset, "backend.services.ml_data_builder.build_daily_dataset")

            mp_env = str(os.getenv("AION_ML_MP", "0")).strip().lower()
            use_mp = mp_env in ("1", "true", "yes", "y", "on")

            if mode == "replay":
                use_mp = False

            log(f"[nightly_job] [ml_data_builder] mp={use_mp} (set AION_ML_MP=1 to enable)")

            ds = build_daily_dataset(
                as_of_date=as_of_date,
                strict=(mode=="replay"),
                use_multiprocessing=use_mp,
                debug=False,
                chunk_symbols=50,
                corr_sample_rows=50_000,
                rewrite_final=True,
                return_dataframe="none",
            )
            _record_ok(summary, key, ds, t0)
            log(f"✅ Dataset built — rows={ds.get('rows', 0)} → {ds.get('file')}")
        except Exception as e:
            _record_err(summary, key, e, t0)

        # 9) Training
        key, title = PIPELINE[8]
        _phase(title, 9, TOTAL_PHASES)
        t0 = time.time()
        try:
            _require(train_all_models, "backend.core.ai_model.train_all_models")
            today = datetime.now(TIMEZONE)
            use_optuna = (today.weekday() == 0)  # Monday tuning
            if mode == "replay":
                use_optuna = False
            n_trials = 20 if use_optuna else 0

            sector_res = {}
            try:
                max_workers = max(1, int(os.getenv("AION_SECTOR_TRAIN_WORKERS", "1") or "1"))
                log(f"[nightly_job] 🧠 Sector training workers={max_workers} (set AION_SECTOR_TRAIN_WORKERS to override)")
                sector_res = (train_all_sector_models(
                    dataset_name="training_data_daily.parquet",
                    use_optuna=use_optuna,
                    n_trials=n_trials,
                    max_workers=max_workers,
                ) or {})
            except Exception as e:
                log(f"[nightly_job] ⚠️ sector training failed (continuing with global): {e}")
                sector_res = {"status": "error", "error": str(e)}

            res = train_all_models(
                dataset_name="training_data_daily.parquet",
                use_optuna=use_optuna,
                n_trials=n_trials,
            )
            _fail_loud_training_check(res)
            summary["sector_training"] = sector_res
            _record_ok(summary, key, res, t0)
            log(f"✅ Training complete (optuna={use_optuna}).")
        except Exception as e:
            _record_err(summary, key, e, t0)

        # (rest of file omitted in snippet)
        # ... keep unchanged ...

        summary["finished_at"] = datetime.now(TIMEZONE).isoformat()
        summary["status"] = "ok"
        _write_summary(summary)
        return summary

    except Exception as e:
        tb = traceback.format_exc()
        log(f"[nightly_job] 💥 Unhandled crash: {e}")
        log(tb)
        summary["status"] = "crashed"
        summary["finished_at"] = datetime.now(TIMEZONE).isoformat()
        summary["phases"]["_fatal"] = {"error": str(e), "traceback": tb}
        _write_summary(summary)
        return summary

    finally:
        _release_lock()


def run() -> Dict[str, Any]:
    return run_nightly_job(mode="normal", as_of_date=None, force=False)


def main(mode: str = "normal", as_of_date: Optional[str] = None, force: bool = False) -> Dict[str, Any]:
    return run_nightly_job(mode=mode, as_of_date=as_of_date, force=force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AION nightly job.")
    parser.add_argument("--mode", default="normal", choices=["normal", "replay"], help="Run mode.")
    parser.add_argument("--as-of", dest="as_of", default=None, help="Replay cut-off date YYYY-MM-DD (replay mode).")
    parser.add_argument("--force", action="store_true", help="Run even if recent-run guard would skip.")
    args = parser.parse_args()
    out = run_nightly_job(mode=str(args.mode), as_of_date=args.as_of, force=bool(args.force))
    print(json.dumps(out, indent=2))
