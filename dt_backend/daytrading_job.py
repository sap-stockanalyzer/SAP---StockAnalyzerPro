# dt_backend/daytrading_job.py — AION Intraday Trader v1.1 (fixed)
# Live refresh → Intraday ML dataset → Train (fast) → Predict → Rank file → Signals → Online learn → Sync

from __future__ import annotations
import os, sys, json, gzip, time, traceback
from datetime import datetime
from pathlib import Path

# Allow dt_backend to import backend utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Reuse backend logger
from backend.data_pipeline import log  # type: ignore

# DT config (all paths isolated under *_dt/)
from dt_backend.config_dt import DT_PATHS

# ------------------------------------------------------------
# Ensure required folders exist (only called from the runner)
# ------------------------------------------------------------
def _ensure_dt_dirs():
    dirs = [
        DT_PATHS.get("data_dt") or DT_PATHS.get("dtdata"),                # ✅ supports either alias
        DT_PATHS.get("dtml_data") or DT_PATHS.get("ml_data_dt"),          # ✅ consistent with config_dt.py
        (DT_PATHS.get("dtml_data") or DT_PATHS.get("ml_data_dt")) / "signals",
        DT_PATHS.get("dtmodels"),
        (DT_PATHS.get("stock_cache") or (DT_PATHS.get("data_dt") / "stock_cache")) / "master" / "backups",
    ]
    for d in dirs:
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Job lock (separate from nightly; lives under data_dt/)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
JOB_LOCK_PATH = BASE_DIR / "data_dt" / "daytrading_job.lock"

def _blocking_acquire_lock(lock_path: Path, poll_secs: float = 0.5) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"pid={os.getpid()} ts={datetime.utcnow().isoformat()}")
            return True
        except FileExistsError:
            time.sleep(poll_secs)
        except Exception as e:
            log(f"⚠️ dt lock create failed: {e}")
            time.sleep(poll_secs)

def _release_lock(lock_path: Path):
    try:
        if lock_path.exists() and lock_path.is_file():
            os.remove(lock_path)
    except Exception as e:
        log(f"⚠️ Could not remove dt job lock: {e}")

def _phase(title: str):
    log(f"——— {title} (DT) ———")

# ------------------------------------------------------------
# Minimal DT rolling I/O (kept separate from nightly rolling)
# ------------------------------------------------------------
_DT_ROLLING = DT_PATHS["dtrolling"]

def _read_dt_rolling() -> dict:
    if not _DT_ROLLING.exists():
        return {}
    try:
        with gzip.open(_DT_ROLLING, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"⚠️ Failed to read DT rolling: {e}")
        return {}

def _save_dt_rolling(obj: dict) -> None:
    try:
        _DT_ROLLING.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(_DT_ROLLING) + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(obj or {}, f, ensure_ascii=False)
        os.replace(tmp, _DT_ROLLING)
    except Exception as e:
        log(f"⚠️ Failed to save DT rolling: {e}")

# ------------------------------------------------------------
# Optional: cloud sync placeholder (wire *_dt folders later)
# ------------------------------------------------------------
def _dt_cloud_sync():
    try:
        from backend.cloud_sync import sync_all  # type: ignore
        sync_all()  # current cloud_sync may not include *_dt paths; that's fine
    except Exception:
        log("☁️ DT cloud sync skipped (not configured for *_dt paths).")

# ------------------------------------------------------------
# Helper: seed a minimal rank file if predictions absent
# ------------------------------------------------------------
def _write_seed_rank_from_rolling(rolling_path: Path) -> str | None:
    try:
        with gzip.open(rolling_path, "rt", encoding="utf-8") as f:
            js = json.load(f)
        symbols = sorted((js.get("bars") or {}).keys())
    except Exception:
        symbols = []
    out_dir = DT_PATHS["ml_data_dt"] / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "prediction_rank_fetch.json.gz"
    payload = {s: {"rank": i + 1} for i, s in enumerate(symbols)}
    try:
        with gzip.open(out_file, "wt", encoding="utf-8") as g:
            json.dump(payload, g)
        return str(out_file)
    except Exception as e:
        log(f"⚠️ Could not write seed rank file: {e}")
        return None

# ------------------------------------------------------------
# Main Orchestration
# ------------------------------------------------------------
def run_daytrading_job(mode: str = "full") -> dict:
    """Main function to run the intraday day-trading job."""
    log(f"[DT] ⚡ Day Trading Job — AION Intraday Trader v1.0 (mode={mode})")
    t0 = time.time()

    summary: dict = {
        "status": "ok",
        "mode": mode,
        "live_refresh": None,
        "dataset_rows": 0,
        "training": None,
        "predictions": 0,
        "signals": None,
        "online_learning": None,
        "trade_dispatch": None,
        "synced": False,
    }

    _ensure_dt_dirs()
    if not _blocking_acquire_lock(JOB_LOCK_PATH):
        return {"status": "skipped_locked", **summary}

    try:
        # 1) Load DT rolling snapshot (independent from nightly)
        try:
            dt_rolling = _read_dt_rolling() or {}
            log(f"📦 DT rolling present — {len(dt_rolling):,} tickers loaded.")
        except Exception as e:
            log(f"❌ Failed to load DT rolling: {e}")
            dt_rolling = {}
            summary["status"] = "degraded"

        # 2) Live refresh (optional fetcher)
        _phase("Live price refresh")
        try:
            try:
                from backend.live_prices_router import fetch_live_prices  # optional
            except Exception:
                fetch_live_prices = None

            if fetch_live_prices is None:
                log("[DT] ⚠️ No live price fetcher available (backend.live_prices_router.fetch_live_prices missing).")
                summary["live_refresh"] = "skipped_no_fetcher"
            else:
                live = fetch_live_prices() or {}
                _merge_live_prices_into_dt_rolling(live, dt_rolling)
                summary["live_refresh"] = {"status": "ok", "symbols": len(live)}
                log(f"💹 Live prices refreshed for {len(live)} symbols (DT rolling updated).")
        except Exception as e:
            log(f"[DT] ⚠️ Live refresh failed: {e}")
            summary["live_refresh"] = f"error: {e}"

        # 3) Build intraday ML dataset → ml_data_dt/training_data_intraday.parquet
        _phase("Build intraday dataset")
        try:
            from dt_backend.ml_data_builder_intraday import build_intraday_dataset
            ddf = build_intraday_dataset(dt_rolling=dt_rolling)
            summary["dataset_rows"] = int(getattr(ddf, "shape", [0])[0]) if ddf is not None else 0
            log(f"📊 Intraday dataset → {summary['dataset_rows']} rows.")
        except Exception as e:
            log(f"[DT] ⚠️ Intraday dataset build failed: {e}")
            summary["dataset_rows"] = 0

        # 4) Train intraday model(s) → ml_data_dt/models/intraday/
        _phase("Train intraday models")
        try:
            from dt_backend.train_lightgbm_intraday import train_intraday_models
            summary["training"] = train_intraday_models()
            log(f"🧠 DT training complete: {summary['training']}")
        except Exception as e:
            log(f"[DT] ⚠️ Intraday training failed: {e}")
            summary["training"] = f"error: {e}"

        # 5) Predict intraday signals + ranked file for scheduler
        _phase("Predict intraday signals")
        try:
            from dt_backend.ai_model_intraday import score_intraday_tickers  # your file's function
            from dt_backend.signals_rank_builder import build_intraday_signals   # writes ranked JSON/GZ

            preds = score_intraday_tickers() or {}
            summary["predictions"] = len(preds)
            log(f"🤖 DT predictions generated for {summary['predictions']} tickers.")

            if preds:
                rank_path = build_intraday_signals(preds)
                log(f"📊 Intraday rank file generated → {rank_path}")
            else:
                # seed a rank file so rank_fetch_scheduler has something to chew on
                seeded = _write_seed_rank_from_rolling(_DT_ROLLING)
                if seeded:
                    log(f"ℹ️ Seed rank file written → {seeded}")
                else:
                    log("⚠️ No predictions to rank — seed rank creation also failed.")

            # Push predictions into broker queue for bots
            try:
                from dt_backend.trade_executor import sync_predictions_to_broker

                summary["trade_dispatch"] = sync_predictions_to_broker(preds)
                log(f"🤝 Trade dispatch summary: {summary['trade_dispatch']}")
            except Exception as broker_err:
                log(f"[DT] ⚠️ Broker dispatch skipped: {broker_err}")
        except Exception as e:
            log(f"[DT] ⚠️ Intraday prediction failed: {e}")
            summary["predictions"] = 0

        # 5.5) Context/regime/policy (optional)
        _phase("Context/regime/policy")
        try:
            import dt_backend.context_state_dt as context_state_dt
            import dt_backend.regime_detector_dt as regime_detector_dt
            try:
                from dt_backend.policy_engine_dt import apply as apply_dt_policy  # optional
            except Exception:
                apply_dt_policy = None

            context_state_dt.update()
            regime_detector_dt.run()
            dt_rolling = _read_dt_rolling() or {}
            if apply_dt_policy:
                try:
                    dt_rolling = apply_dt_policy(dt_rolling)
                except Exception:
                    pass
            _save_dt_rolling(dt_rolling)
            log("[DT] Context/regime/policy applied.")
        except Exception as e:
            log(f"[DT] ⚠️ DT context/policy layer skipped: {e}")

        # 6) Build signal board (Top buys/sells) → ml_data_dt/signals/intraday_predictions.json
        _phase("Build signal board")
        try:
            from dt_backend.signals_builder import write_intraday_signals
            out_path = write_intraday_signals()
            summary["signals"] = str(out_path) if out_path else None
            log(f"🏁 Signals written → {summary['signals']}")
        except Exception as e:
            log(f"[DT] ⚠️ Signals build failed: {e}")
            summary["signals"] = None

        # 7) Online incremental learning (same-day P&L / realized outcomes)
        _phase("Online incremental learning")
        try:
            from dt_backend.continuous_learning_intraday import train_incremental_intraday
            learn_res = train_incremental_intraday()
            summary["online_learning"] = learn_res
            log(f"🔁 Online learning done: {learn_res}")
        except Exception as e:
            log(f"[DT] ⚠️ Online learning failed: {e}")
            summary["online_learning"] = f"error: {e}"

        # 8) Save DT rolling + optional sync
        _phase("Save & sync")
        try:
            _save_dt_rolling(dt_rolling)
            _dt_cloud_sync()
            summary["synced"] = True
        except Exception as e:
            log(f"[DT] ℹ️ DT cloud sync skipped: {e}")
            summary["synced"] = False

        dur = time.time() - t0
        log(f"\n✅ Day trading job complete in {dur:.1f}s.")
        return summary

    except Exception as e:
        log(f"❌ DT job fatal error: {e}")
        traceback.print_exc()
        summary["status"] = "error"
        summary["error"] = str(e)
        return summary

    finally:
        _release_lock(JOB_LOCK_PATH)

# ------------------------------------------------------------
# Merge helper used by live price refresh step
# ------------------------------------------------------------
def _merge_live_prices_into_dt_rolling(live: dict, dt_rolling: dict):
    if not isinstance(live, dict):
        return
    node = dt_rolling if isinstance(dt_rolling, dict) else {}
    price_map = {s: (d.get("price") if isinstance(d, dict) else None) for s, d in live.items()}
    # Expect dt_rolling like {"bars": {...}, "prices": {...}}; keep it simple:
    prices = node.get("prices") or {}
    for s, px in price_map.items():
        if isinstance(px, (int, float)):
            prices[s] = float(px)
    node["prices"] = prices
    # Save immediately to avoid losing updates on crash
    _save_dt_rolling(node)

# ------------------------------------------------------------
# Alias for backend_service auto-start compatibility
# ------------------------------------------------------------
def run(mode: str = "full"):
    """Alias for run_daytrading_job(), used by backend_service."""
    return run_daytrading_job(mode)

# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    log("⚡ Starting daytrading_job as standalone module...")
    try:
        res = run_daytrading_job("full")
        log(f"[DT] ✅ Finished with status={res.get('status')}, rows={res.get('dataset_rows')}, preds={res.get('predictions')}")
    except Exception as e:
        log(f"⚠️ daytrading_job failed: {e}")
