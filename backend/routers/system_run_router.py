# backend/routers/system_run_router.py
"""
System Run Router — Manual Overrides

POST /api/system/run/{task}

This router triggers heavyweight maintenance tasks in background threads
so the API response returns immediately.

IMPORTANT (multi-worker safe):
- If ENABLE_SCHEDULER=1: execute tasks locally (the scheduler instance)
- If ENABLE_SCHEDULER=0: proxy to the scheduler instance to avoid duplicates

Tasks supported:
  - nightly
  - train
  - insights
  - metrics
  - fundamentals
  - news
  - verify
  - dashboard (stub / not implemented)
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, Callable, Tuple

from fastapi import APIRouter, HTTPException

from backend.core.data_pipeline import _read_rolling, _read_brain, log
from backend.core.config import PATHS

router = APIRouter(prefix="/api/system", tags=["System"])


def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.environ.get(name, "") or "").strip().lower()
    if v == "":
        return default
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _scheduler_enabled() -> bool:
    # Single-owner mode: only the scheduler process should have this ON.
    return _env_bool("ENABLE_SCHEDULER", default=False)


def _scheduler_base_url() -> str:
    """
    Where to proxy heavy tasks when running in API/UI mode.

    Priority:
      1) SCHEDULER_URL
      2) SCHEDULER_HOST + SCHEDULER_PORT
      3) defaults: 127.0.0.1:8001
    """
    url = (os.environ.get("SCHEDULER_URL", "") or "").strip()
    if url:
        return url.rstrip("/")

    host = (os.environ.get("SCHEDULER_HOST", "") or "").strip() or "127.0.0.1"
    port = (os.environ.get("SCHEDULER_PORT", "") or "").strip() or "8001"
    return f"http://{host}:{port}"


def _proxy_to_scheduler(task: str) -> Tuple[bool, Any]:
    """
    POST the same endpoint to the scheduler instance.
    Returns (ok, payload_or_error).
    """
    base = _scheduler_base_url()
    url = f"{base}/api/system/run/{task}"

    req = urllib.request.Request(
        url=url,
        method="POST",
        data=b"",  # empty body
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return True, json.loads(raw) if raw else {"status": "ok"}
            except Exception:
                return True, {"status": "ok", "raw": raw}
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return False, {"error": "scheduler_http_error", "code": e.code, "body": body}
    except Exception as e:
        return False, {"error": "scheduler_unreachable", "detail": str(e), "url": url}


def _run_bg(fn: Callable[[], Any], name: str) -> None:
    def _wrapped():
        try:
            log(f"[system_run] 🚀 START task={name}")
            out = fn()
            log(f"[system_run] ✅ DONE task={name} result={out}")
        except Exception as e:
            log(f"[system_run] ❌ FAIL task={name} err={e}")

    threading.Thread(target=_wrapped, daemon=True).start()


# -------------------------
# Task implementations
# -------------------------

def _task_nightly():
    # Try common entrypoints without being fragile.
    try:
        from backend.jobs.nightly_job import main as nightly_main  # type: ignore
        return nightly_main()
    except Exception:
        from backend.jobs.nightly_job import run as nightly_run  # type: ignore
        return nightly_run()


def _task_train():
    from backend.core.ai_model.core_training import train_all_models
    # Use n_trials=100 for proper Optuna hyperparameter optimization
    # This ensures meaningful tuning (3-5 min per horizon vs 30s with default)
    return train_all_models(n_trials=100)


def _task_insights():
    from backend.services.insights_builder import build_daily_insights
    return build_daily_insights(limit=50)


def _task_metrics():
    from backend.services.metrics_fetcher import build_latest_metrics
    return build_latest_metrics()


def _task_fundamentals():
    from backend.services.fundamentals_fetcher import enrich_fundamentals
    return enrich_fundamentals()


def _task_news():
    # Manual override should be stable, so force MP off.
    from backend.services.news_fetcher import run_news_fetch
    return run_news_fetch(days_back=2, use_multiprocessing=False)


def _task_verify() -> Dict[str, Any]:
    rolling = _read_rolling() or {}
    brain = _read_brain() or {}

    rolling_syms = [k for k in rolling.keys() if not str(k).startswith("_")]
    brain_syms = [k for k in brain.keys() if not str(k).startswith("_")]

    return {
        "status": "ok",
        "paths": {
            "rolling": str(PATHS.get("rolling")),
            "brain": str(PATHS.get("rolling_brain")),
            "backups": str(PATHS.get("rolling_backups")),
        },
        "counts": {
            "rolling_keys": len(rolling),
            "rolling_symbols": len(rolling_syms),
            "brain_keys": len(brain),
            "brain_symbols": len(brain_syms),
        },
        "scheduler_enabled": _scheduler_enabled(),
    }


def _task_dashboard() -> Dict[str, Any]:
    """Compute and cache dashboard metrics."""
    try:
        from backend.services.unified_cache_service import UnifiedCacheService
        
        # Update the unified cache which aggregates all dashboard data
        cache_service = UnifiedCacheService()
        if cache_service is None:
            log("[system_run] ❌ Failed to initialize UnifiedCacheService")
            return {
                "status": "error",
                "error": "Failed to initialize cache service",
            }
        
        result = cache_service.update_all()
        
        if result is None:
            log("[system_run] ❌ Cache update returned None")
            return {
                "status": "error",
                "error": "Cache update failed",
            }
        
        return {
            "status": "ok",
            "timestamp": result.get("timestamp"),
            "sections_updated": list(result.get("data", {}).keys()),
            "errors": result.get("errors", {}),
        }
    except Exception as e:
        log(f"[system_run] ❌ Dashboard task error: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


TASKS: Dict[str, Callable[[], Any]] = {
    "nightly": _task_nightly,
    "train": _task_train,
    "insights": _task_insights,
    "metrics": _task_metrics,
    "fundamentals": _task_fundamentals,
    "news": _task_news,
    "verify": _task_verify,
    "dashboard": _task_dashboard,  # Now implemented
}


# These are the tasks that should NEVER run inside random API workers.
HEAVY_TASKS = {"nightly", "train", "insights", "metrics", "fundamentals", "news"}


@router.post("/run/{task}")
def run_task(task: str):
    task_key = (task or "").strip().lower()

    fn = TASKS.get(task_key)
    if not fn:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Unknown task '{task_key}'",
                "allowed": sorted(TASKS.keys()),
            },
        )

    # ✅ Multi-worker safe behavior:
    # If we're NOT the scheduler owner, proxy heavy jobs to the scheduler instance.
    if (task_key in HEAVY_TASKS) and (not _scheduler_enabled()):
        ok, payload = _proxy_to_scheduler(task_key)
        if not ok:
            raise HTTPException(status_code=503, detail=payload)
        return {
            "status": "proxied",
            "task": task_key,
            "scheduler": _scheduler_base_url(),
            "scheduler_response": payload,
        }

    # Otherwise run locally (scheduler owner, or lightweight verify/dashboard).
    _run_bg(fn, task_key)
    return {"status": "started", "task": task_key, "scheduler_enabled": _scheduler_enabled()}
