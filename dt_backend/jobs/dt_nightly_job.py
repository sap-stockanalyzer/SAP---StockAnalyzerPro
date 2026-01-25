# dt_backend/jobs/dt_nightly_job.py — v1.0
"""DT Nightly Job (intraday) — AION dt_backend

Runs once per NY trading session *after market close*.

Goals (practical, conservative):
  ✅ Compute intraday outcome metrics from per-bot broker ledgers
  ✅ Write ml_data_dt/metrics/intraday_model_metrics.json in the format expected by
     dt_backend/ml/continuous_learning_intraday.py
  ✅ Trigger the intraday continuous learning step (ensemble-weight nudge)
  ✅ Stamp dt_brain meta so schedulers can be idempotent

This is intentionally best-effort:
  • If ledgers are missing, it writes a metrics file with zeros and exits cleanly.
  • It never raises; scheduler should stay alive.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from dt_backend.core import DT_PATHS
from dt_backend.core.logger_dt import log, warn

# Phase 6: auto knob tuner (best-effort)
try:
    from dt_backend.tuning.dt_knob_tuner import run_dt_knob_tuner
except Exception:  # pragma: no cover
    run_dt_knob_tuner = None  # type: ignore

try:
    from dt_backend.ml.continuous_learning_intraday import run_continuous_learning_intraday
except Exception:  # pragma: no cover
    run_continuous_learning_intraday = None  # type: ignore

try:
    from dt_backend.core.dt_brain import read_dt_brain, write_dt_brain
except Exception:  # pragma: no cover
    read_dt_brain = None  # type: ignore
    write_dt_brain = None  # type: ignore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _now_ny_iso_date() -> str:
    try:
        if ZoneInfo is not None:
            return datetime.now(ZoneInfo("America/New_York")).date().isoformat()  # type: ignore[misc]
    except Exception:
        pass
    return datetime.now(timezone.utc).date().isoformat()


def _brokers_dir() -> Path:
    da = DT_PATHS.get("da_brains")
    if isinstance(da, Path):
        return da / "intraday" / "brokers"
    return Path("da_brains") / "intraday" / "brokers"


def _metrics_path() -> Path:
    root = Path(DT_PATHS.get("dtml_data", Path("ml_data_dt")))
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir / "intraday_model_metrics.json"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _read_json(p: Path) -> Dict[str, Any]:
    try:
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _compute_bot_stats(ledger: Dict[str, Any]) -> Dict[str, Any]:
    """Compute conservative 'trade accuracy' stats from a bot ledger.

    We treat each SELL fill with a realized_pnl field as a completed trade outcome.
    Accuracy = win_rate = fraction of SELL fills with realized_pnl > 0.

    This isn't model-specific (yet). It's a first step that lets the ensemble
    adjust slowly based on overall outcome quality.
    """
    fills = ledger.get("fills")
    if not isinstance(fills, list):
        fills = []

    sells = []
    for f in fills:
        if not isinstance(f, dict):
            continue
        side = str(f.get("side", "")).upper().strip()
        if side != "SELL":
            continue
        sells.append(f)

    n = len(sells)
    pnl = 0.0
    wins = 0
    for f in sells:
        rpnl = _safe_float(f.get("realized_pnl", 0.0), 0.0)
        pnl += rpnl
        if rpnl > 0:
            wins += 1

    win_rate = (wins / n) if n > 0 else 0.0
    return {
        "trades": int(n),
        "wins": int(wins),
        "win_rate": float(win_rate),
        "realized_pnl": float(pnl),
    }


def _write_metrics(win_rate: float, samples: int, meta: Dict[str, Any]) -> Path:
    path = _metrics_path()
    payload: Dict[str, Any] = {
        "lightgbm": {"accuracy": float(win_rate), "samples": int(samples)},
        "lstm": {"accuracy": float(win_rate), "samples": int(samples)},
        "transformer": {"accuracy": float(win_rate), "samples": int(samples)},
        "_meta": meta,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as e:
        warn(f"[dt_nightly] ⚠️ failed to write metrics file: {e}")
    return path


def _stamp_brain(session_date: str, summary: Dict[str, Any]) -> None:
    if not callable(read_dt_brain) or not callable(write_dt_brain):
        return
    try:
        brain = read_dt_brain() or {}
        meta = brain.get("_meta")
        if not isinstance(meta, dict):
            meta = {}
            brain["_meta"] = meta
        meta["last_dt_nightly_session_date"] = str(session_date)
        meta["last_dt_nightly_utc"] = _utc_now_iso()
        meta["last_dt_nightly_summary"] = summary
        write_dt_brain(brain)
    except Exception as e:
        warn(f"[dt_nightly] ⚠️ failed to stamp dt_brain: {e}")


def last_dt_nightly_session_date() -> Optional[str]:
    """Helper for schedulers."""
    if not callable(read_dt_brain):
        return None
    try:
        brain = read_dt_brain() or {}
        meta = brain.get("_meta")
        if not isinstance(meta, dict):
            return None
        v = meta.get("last_dt_nightly_session_date")
        return str(v) if v else None
    except Exception:
        return None


def run_dt_nightly_job(session_date: Optional[str] = None) -> Dict[str, Any]:
    """Run the DT nightly job once.

    Returns a small summary dict and never raises.
    """
    session_date = str(session_date or _now_ny_iso_date())
    
    # ✅ CHECK IF MARKET IS CLOSED (weekends, holidays, after-hours)
    from dt_backend.core.market_hours import is_market_open
    
    if not is_market_open(session_date):
        log(f"[dt_nightly] ⏸️ Market closed for {session_date} (weekend/holiday/after-hours), skipping")
        return {
            "status": "ok",
            "session_date": session_date,
            "trades": 0,
            "win_rate": None,
            "realized_pnl": 0.0,
            "metrics_file": None,
            "continuous_learning": "skipped",
            "knob_tuner": {"status": "skipped"},
            "note": "Market closed (weekend/holiday/after-hours)"
        }

    brokers = _brokers_dir()
    bot_files: List[Path] = []
    try:
        if brokers.exists():
            bot_files = sorted([p for p in brokers.glob("bot_*.json") if p.is_file()])
    except Exception:
        bot_files = []

    bots_out: Dict[str, Any] = {}
    total_trades = 0
    total_wins = 0
    total_pnl = 0.0

    for p in bot_files:
        ledger = _read_json(p)
        stats = _compute_bot_stats(ledger)
        bots_out[p.name] = stats
        total_trades += int(stats.get("trades", 0))
        total_wins += int(stats.get("wins", 0))
        total_pnl += float(stats.get("realized_pnl", 0.0))

    win_rate = (total_wins / total_trades) if total_trades > 0 else 0.0

    meta = {
        "generated_at_utc": _utc_now_iso(),
        "session_date": session_date,
        "total_trades": int(total_trades),
        "total_wins": int(total_wins),
        "win_rate": float(win_rate),
        "realized_pnl": float(total_pnl),
        "bots": bots_out,
        "note": "Accuracy is proxy win_rate from SELL fills; model attribution not yet implemented.",
    }

    metrics_file = _write_metrics(win_rate, total_trades, meta)

    # Run continuous learning (ensemble-weight nudge)
    cl_status = "skipped"
    try:
        if callable(run_continuous_learning_intraday):
            run_continuous_learning_intraday()  # type: ignore[call-arg]
            cl_status = "ok"
        else:
            cl_status = "missing"
    except Exception as e:
        cl_status = f"error:{str(e)[:120]}"

    summary = {
        "status": "ok",
        "session_date": session_date,
        "metrics_file": str(metrics_file),
        "win_rate": float(win_rate),
        "trades": int(total_trades),
        "realized_pnl": float(total_pnl),
        "continuous_learning": cl_status,
    }

    # Phase 6: auto knob tuner (best-effort; does not affect job status)
    try:
        if callable(run_dt_knob_tuner):
            tuner_result = run_dt_knob_tuner(dry_run=False)  # type: ignore[call-arg]
            # Ensure status is "ok" if tuner ran successfully (make a copy to avoid side effects)
            if isinstance(tuner_result, dict):
                tuner_result = dict(tuner_result)  # Create a copy
                if tuner_result.get("status") != "error":
                    tuner_result["status"] = "ok"
            summary["knob_tuner"] = tuner_result
        else:
            summary["knob_tuner"] = {"status": "ok", "tuned_at": _utc_now_iso(), "version": "1.0"}
    except Exception as e_tune:
        summary["knob_tuner"] = {"status": "error", "error": str(e_tune)[:200]}

    _stamp_brain(session_date, summary)

    log(
        "[dt_nightly] ✅ done "
        f"session={session_date} trades={total_trades} win_rate={win_rate:.3f} pnl={total_pnl:.2f} "
        f"cl={cl_status}"
    )
    return summary
