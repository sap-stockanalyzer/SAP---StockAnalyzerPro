# dt_backend/historical_replay/replay_runner_dt.py
"""Phase 6 replay/backtest harness with checkpoints.

Runs day-by-day step replays and writes:
  - per-day metrics JSON
  - run_summary JSON
  - replay_state.json (resume/checkpoint)

Usage (example):
  python -m dt_backend.historical_replay.replay_runner_dt \
    --start 2025-12-01 --end 2025-12-31 \
    --step-minutes 5 --version dt_v1

Important
---------
This tool defaults to a *separate artifact root* under:
  <ml_data_dt>/intraday/replay/runs/<run_id>/...

So it won’t touch your live DT artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional

from dt_backend.core.config_dt import DT_PATHS
from dt_backend.core.logger_dt import log

from dt_backend.historical_replay.step_replay_engine_dt import StepReplayConfig, replay_intraday_day_step
from dt_backend.historical_replay.replay_metrics_dt import compute_metrics_from_trades


@dataclass
class ReplayState:
    version: str
    status: str  # INCOMPLETE|COMPLETE
    start_date: str
    end_date: str
    next_date: str
    run_id: str
    updated_at: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_date(s: str) -> date:
    return date.fromisoformat(s.strip())


def _date_range(start: date, end: date) -> List[date]:
    out: List[date] = []
    d = start
    while d <= end:
        out.append(d)
        d = d + timedelta(days=1)
    return out


def _runs_root() -> Path:
    root = Path(DT_PATHS.get("dtml_data", Path("ml_data_dt")))
    return root / "intraday" / "replay" / "runs"


def _load_state(state_path: Path) -> Optional[ReplayState]:
    try:
        if not state_path.exists():
            return None
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return ReplayState(**data)
    except Exception:
        return None


def _save_state(state_path: Path, st: ReplayState) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(asdict(st), indent=2, sort_keys=True), encoding="utf-8")


def _init_run(start: str, end: str, version: str) -> ReplayState:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ReplayState(
        version=version,
        status="INCOMPLETE",
        start_date=start,
        end_date=end,
        next_date=start,
        run_id=run_id,
        updated_at=_utc_now_iso(),
    )


def _configure_env(run_dir: Path, day: str) -> Dict[str, str]:
    """Set environment overrides for replay-safe artifacts."""
    # Everything per-day so logs stay human readable.
    day_dir = run_dir / "truth" / day
    day_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "DT_TRUTH_DIR": str(day_dir),
        "DT_ROLLING_PATH": str(run_dir / "rolling" / f"{day}.rolling.json.gz"),
        "DT_LOCK_PATH": str(run_dir / "rolling" / f"{day}.rolling.lock"),
        "DT_BOT_LEDGER_PATH": str(run_dir / "brokers" / f"{day}.ledger.json"),
        # Hard-disable remote broker in replay. BrokerAPI will fall back to local ledger.
        "ALPACA_API_KEY_ID": "",
        "ALPACA_API_SECRET_KEY": "",
    }
    (run_dir / "rolling").mkdir(parents=True, exist_ok=True)
    (run_dir / "brokers").mkdir(parents=True, exist_ok=True)
    return env


def run_replay(
    *,
    start: str,
    end: str,
    step_minutes: int,
    version: str,
    resume: bool,
    force_restart: bool,
    max_days: Optional[int] = None,
) -> Dict[str, Any]:
    runs_root = _runs_root()
    runs_root.mkdir(parents=True, exist_ok=True)

    state_path = runs_root / "replay_state.json"
    existing = _load_state(state_path)

    if existing and existing.status == "COMPLETE" and existing.version == version and not force_restart:
        return {
            "status": "complete",
            "message": "Replay already complete for this version. Use --force-restart to rerun.",
            "run_id": existing.run_id,
        }

    if existing and resume and existing.status != "COMPLETE" and existing.version == version and not force_restart:
        st = existing
        log(f"[replay_runner] ↩️ resuming run_id={st.run_id} from {st.next_date}")
    else:
        st = _init_run(start, end, version)
        log(f"[replay_runner] ▶️ starting new run_id={st.run_id} ({start} → {end})")

    run_dir = runs_root / st.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy state into run folder too (handy for archiving)
    _save_state(state_path, st)
    _save_state(run_dir / "replay_state.json", st)

    dates = _date_range(_parse_date(st.next_date), _parse_date(st.end_date))
    if max_days is not None:
        dates = dates[: max(0, int(max_days))]

    day_reports: List[Dict[str, Any]] = []

    for d in dates:
        day = d.isoformat()

        # Env overrides (replay-safe)
        env = _configure_env(run_dir, day)
        old_env = {k: os.environ.get(k) for k in env.keys()}
        os.environ.update(env)

        try:
            cfg = StepReplayConfig(step_minutes=int(step_minutes), max_symbols=None)
            out = replay_intraday_day_step(day, cfg=cfg)

            trades_path = Path(env["DT_TRUTH_DIR"]) / "dt_trades.jsonl"
            metrics = compute_metrics_from_trades(trades_path)

            report = {"date": day, "engine": out, "metrics": metrics}
            day_reports.append(report)

            # Write per-day report
            rep_dir = run_dir / "reports"
            rep_dir.mkdir(parents=True, exist_ok=True)
            (rep_dir / f"{day}.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

            # Advance checkpoint
            st.next_date = (d + timedelta(days=1)).isoformat()
            st.updated_at = _utc_now_iso()
            _save_state(state_path, st)
            _save_state(run_dir / "replay_state.json", st)

            log(f"[replay_runner] ✅ {day} done: trades={metrics.get('trades',0)} avgR={metrics.get('avg_R',0):.3f}")

        finally:
            # Restore env
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    # Final summary
    summary = {
        "run_id": st.run_id,
        "version": st.version,
        "start": st.start_date,
        "end": st.end_date,
        "step_minutes": step_minutes,
        "days": len(day_reports),
        "reports": day_reports,
    }

    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    # Mark complete if we covered full range
    if _parse_date(st.next_date) > _parse_date(st.end_date):
        st.status = "COMPLETE"
        st.updated_at = _utc_now_iso()
        _save_state(state_path, st)
        _save_state(run_dir / "replay_state.json", st)

    return {"status": "ok", "run_id": st.run_id, "run_dir": str(run_dir)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--step-minutes", type=int, default=5)
    ap.add_argument("--version", default="dt_replay_v1")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-restart", action="store_true")
    ap.add_argument("--max-days", type=int, default=None)

    args = ap.parse_args()

    out = run_replay(
        start=args.start,
        end=args.end,
        step_minutes=args.step_minutes,
        version=args.version,
        resume=bool(args.resume),
        force_restart=bool(args.force_restart),
        max_days=args.max_days,
    )
    log(f"[replay_runner] done: {out}")


if __name__ == "__main__":
    main()
