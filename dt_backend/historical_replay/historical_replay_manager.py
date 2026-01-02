# dt_backend/historical_replay/historical_replay_manager.py
"""Historical replay manager (Phase 6).

Runs intraday replays across multiple days, aggregates performance, tracks
progress in a durable replay_state.json, and appends run summaries to
replay_log.json.

Replay state rules
------------------
- If replay_state.json says status="complete" and version matches the current replay version, the manager refuses to rerun.
- If version changes, the manager resets the state and starts clean.
- If status is incomplete and version matches, it resumes from the next day after last_completed.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dt_backend.core.config_dt import DT_PATHS
from dt_backend.core.data_pipeline_dt import log
from .historical_replay_engine import replay_intraday_day, ReplayResult


@dataclass
class ReplaySummary:
    start_date: str
    end_date: str
    n_days: int
    total_gross_pnl: float
    avg_pnl_per_day: float
    avg_pnl_per_trade: float
    avg_hit_rate: float
    breakdown_by_regime: Dict[str, Any]
    days: List[Dict[str, Any]]


def _replay_root() -> Path:
    root = Path(DT_PATHS.get("dtml_data", Path("ml_data_dt")))
    return root / "intraday" / "replay"


def _raw_days_dir() -> Path:
    return _replay_root() / "raw_days"


def _replay_state_path() -> Path:
    return _replay_root() / "replay_state.json"


def _discover_dates() -> List[str]:
    """Discover available replay days from raw_days directory."""
    d = _raw_days_dir()
    if not d.exists():
        return []

    dates: set[str] = set()
    for p in d.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if name.endswith(".json.gz"):
            dates.add(name[:-len(".json.gz")])
        elif name.endswith(".json"):
            dates.add(name[:-len(".json")])
    return sorted(dates)


def _load_state() -> Dict[str, Any]:
    p = _replay_state_path()
    try:
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    p = _replay_state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        state = state if isinstance(state, dict) else {}
        state["ts"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception as e:
        log(f"[replay_manager] ⚠️ failed to write replay_state: {e}")


def _default_version() -> str:
    return (os.getenv("DT_REPLAY_VERSION", "phase6_v1") or "phase6_v1").strip()


def _tally_regimes(results: List[ReplayResult]) -> Dict[str, Any]:
    by_label: Dict[str, Dict[str, float]] = {}

    def add(label: str, gross: float, trades: int, hit_rate: float) -> None:
        label = (label or "UNKNOWN").upper()
        d = by_label.setdefault(label, {"days": 0.0, "gross_pnl": 0.0, "trades": 0.0, "hit_w": 0.0})
        d["days"] += 1.0
        d["gross_pnl"] += float(gross)
        d["trades"] += float(trades)
        d["hit_w"] += float(hit_rate) * float(max(1, trades))

    for r in results:
        label = "UNKNOWN"
        try:
            reg_dt = (r.meta or {}).get("regime_dt")
            if isinstance(reg_dt, dict) and reg_dt.get("label"):
                label = str(reg_dt.get("label"))
        except Exception:
            label = "UNKNOWN"
        add(label, r.gross_pnl, r.n_trades, r.hit_rate)

    out: Dict[str, Any] = {}
    for label, d in by_label.items():
        days = int(d["days"])
        trades = int(d["trades"])
        out[label] = {
            "days": days,
            "gross_pnl": float(d["gross_pnl"]),
            "avg_pnl_per_day": float(d["gross_pnl"]) / days if days else 0.0,
            "avg_pnl_per_trade": float(d["gross_pnl"]) / trades if trades else 0.0,
            "trades": trades,
            "avg_hit_rate": float(d["hit_w"]) / trades if trades else 0.0,
        }
    return out


def run_replay_range(
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    version: Optional[str] = None,
    resume: bool = True,
) -> ReplaySummary | None:
    version = (version or _default_version()).strip() or "phase6_v1"

    dates = _discover_dates()
    if not dates:
        log("[replay_manager] ⚠️ no raw days found.")
        return None

    if start:
        dates = [d for d in dates if d >= start]
    if end:
        dates = [d for d in dates if d <= end]

    if not dates:
        log("[replay_manager] ⚠️ nothing in requested range.")
        return None

    # --- state handling ---
    state = _load_state()
    st_version = str(state.get("version") or "")
    st_status = str(state.get("status") or "").lower()

    if st_status == "complete" and st_version == version:
        log(f"[replay_manager] ✅ replay_state says COMPLETE for version={version}; refusing to rerun.")
        return None

    # Reset if version changed
    if st_version and st_version != version:
        log(f"[replay_manager] 🔁 version changed {st_version} → {version}; resetting replay_state.")
        state = {}

    # Resume if incomplete and same version
    if resume and st_status == "incomplete" and st_version == version:
        last_done = str(state.get("last_completed") or "").strip()
        if last_done:
            dates = [d for d in dates if d > last_done]

    # Initialize state for this run
    state.update({
        "version": version,
        "status": "incomplete",
        "start_date": dates[0],
        "end_date": dates[-1],
        "last_completed": state.get("last_completed"),
        "completed_days": int(state.get("completed_days") or 0),
        "notes": state.get("notes") or "",
    })
    _save_state(state)

    # --- run replays ---
    results: List[ReplayResult] = []
    for ds in dates:
        r = replay_intraday_day(ds)
        if r:
            results.append(r)
            state["last_completed"] = ds
            state["completed_days"] = int(state.get("completed_days") or 0) + 1
            _save_state(state)

    if not results:
        log("[replay_manager] ⚠️ no successful replays.")
        return None

    total_gross = sum(r.gross_pnl for r in results)
    total_trades = sum(r.n_trades for r in results)
    total_hit_weighted = sum(r.hit_rate * max(1, r.n_trades) for r in results)

    n_days = len(results)
    avg_pnl_day = total_gross / n_days if n_days > 0 else 0.0
    avg_pnl_trade = total_gross / total_trades if total_trades > 0 else 0.0
    avg_hit = total_hit_weighted / total_trades if total_trades > 0 else 0.0

    breakdown = _tally_regimes(results)

    summary = ReplaySummary(
        start_date=results[0].date,
        end_date=results[-1].date,
        n_days=n_days,
        total_gross_pnl=float(total_gross),
        avg_pnl_per_day=float(avg_pnl_day),
        avg_pnl_per_trade=float(avg_pnl_trade),
        avg_hit_rate=float(avg_hit),
        breakdown_by_regime=breakdown,
        days=[asdict(r) for r in results],
    )

    # Mark state complete
    state["status"] = "complete"
    state["last_completed"] = summary.end_date
    _save_state(state)

    # Append to replay_log.json
    log_path = _replay_root() / "replay_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        existing = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {"meta": {}, "runs": []}
    except Exception:
        existing = {"meta": {}, "runs": []}

    existing.setdefault("meta", {})
    existing.setdefault("runs", [])
    existing["runs"].append({"version": version, **asdict(summary)})
    existing["meta"]["ts_last_run"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    log_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    log(
        f"[replay_manager] ✅ replayed {summary.n_days} days ({summary.start_date}→{summary.end_date}), "
        f"P={summary.total_gross_pnl:.4f}, avg/day={summary.avg_pnl_per_day:.4f}"
    )

    return summary


def main() -> None:
    run_replay_range()


if __name__ == "__main__":
    main()
