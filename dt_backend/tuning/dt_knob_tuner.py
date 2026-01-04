"""dt_backend.tuning.dt_knob_tuner — Phase 6 (Auto-knob tuner v1)

This is the DT twin of backend.tuning.swing_knob_tuner.

DT reality check: different installs may or may not have DT missed-opportunity
reports wired up yet. Therefore this tuner is extremely defensive:
  * If the report is missing or empty: skip.
  * One knob change max per run.
  * Hard bounds on every knob.

Default inputs
--------------
Looks for a DT missed report at:
  da_brains/intraday/dt_missed_report.json

Outputs
-------
Writes per-profile overrides:
  ml_data_dt/config/dt_knob_overrides.json
and appends:
  ml_data_dt/config/dt_tuning_log.jsonl
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _truthy(v: str) -> bool:
    v = (v or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _read_json(path: Path, default: Any = None) -> Any:
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
    except Exception:
        return


def _append_jsonl(path: Path, obj: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        return


def _paths() -> Tuple[Path, Path, Path]:
    rp = (os.getenv("DT_MISSED_REPORT_PATH", "") or "").strip()
    op = (os.getenv("DT_KNOB_OVERRIDES_PATH", "") or "").strip()
    lp = (os.getenv("DT_TUNING_LOG_PATH", "") or "").strip()
    if rp and op and lp:
        return Path(rp), Path(op), Path(lp)

    try:
        from dt_backend.core import DT_PATHS

        overrides = DT_PATHS.get("dt_knob_overrides")
        logp = DT_PATHS.get("dt_tuning_log")
        report = Path("da_brains") / "intraday" / "dt_missed_report.json"
        if isinstance(overrides, Path) and isinstance(logp, Path):
            return report, overrides, logp
    except Exception:
        pass

    return (
        Path("da_brains") / "intraday" / "dt_missed_report.json",
        Path("ml_data_dt") / "config" / "dt_knob_overrides.json",
        Path("ml_data_dt") / "config" / "dt_tuning_log.jsonl",
    )



# ============================================================
#  PHASE 7 — EXPLORATION BUDGET
# ============================================================

def _state_path_fallback() -> Path:
    try:
        from dt_backend.core import DT_PATHS
        p = DT_PATHS.get("dt_tuner_state")
        if isinstance(p, Path):
            return p
    except Exception:
        pass
    return Path("ml_data_dt") / "config" / "dt_tuner_state.json"


def _budget_path_fallback() -> Path:
    try:
        from dt_backend.core import DT_PATHS
        p = DT_PATHS.get("dt_exploration_budget")
        if isinstance(p, Path):
            return p
    except Exception:
        pass
    return Path("ml_data_dt") / "config" / "exploration_budget.json"


def _period_key(period: str) -> str:
    now = datetime.now(timezone.utc)
    p = (period or "daily").lower()
    if p.startswith("week"):
        y, w, _ = now.isocalendar()
        return f"{y}-W{int(w):02d}"
    return now.date().isoformat()


def _load_budget() -> Dict[str, Any]:
    obj = _read_json(_budget_path_fallback(), {})
    return obj if isinstance(obj, dict) else {}


def _load_state() -> Dict[str, Any]:
    obj = _read_json(_state_path_fallback(), {})
    return obj if isinstance(obj, dict) else {}


def _save_state(state: Dict[str, Any]) -> None:
    _atomic_write_json(_state_path_fallback(), state)


def _can_explore(budget: Dict[str, Any], state: Dict[str, Any]) -> bool:
    if not _truthy(os.getenv("AION_TUNER_ENABLE_EXPLORATION", "1")):
        return False
    if isinstance(budget, dict) and not bool(budget.get("enabled", True)):
        return False
    period = str(budget.get("period") or "daily")
    key = _period_key(period)
    used = int(_safe_float(state.get("used", 0), 0))
    maxn = int(_safe_float(budget.get("max_experiments", 0), 0))
    if str(state.get("period_key") or "") != key:
        used = 0
    return used < maxn


def _mark_explore_used(budget: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    period = str(budget.get("period") or "daily")
    key = _period_key(period)
    used = int(_safe_float(state.get("used", 0), 0))
    if str(state.get("period_key") or "") != key:
        used = 0
    used += 1
    out = dict(state)
    out.update({"period": period, "period_key": key, "used": used, "updated_at": _utc_iso()})
    return out


def _pick_exploration_change(budget: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cands = budget.get("candidate_knobs")
    if not isinstance(cands, list) or not cands:
        cands = ["DT_STRAT_MIN_CONF", "DT_STRAT_MIN_SCORE"]

    steps = budget.get("steps") if isinstance(budget.get("steps"), dict) else {}
    bounds = budget.get("bounds") if isinstance(budget.get("bounds"), dict) else {}

    options = []
    for k in cands:
        if not isinstance(k, str):
            continue
        if k == "DT_STRAT_MIN_CONF":
            cur = _get_env_float(k, 0.32)
            step = float(steps.get(k, 0.01))
            lo, hi = bounds.get(k, [0.15, 0.60])
            new = _bounded(cur, cur - step, lo, hi)
            options.append((k, cur, new))
        elif k == "DT_STRAT_MIN_SCORE":
            cur = _get_env_float(k, 7.0)
            step = float(steps.get(k, 0.25))
            lo, hi = bounds.get(k, [3.0, 12.0])
            new = _bounded(cur, cur - step, lo, hi)
            options.append((k, cur, new))

    best = None
    best_gap = -1.0
    for k, cur, new in options:
        gap = abs(cur - new)
        if gap > best_gap and abs(new - cur) >= 1e-9:
            best_gap = gap
            best = (k, cur, new)
    if not best:
        return None
    k, cur, new = best
    return {"knob": k, "old": cur, "new": new, "why": "Exploration budget micro-loosen (Phase 7)."}

def _active_profile() -> str:
    env = (os.getenv("DT_PROFILE", "") or "").strip()
    if env:
        return env
    # Optional profile pin file: ml_data_dt/config/dt_knob_profiles/_active_profile.json
    try:
        p = Path("ml_data_dt") / "config" / "dt_knob_profiles" / "_active_profile.json"
        obj = _read_json(p, {})
        if isinstance(obj, dict):
            v = (obj.get("profile") or "").strip()
            if v:
                return str(v)
    except Exception:
        pass
    return "default"


@dataclass
class ReasonStats:
    n: int
    hit_rate: float


def _extract_reason_stats(report: Dict[str, Any]) -> Dict[str, ReasonStats]:
    # Accept either a rich dict, or a simplified {"top_reasons": [[reason,count],...]}
    out: Dict[str, ReasonStats] = {}
    by_reason = report.get("by_reason")
    if isinstance(by_reason, dict):
        for r, v in by_reason.items():
            if not isinstance(r, str):
                continue
            if isinstance(v, dict):
                n = int(_safe_float(v.get("n"), _safe_float(v.get("count"), 0)))
                hr = float(_safe_float(v.get("hit_rate"), _safe_float(v.get("win_rate"), 0)))
                if n > 0:
                    out[r] = ReasonStats(n=n, hit_rate=hr)
        return out

    top = report.get("top_reasons")
    if isinstance(top, list):
        for row in top:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                r = str(row[0])
                n = int(_safe_float(row[1], 0))
                # No hit rate available; treat as unknown.
                out[r] = ReasonStats(n=n, hit_rate=0.0)
    return out


def _bounded(current: float, new: float, lo: float, hi: float) -> float:
    try:
        return max(float(lo), min(float(hi), float(new)))
    except Exception:
        return float(current)


def _get_env_float(name: str, fallback: float) -> float:
    try:
        raw = (os.getenv(name, "") or "").strip()
        return float(raw) if raw else float(fallback)
    except Exception:
        return float(fallback)


def _pick_change(stats: Dict[str, ReasonStats]) -> Optional[Dict[str, Any]]:
    ranked = sorted(stats.items(), key=lambda kv: (kv[1].n, kv[1].hit_rate), reverse=True)
    for reason, st in ranked[:30]:
        # If we have hit_rate, demand evidence.
        if st.hit_rate > 0 and (st.n < 50 or st.hit_rate < 0.60):
            continue
        if st.n < 80:
            continue
        r = reason.lower()

        # Two high-leverage, universal DT gates (exist in strategy_engine_dt.py)
        if "conf" in r:
            cur = _get_env_float("DT_STRAT_MIN_CONF", 0.32)
            new = _bounded(cur, cur - 0.02, 0.15, 0.60)
            if abs(new - cur) >= 1e-9:
                return {"knob": "DT_STRAT_MIN_CONF", "old": cur, "new": new, "why": f"Many rejects due to {reason}."}

        if "score" in r:
            cur = _get_env_float("DT_STRAT_MIN_SCORE", 7.0)
            new = _bounded(cur, cur - 0.5, 3.0, 12.0)
            if abs(new - cur) >= 1e-9:
                return {"knob": "DT_STRAT_MIN_SCORE", "old": cur, "new": new, "why": f"Many rejects due to {reason}."}

    # If we have no good reason-based signal, do nothing.
    return None


def _cooldown_ok(log_path: Path, profile: str, *, hours: int = 24) -> bool:
    try:
        if not log_path.exists():
            return True
        lines = log_path.read_text(encoding="utf-8").splitlines()[-400:]
        last: Optional[datetime] = None
        for line in reversed(lines):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            if str(obj.get("profile") or "") != str(profile):
                continue
            if str(obj.get("action") or "") not in {"change", "rollback"}:
                continue
            ts = obj.get("ts")
            if not ts:
                continue
            s = str(ts)
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            try:
                last = datetime.fromisoformat(s)
            except Exception:
                last = None
            break
        if last is None:
            return True
        age = (datetime.now(timezone.utc) - last.astimezone(timezone.utc)).total_seconds()
        return age >= float(hours * 3600)
    except Exception:
        return True


def run_dt_knob_tuner(*, dry_run: bool = False) -> Dict[str, Any]:
    if _truthy(os.getenv("AION_DISABLE_AUTO_TUNER", "")):
        return {"status": "disabled"}

    report_path, overrides_path, log_path = _paths()
    profile = _active_profile()

    report = _read_json(report_path, {})
    if not isinstance(report, dict) or not report:
        return {"status": "skipped", "reason": "no_report", "report_path": str(report_path)}

    stats = _extract_reason_stats(report)
    if not stats:
        return {"status": "skipped", "reason": "no_reason_stats", "report_path": str(report_path)}

    if not _cooldown_ok(log_path, profile, hours=int(_safe_float(os.getenv("AION_TUNER_COOLDOWN_HOURS", "24"), 24))):
        return {"status": "skipped", "reason": "cooldown", "profile": profile}

change = _pick_change(stats)
mode = "evidence"
if not change:
    budget = _load_budget()
    state = _load_state()
    only_when_no_evidence = _truthy(os.getenv("AION_TUNER_EXPLORATION_ONLY_WHEN_NO_EVIDENCE", "1"))
    if only_when_no_evidence and _can_explore(budget, state):
        change = _pick_exploration_change(budget)
        if change:
            mode = "explore"
            state2 = _mark_explore_used(budget, state)
            if not dry_run:
                _save_state(state2)
    if not change:
        return {"status": "skipped", "reason": "no_safe_change", "profile": profile}

    existing = _read_json(overrides_path, {})
    if not isinstance(existing, dict):
        existing = {}

    profiles = existing.get("profiles") if isinstance(existing.get("profiles"), dict) else {}
    if not isinstance(profiles, dict):
        profiles = {}
    prof_over = profiles.get(profile)
    if not isinstance(prof_over, dict):
        prof_over = {}
    prof_over[str(change["knob"])] = float(change["new"]) if isinstance(change["new"], (int, float)) else change["new"]
    profiles[profile] = prof_over

    new_file = {
        "version": int(existing.get("version") or 1),
        "updated_at": _utc_iso(),
        "profile": profile,
        "profiles": profiles,
        "note": "Auto-generated by dt_knob_tuner (Phase 6).",
    }

    log_row = {
        "ts": _utc_iso(),
        "engine": "dt",
        "profile": profile,
        "action": "change",
        "mode": mode,
        "knob": change["knob"],
        "old": change["old"],
        "new": change["new"],
        "why": change.get("why"),
        "report_ts": report.get("ts"),
    }

    if not dry_run:
        _atomic_write_json(overrides_path, new_file)
        _append_jsonl(log_path, log_row)
        try:
            os.environ[str(change["knob"])] = str(change["new"])
        except Exception:
            pass

    return {
        "status": "ok" if not dry_run else "dry_run",
        "profile": profile,
        "knob": change["knob"],
        "old": change["old"],
        "new": change["new"],
        "overrides_path": str(overrides_path),
        "log_path": str(log_path),
        "report_path": str(report_path),
    }


def main() -> None:
    out = run_dt_knob_tuner(dry_run=_truthy(os.getenv("AION_TUNER_DRY_RUN", "")))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
