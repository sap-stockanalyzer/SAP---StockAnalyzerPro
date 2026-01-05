"""backend.tuning.swing_knob_tuner — Phase 6 (Auto-knob tuner v1)

Goal
----
Use missed-opportunity analytics to *safely* loosen or tighten a small set of
swing knobs over time — without turning the system into a reckless gremlin.

Key properties
--------------
* Bounded: every knob has a min/max.
* Slow: at most one knob change per run.
* Reversible: append-only tune log + easy rollback.
* Best-effort: never raises.

Inputs
------
We prefer the compact report produced by:
  backend.analytics.missed_opportunities_swing.evaluate_missed_opportunities

Output
------
Writes (per profile) a JSON overrides file:
  ml_data/config/swing_knob_overrides.json
and appends to:
  ml_data/config/swing_tuning_log.jsonl
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



# ============================================================
#  PHASE 7 — EXPLORATION BUDGET
# ============================================================

def _state_path_fallback() -> Path:
    try:
        from backend.core.config import PATHS
        p = PATHS.get("swing_tuner_state")
        if isinstance(p, Path):
            return p
    except Exception:
        pass
    return Path("ml_data") / "config" / "swing_tuner_state.json"


def _budget_path_fallback() -> Path:
    try:
        from backend.core.config import PATHS
        p = PATHS.get("swing_exploration_budget")
        if isinstance(p, Path):
            return p
    except Exception:
        pass
    return Path("ml_data") / "config" / "exploration_budget.json"


def _period_key(period: str) -> str:
    now = datetime.now(timezone.utc)
    p = (period or "weekly").lower()
    if p.startswith("day"):
        return now.date().isoformat()
    y, w, _ = now.isocalendar()
    return f"{y}-W{int(w):02d}"


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
    period = str(budget.get("period") or "weekly")
    key = _period_key(period)
    used = int(_safe_float(state.get("used", 0), 0))
    maxn = int(_safe_float(budget.get("max_experiments", 0), 0))
    if str(state.get("period_key") or "") != key:
        used = 0
    return used < maxn


def _mark_explore_used(budget: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    period = str(budget.get("period") or "weekly")
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
        cands = ["SWING_CONF_THRESHOLD", "SWING_MIN_PHIT", "SWING_LOSS_EST_PCT"]

    steps = budget.get("steps") if isinstance(budget.get("steps"), dict) else {}
    bounds = budget.get("bounds") if isinstance(budget.get("bounds"), dict) else {}

    def cur_of(k: str) -> float:
        if k == "SWING_CONF_THRESHOLD":
            return _get_current_env_float(k, 0.32)
        if k == "SWING_MIN_PHIT":
            return _get_current_env_float(k, 0.60)
        if k == "SWING_LOSS_EST_PCT":
            return _get_current_env_float(k, 0.10)
        return _get_current_env_float(k, 0.0)

    options = []
    for k in cands:
        if not isinstance(k, str):
            continue
        cur = cur_of(k)
        step = float(steps.get(k, 0.01))
        lo, hi = bounds.get(k, [0.0, 1.0])
        new = _bounded(cur, cur - step, lo, hi)
        if abs(new - cur) >= 1e-9:
            options.append((k, cur, new))
    if not options:
        return None
    k, cur, new = sorted(options, key=lambda t: abs(t[1]-t[2]), reverse=True)[0]
    return {"knob": k, "old": cur, "new": new, "why": "Exploration budget micro-loosen (Phase 7)."}

def _paths() -> Tuple[Path, Path, Path]:
    """Return (report_path, overrides_path, log_path)."""
    # Allow direct overrides for debugging.
    rp = (os.getenv("SWING_MISSED_REPORT_PATH", "") or "").strip()
    op = (os.getenv("SWING_KNOB_OVERRIDES_PATH", "") or "").strip()
    lp = (os.getenv("SWING_TUNING_LOG_PATH", "") or "").strip()

    if rp and op and lp:
        return Path(rp), Path(op), Path(lp)

    try:
        from backend.core.config import PATHS  # shim

        report = Path("da_brains") / "swing" / "missed_opportunities.json"
        overrides = PATHS.get("swing_knob_overrides")
        logp = PATHS.get("swing_tuning_log")
        if isinstance(overrides, Path) and isinstance(logp, Path):
            return report, overrides, logp
    except Exception:
        pass

    return (
        Path("da_brains") / "swing" / "missed_opportunities.json",
        Path("ml_data") / "config" / "swing_knob_overrides.json",
        Path("ml_data") / "config" / "swing_tuning_log.jsonl",
    )


def _active_profile() -> str:
    env = (os.getenv("SWING_PROFILE", "") or "").strip()
    if env:
        return env
    # Optional: read ml_data/config/swing_knob_profiles/_active_profile.json
    try:
        from backend.core.config import PATHS

        prof_dir = PATHS.get("swing_knob_profiles_dir")
        if isinstance(prof_dir, Path):
            p = prof_dir / "_active_profile.json"
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
    avg_end_return: float


def _compute_reason_stats(report: Dict[str, Any]) -> Dict[str, ReasonStats]:
    samples = report.get("samples")
    if not isinstance(samples, list):
        return {}

    bucket: Dict[str, List[Tuple[bool, float]]] = {}
    for s in samples:
        if not isinstance(s, dict):
            continue
        r = str(s.get("reason") or "").strip()[:80] or "(none)"
        hit = bool(s.get("hit"))
        end_ret = _safe_float(s.get("end_return"), 0.0)
        bucket.setdefault(r, []).append((hit, end_ret))

    out: Dict[str, ReasonStats] = {}
    for r, rows in bucket.items():
        n = len(rows)
        if n <= 0:
            continue
        hits = sum(1 for h, _ in rows if h)
        avg_end = sum(er for _, er in rows) / max(1, n)
        out[r] = ReasonStats(n=n, hit_rate=float(hits / n), avg_end_return=float(avg_end))
    return out


def _bounded(current: float, new: float, lo: float, hi: float) -> float:
    try:
        return max(float(lo), min(float(hi), float(new)))
    except Exception:
        return float(current)


def _get_current_env_float(name: str, fallback: float) -> float:
    try:
        raw = (os.getenv(name, "") or "").strip()
        return float(raw) if raw else float(fallback)
    except Exception:
        return float(fallback)


def _pick_change(reason_stats: Dict[str, ReasonStats]) -> Optional[Dict[str, Any]]:
    """Pick a single, conservative knob change based on reasons."""
    # Prefer reasons that are common + have high hit-rate.
    ranked = sorted(
        reason_stats.items(),
        key=lambda kv: (kv[1].n, kv[1].hit_rate, kv[1].avg_end_return),
        reverse=True,
    )

    for reason, st in ranked[:25]:
        # Only act on reasonably-sized evidence.
        if st.n < 40:
            continue
        if st.hit_rate < 0.60:
            continue

        r = reason.lower()
        # 1) Confidence gate
        if "conf_below_threshold" in r or "conf" in r:
            cur = _get_current_env_float("SWING_CONF_THRESHOLD", 0.32)
            new = _bounded(cur, cur - 0.02, 0.18, 0.50)
            if abs(new - cur) >= 1e-9:
                return {
                    "knob": "SWING_CONF_THRESHOLD",
                    "old": cur,
                    "new": new,
                    "why": f"Missed candidates with '{reason}' had hit_rate={st.hit_rate:.2f} (n={st.n}).",
                }

        # 2) P(hit) gate
        if "phit_below_threshold" in r or "phit" in r:
            cur = _get_current_env_float("SWING_MIN_PHIT", 0.60)
            new = _bounded(cur, cur - 0.01, 0.50, 0.80)
            if abs(new - cur) >= 1e-9:
                return {
                    "knob": "SWING_MIN_PHIT",
                    "old": cur,
                    "new": new,
                    "why": f"Missed candidates with '{reason}' had hit_rate={st.hit_rate:.2f} (n={st.n}).",
                }

        # 3) EV gate (make it slightly easier to be positive EV)
        if "non_positive_ev" in r or "ev" in r:
            cur = _get_current_env_float("SWING_LOSS_EST_PCT", 0.10)
            new = _bounded(cur, cur - 0.01, 0.03, 0.25)
            if abs(new - cur) >= 1e-9:
                return {
                    "knob": "SWING_LOSS_EST_PCT",
                    "old": cur,
                    "new": new,
                    "why": f"Many rejects due to EV; hit_rate={st.hit_rate:.2f} suggests loss_est too pessimistic.",
                }

    return None


def _cooldown_ok(log_path: Path, profile: str, *, hours: int = 24) -> bool:
    """Prevent rapid oscillations: allow 1 change per profile per cooldown window."""
    try:
        if not log_path.exists():
            return True
        lines = log_path.read_text(encoding="utf-8").splitlines()[-400:]
        last_ts: Optional[datetime] = None
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
                last_ts = datetime.fromisoformat(s)
            except Exception:
                last_ts = None
            break
        if last_ts is None:
            return True
        age_sec = (datetime.now(timezone.utc) - last_ts.astimezone(timezone.utc)).total_seconds()
        return age_sec >= float(hours * 3600)
    except Exception:
        return True


def run_swing_knob_tuner(*, dry_run: bool = False) -> Dict[str, Any]:
    """Main entry: safe to call from nightly_job."""
    if _truthy(os.getenv("AION_DISABLE_AUTO_TUNER", "")):
        return {"status": "disabled"}

    report_path, overrides_path, log_path = _paths()
    profile = _active_profile()

    report = _read_json(report_path, {})
    if not isinstance(report, dict) or not report:
        # Try to (re)generate the report on the fly.
        try:
            from backend.analytics.missed_opportunities_swing import evaluate_missed_opportunities

            evaluate_missed_opportunities()
            report = _read_json(report_path, {})
        except Exception:
            report = report
    if not isinstance(report, dict) or not report:
        return {"status": "skipped", "reason": "no_report", "report_path": str(report_path)}

    reason_stats = _compute_reason_stats(report)
    if not reason_stats:
        return {"status": "skipped", "reason": "no_samples", "report_path": str(report_path)}

    if not _cooldown_ok(log_path, profile, hours=int(_safe_float(os.getenv("AION_TUNER_COOLDOWN_HOURS", "24"), 24))):
        return {"status": "skipped", "reason": "cooldown", "profile": profile}

    change = _pick_change(reason_stats)
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

    # Load existing overrides file
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
        # ... the rest of your file continues here, still indented ...
    }

    log_row = {
        "ts": _utc_iso(),
        "engine": "swing",
        "profile": profile,
        "action": "change",
        "mode": mode,
        "knob": change["knob"],
        "old": change["old"],
        "new": change["new"],
        "why": change.get("why"),
        "report_ts": report.get("ts"),
        "evidence": {
            "top_reason": change.get("why"),
        },
    }

    if not dry_run:
        _atomic_write_json(overrides_path, new_file)
        _append_jsonl(log_path, log_row)

        # Also apply immediately in-process for this run.
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
    out = run_swing_knob_tuner(dry_run=_truthy(os.getenv("AION_TUNER_DRY_RUN", "")))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
