# dt_backend/bandit/contextual_bandit_dt.py — v1.0 (Phase 4.5, shadow-first)
"""Contextual bandit allocator (Phase 4.5).

Reality check (so the system stays sane)
--------------------------------------
A real contextual bandit needs clean, attributable rewards (per strategy, per
context). Live DT logging doesn't always contain that attribution yet, so this
module starts "research mode":

- Update is based on replay/backtest summaries (deterministic, attributable).
- Live mode keeps it OFF unless explicitly enabled.

This still gives you the core *plumbing*:
- a persistent bandit_state.json
- suggested bot weights that the meta-controller can optionally apply

State format
------------
{
  "ts": "...Z",
  "bots": {
     "ORB": {"mean_r": 0.12, "win_rate": 0.53, "trades": 220},
     ...
  }
}
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dt_backend.core import DT_PATHS
from dt_backend.core.logger_dt import log


BOTS = ["VWAP_MR", "ORB", "TREND_PULLBACK", "SQUEEZE"]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ml_data_root() -> Path:
    da = DT_PATHS.get("da_brains")
    return da if isinstance(da, Path) else Path("da_brains")


def _bandit_path() -> Path:
    override = (os.getenv("DT_TRUTH_DIR", "") or "").strip()
    if override:
        base = Path(override) / "intraday" / "bandit"
    else:
        base = _ml_data_root() / "intraday" / "bandit"
    base.mkdir(parents=True, exist_ok=True)
    return base / "bandit_state.json"


def _read_json(p: Path, default: Any) -> Any:
    try:
        if not p.exists():
            return default
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json_atomic(p: Path, obj: Any) -> None:
    try:
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        return


def load_bandit_state() -> Dict[str, Any]:
    st = _read_json(_bandit_path(), {})
    return st if isinstance(st, dict) else {}


def suggest_bot_weights(context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Return normalized bot weights from bandit priors.

    For now this is *not* truly contextual; it uses global bot priors.
    That's intentional: better a slightly-smart system than a confidently-wrong one.
    """
    st = load_bandit_state()
    bots = st.get("bots") if isinstance(st.get("bots"), dict) else {}

    # Fallback: uniform weights
    if not bots:
        return {b: 1.0 / len(BOTS) for b in BOTS}

    # Score = mean_r * (win_rate - 0.5) with mild trade-count confidence.
    scores: Dict[str, float] = {}
    for b in BOTS:
        bd = bots.get(b) if isinstance(bots.get(b), dict) else {}
        mean_r = float(bd.get("mean_r") or 0.0)
        win = float(bd.get("win_rate") or 0.0)
        n = float(bd.get("trades") or 0.0)
        conf = min(1.0, (n / 200.0) ** 0.5)  # saturates around ~200 trades
        score = max(0.0, mean_r) * max(0.0, win - 0.5) * (0.25 + 0.75 * conf)
        scores[b] = score

    s = sum(scores.values())
    if s <= 0:
        return {b: 1.0 / len(BOTS) for b in BOTS}
    return {b: float(scores[b] / s) for b in BOTS}


def _latest_replay_summary() -> Optional[Path]:
    """Find the most recent replay run summary.json."""
    base = _ml_data_root() / "intraday" / "replay" / "runs"
    if not base.exists():
        return None

    summaries = sorted(base.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return summaries[0] if summaries else None


def update_bandit_from_replay_summary(summary_path: Optional[Path] = None) -> Dict[str, Any]:
    """Update bandit priors from a replay summary.json.

    Expected keys (from backtest_runner_dt.py):
      summary["by_bot"][bot] = {"trades":..., "win_rate":..., "avg_r":...}

    Returns updated state summary.
    """
    p = summary_path or _latest_replay_summary()
    if p is None or not p.exists():
        return {"ok": False, "reason": "no_replay_summary"}

    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"ok": False, "reason": "bad_json"}

    by_bot = s.get("by_bot") if isinstance(s, dict) else None
    if not isinstance(by_bot, dict):
        return {"ok": False, "reason": "missing_by_bot"}

    st = load_bandit_state()
    out_bots = st.get("bots") if isinstance(st.get("bots"), dict) else {}

    updated = 0
    for b in BOTS:
        row = by_bot.get(b) if isinstance(by_bot.get(b), dict) else None
        if not isinstance(row, dict):
            continue
        trades = int(float(row.get("trades") or 0))
        win = float(row.get("win_rate") or 0.0)
        avg_r = float(row.get("avg_r") or row.get("avg_R") or 0.0)

        out_bots[b] = {
            "trades": trades,
            "win_rate": win,
            "mean_r": avg_r,
            "source": str(p),
        }
        updated += 1

    st = {
        "ts": _utc_iso(),
        "source": str(p),
        "bots": out_bots,
    }
    _write_json_atomic(_bandit_path(), st)
    log(f"[bandit] updated priors from replay summary: {p.name} (bots={updated})")
    return {"ok": True, "updated": updated, "source": str(p)}


def update_bandit_from_trades() -> Dict[str, Any]:
    """Phase 4.5 hook called from the live cycle.

    Today this simply refreshes bandit priors from the most recent replay run.
    That's conservative and keeps the system deterministic.

    Later, once live trade attribution is solid, this can incorporate live outcomes.
    """
    return update_bandit_from_replay_summary()

