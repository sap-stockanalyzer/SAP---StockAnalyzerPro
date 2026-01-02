# dt_backend/ab/promotion_gate_dt.py — Phase 6.5
"""Promotion gate for A/B upgrades.

This does **not** auto-deploy. It produces a decision report.

Inputs are metrics JSON files produced by either:
  - dt_truth_store.write_metrics_snapshot (live-ish)
  - historical replay runner outputs

We score versions using conservative KPIs:
  - expectancy (avg R)
  - max drawdown (R)
  - trade count (for significance)
  - win_rate (supporting)

Promotion rule (v1)
-------------------
Promote if:
  - candidate expectancy > baseline expectancy + min_lift
  - candidate drawdown <= baseline drawdown * max_dd_mult
  - candidate trades >= min_trades

Usage
-----
python -m dt_backend.ab.promotion_gate_dt \
  --baseline path/to/baseline_metrics.json \
  --candidate path/to/candidate_metrics.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _get(obj: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    cur: Any = obj
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return float(default)
    try:
        return float(cur)
    except Exception:
        return float(default)


@dataclass
class GateConfig:
    min_trades: int = 80
    min_lift_expectancy: float = 0.05  # R
    max_dd_mult: float = 1.05          # allow +5% DD


def decide_promotion(baseline: Dict[str, Any], candidate: Dict[str, Any], cfg: GateConfig) -> Dict[str, Any]:
    # Try a few common key layouts.
    b_exp = _get(baseline, "expectancy", default=_get(baseline, "summary", "expectancy", default=0.0))
    c_exp = _get(candidate, "expectancy", default=_get(candidate, "summary", "expectancy", default=0.0))
    b_dd = _get(baseline, "max_drawdown_r", default=_get(baseline, "summary", "max_drawdown_r", default=0.0))
    c_dd = _get(candidate, "max_drawdown_r", default=_get(candidate, "summary", "max_drawdown_r", default=0.0))
    b_tr = int(_get(baseline, "trades", default=_get(baseline, "summary", "trades", default=0.0)))
    c_tr = int(_get(candidate, "trades", default=_get(candidate, "summary", "trades", default=0.0)))
    b_wr = _get(baseline, "win_rate", default=_get(baseline, "summary", "win_rate", default=0.0))
    c_wr = _get(candidate, "win_rate", default=_get(candidate, "summary", "win_rate", default=0.0))

    reasons = []
    promote = True

    if c_tr < cfg.min_trades:
        promote = False
        reasons.append(f"insufficient_trades:{c_tr}<{cfg.min_trades}")

    if (c_exp - b_exp) < cfg.min_lift_expectancy:
        promote = False
        reasons.append(f"expectancy_lift_too_small:{c_exp-b_exp:.3f}<{cfg.min_lift_expectancy}")

    # Lower (more negative) drawdown is worse. We assume DD is positive magnitude.
    if b_dd > 0 and c_dd > (b_dd * cfg.max_dd_mult):
        promote = False
        reasons.append(f"drawdown_worse:{c_dd:.3f}>{b_dd*cfg.max_dd_mult:.3f}")

    if not reasons:
        reasons.append("passed")

    return {
        "promote": bool(promote),
        "reasons": reasons,
        "baseline": {"expectancy": b_exp, "max_drawdown_r": b_dd, "trades": b_tr, "win_rate": b_wr},
        "candidate": {"expectancy": c_exp, "max_drawdown_r": c_dd, "trades": c_tr, "win_rate": c_wr},
        "cfg": {
            "min_trades": cfg.min_trades,
            "min_lift_expectancy": cfg.min_lift_expectancy,
            "max_dd_mult": cfg.max_dd_mult,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--min-trades", type=int, default=GateConfig.min_trades)
    ap.add_argument("--min-lift", type=float, default=GateConfig.min_lift_expectancy)
    ap.add_argument("--max-dd-mult", type=float, default=GateConfig.max_dd_mult)
    args = ap.parse_args()

    base = _read_json(args.baseline)
    cand = _read_json(args.candidate)
    cfg = GateConfig(min_trades=args.min_trades, min_lift_expectancy=args.min_lift, max_dd_mult=args.max_dd_mult)
    decision = decide_promotion(base, cand, cfg)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
