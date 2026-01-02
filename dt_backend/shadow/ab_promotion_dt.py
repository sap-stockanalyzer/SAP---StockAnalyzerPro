"""dt_backend/shadow/ab_promotion_dt.py — Phase 6.5

A/B promotion helper.

Shadow mode gives you *decisions*; replay gives you *outcomes*. This helper
compares two replay metric packs and writes a simple promotion decision...

Usage
-----
python -m dt_backend.shadow.ab_promotion_dt \
  --baseline /path/to/baseline_dt_trades.jsonl \
  --candidate /path/to/candidate_dt_trades.jsonl \
  --out /path/to/dt_promotions.json

This is intentionally conservative: it refuses to promote if either gate fails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from dt_backend.historical_replay.replay_metrics_dt import metrics_from_dt_trades_file


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


@dataclass
class PromotionDecision:
    promote: bool
    reasons: list[str]
    baseline: Dict[str, Any]
    candidate: Dict[str, Any]
    thresholds: Dict[str, Any]


def decide_promotion(
    baseline_pack: Dict[str, Any],
    candidate_pack: Dict[str, Any],
    *,
    min_expectancy_delta: float = 0.02,
    max_drawdown_increase: float = 1.25,
) -> PromotionDecision:
    """Return a conservative promotion decision."""

    reasons: list[str] = []

    b_gate = (baseline_pack.get("gate") or {}).get("passed") is True
    c_gate = (candidate_pack.get("gate") or {}).get("passed") is True
    if not b_gate:
        reasons.append("baseline_gate_failed")
    if not c_gate:
        reasons.append("candidate_gate_failed")

    b_m = baseline_pack.get("metrics") if isinstance(baseline_pack.get("metrics"), dict) else {}
    c_m = candidate_pack.get("metrics") if isinstance(candidate_pack.get("metrics"), dict) else {}

    b_exp = _safe_float(b_m.get("expectancy"), 0.0)
    c_exp = _safe_float(c_m.get("expectancy"), 0.0)
    b_dd = _safe_float(b_m.get("drawdown"), 0.0)
    c_dd = _safe_float(c_m.get("drawdown"), 0.0)

    if c_exp < b_exp + min_expectancy_delta:
        reasons.append(f"insufficient_expectancy_gain({c_exp:.3f} < {b_exp:.3f}+{min_expectancy_delta})")

    # Avoid promoting if drawdown balloons.
    if b_dd > 0 and c_dd > b_dd * max_drawdown_increase:
        reasons.append(f"drawdown_worse({c_dd:.2f} > {b_dd:.2f}*{max_drawdown_increase})")

    return PromotionDecision(
        promote=len(reasons) == 0,
        reasons=reasons,
        baseline={"metrics": b_m, "gate": baseline_pack.get("gate"), "source": baseline_pack.get("source")},
        candidate={"metrics": c_m, "gate": candidate_pack.get("gate"), "source": candidate_pack.get("source")},
        thresholds={
            "min_expectancy_delta": min_expectancy_delta,
            "max_drawdown_increase": max_drawdown_increase,
        },
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="dt_trades.jsonl for baseline")
    ap.add_argument("--candidate", required=True, help="dt_trades.jsonl for candidate")
    ap.add_argument("--out", required=True, help="output json file")
    ap.add_argument("--min_delta", type=float, default=0.02)
    ap.add_argument("--max_dd_mult", type=float, default=1.25)
    args = ap.parse_args()

    b = metrics_from_dt_trades_file(Path(args.baseline))
    c = metrics_from_dt_trades_file(Path(args.candidate))
    dec = decide_promotion(b, c, min_expectancy_delta=float(args.min_delta), max_drawdown_increase=float(args.max_dd_mult))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(dec), indent=2), encoding="utf-8")
    print(f"promote={dec.promote} reasons={dec.reasons} -> {out}")


if __name__ == "__main__":
    main()
