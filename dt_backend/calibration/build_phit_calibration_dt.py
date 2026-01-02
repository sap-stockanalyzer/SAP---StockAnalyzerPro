# dt_backend/calibration/build_phit_calibration_dt.py — Phase 7
"""Build calibrated P(hit) tables from Phase-6 replay artifacts.

Usage
-----
python -m dt_backend.calibration.build_phit_calibration_dt \
  --trades <path/to/dt_trades.jsonl> \
  --out <truth_dir>/intraday/calibration/phit_calib.json

We keep it simple:
* For each (bot, regime) bucket, we bin base_conf into deciles.
* Hit definition: exit_reason == "take_profit".
* Calibration per bin = smoothed empirical hit rate.

This is intentionally not fancy ML — it is a reliable calibration backbone.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dt_backend.historical_replay.replay_metrics_dt import _read_jsonl, trades_from_events


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _bin_edges(n: int = 10) -> List[float]:
    return [i / float(n) for i in range(n + 1)]


def _bin_index(conf: float, bins: List[float]) -> int:
    conf = max(0.0, min(1.0, float(conf)))
    if conf <= bins[0]:
        return 0
    for i in range(len(bins) - 1):
        if bins[i] <= conf < bins[i + 1]:
            return i
    return max(0, len(bins) - 2)


def build_tables(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    trades = trades_from_events(events)
    bins = _bin_edges(10)

    # We need base_conf at entry. We grab it from bracket_set.meta if present.
    entry_conf: Dict[Tuple[str, str], float] = {}
    # key: (symbol, entry_ts) -> conf
    # We'll store by symbol only and keep last seen; replay ordering makes this usable.
    sym_last_conf: Dict[str, float] = {}
    sym_last_bot: Dict[str, str] = {}
    sym_last_reg: Dict[str, str] = {}

    for e in events:
        if str(e.get("type") or "") != "bracket_set":
            continue
        sym = str(e.get("symbol") or "").upper()
        meta = e.get("meta") if isinstance(e.get("meta"), dict) else {}
        conf = _safe_float(meta.get("base_conf"), _safe_float(meta.get("confidence"), 0.0))
        bot = str(e.get("bot") or meta.get("bot") or "").upper() or "UNKNOWN"
        reg = str(meta.get("regime") or meta.get("regime_label") or "").upper() or "UNKNOWN"
        sym_last_conf[sym] = conf
        sym_last_bot[sym] = bot
        sym_last_reg[sym] = reg

    # bucket -> bins
    # store counts and hit counts per bin
    counts: Dict[str, List[int]] = {}
    hits: Dict[str, List[int]] = {}

    for t in trades:
        sym = str(t.symbol).upper()
        bot = (str(t.bot or "") or sym_last_bot.get(sym) or "UNKNOWN").upper()
        reg = (str(t.regime or "") or sym_last_reg.get(sym) or "UNKNOWN").upper()
        key = f"{bot}|{reg}"
        conf = sym_last_conf.get(sym, 0.0)
        bi = _bin_index(conf, bins)
        counts.setdefault(key, [0] * (len(bins) - 1))
        hits.setdefault(key, [0] * (len(bins) - 1))
        counts[key][bi] += 1
        if str(t.exit_reason or "") == "take_profit":
            hits[key][bi] += 1

    tables: Dict[str, List[float]] = {}
    # Laplace smoothing to avoid 0/1 extremes on small samples
    alpha = 1.0
    beta = 1.0
    for key, c in counts.items():
        row: List[float] = []
        for i in range(len(c)):
            n = c[i]
            h = hits.get(key, [0] * len(c))[i]
            p = (h + alpha) / (n + alpha + beta) if n > 0 else 0.50
            row.append(float(max(0.0, min(1.0, p))))
        tables[key] = row

    # DEFAULT row: global aggregate
    total_c = [0] * (len(bins) - 1)
    total_h = [0] * (len(bins) - 1)
    for key in counts.keys():
        for i, n in enumerate(counts[key]):
            total_c[i] += n
            total_h[i] += hits.get(key, [0] * len(total_c))[i]
    default_row = []
    for i in range(len(total_c)):
        n = total_c[i]
        h = total_h[i]
        p = (h + alpha) / (n + alpha + beta) if n > 0 else 0.50
        default_row.append(float(max(0.0, min(1.0, p))))
    tables["DEFAULT"] = default_row

    return {
        "version": "dt_v1",
        "built_at": _utc_iso(),
        "bins": bins,
        "tables": tables,
        "stats": {
            "buckets": len([k for k in tables.keys() if k != "DEFAULT"]),
            "trades": len(trades),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True, help="Path to dt_trades.jsonl")
    ap.add_argument("--out", required=True, help="Output path for phit_calib.json")
    args = ap.parse_args()

    events = _read_jsonl(Path(args.trades))
    payload = build_tables(events)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}  buckets={payload['stats']['buckets']} trades={payload['stats']['trades']}")


if __name__ == "__main__":
    main()
