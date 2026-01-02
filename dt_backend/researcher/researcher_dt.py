# dt_backend/researcher/researcher_dt.py — Phase 9 (shadow-only)
"""A conservative "researcher" that proposes *filters*, not new strategies.

It reads replay trades and entry-time metadata, then suggests simple
rule candidates like:

  - disable ORB in HIGH_VOL regimes
  - require rel_volume >= X for ORB
  - disable VWAP_MR during LUNCH

Rules are output as JSON so the policy engine can enforce them.

This layer is intentionally boxed in:
* It cannot create new bots.
* It cannot touch order execution.
* It only proposes gating/filters, and ONLY in shadow mode by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dt_backend.historical_replay.replay_metrics_dt import _read_jsonl, trades_from_events


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _wins(trade) -> bool:
    try:
        return float(trade.pnl) > 0
    except Exception:
        return False


@dataclass
class FilterRule:
    id: str
    bot: str
    kind: str  # "DISABLE_WHEN" | "MIN_FEATURE"
    when: Dict[str, Any]
    params: Dict[str, Any]
    rationale: str
    support: Dict[str, Any]


def _collect_entry_meta(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """symbol -> last bracket_set meta"""
    out: Dict[str, Dict[str, Any]] = {}
    for e in events:
        if str(e.get("type") or "") != "bracket_set":
            continue
        sym = str(e.get("symbol") or "").upper()
        meta = e.get("meta") if isinstance(e.get("meta"), dict) else {}
        if sym:
            out[sym] = meta
    return out


def propose_filters(
    *,
    dt_trades_path: Path,
    min_trades_per_slice: int = 25,
    min_loss_edge: float = 0.12,
) -> Dict[str, Any]:
    """Return a rule proposal pack."""

    events = _read_jsonl(dt_trades_path)
    trades = trades_from_events(events)
    meta_by_sym = _collect_entry_meta(events)

    # Aggregate per (bot, regime) and (bot, micro)
    buckets: Dict[Tuple[str, str, str], List[Any]] = {}
    for t in trades:
        bot = str(t.bot or "UNKNOWN").upper()
        reg = str(t.regime or "UNKNOWN").upper()
        micro = str(t.micro or "UNKNOWN").upper()
        buckets.setdefault((bot, "REGIME", reg), []).append(t)
        buckets.setdefault((bot, "MICRO", micro), []).append(t)

    rules: List[FilterRule] = []

    # 1) Disable bot in regime/micro slices with strongly negative expectancy.
    for (bot, dim, val), ts in buckets.items():
        n = len(ts)
        if n < min_trades_per_slice:
            continue
        avg_r = sum(_safe_float(t.r_multiple, 0.0) for t in ts) / max(1, n)
        win_rate = sum(1 for t in ts if _wins(t)) / max(1, n)
        # "loss edge": how bad is it vs flat
        loss_edge = max(0.0, -avg_r)
        if loss_edge < min_loss_edge:
            continue

        kind = "DISABLE_WHEN"
        when = {"dim": dim, "value": val}
        rid = f"disable_{bot}_{dim.lower()}_{val.lower()}"
        rationale = f"{bot} underperforms in {dim}={val} (avgR={avg_r:.3f}, win={win_rate:.2%}, n={n})"
        rules.append(
            FilterRule(
                id=rid,
                bot=bot,
                kind=kind,
                when=when,
                params={},
                rationale=rationale,
                support={"n": n, "avg_r": avg_r, "win_rate": win_rate},
            )
        )

    # 2) Feature threshold proposal: rel_volume floor when losers cluster at low relvol.
    # Requires we log rel_volume into entry meta (we do in Phase 7+ patches).
    by_bot: Dict[str, List[Tuple[float, bool]]] = {}
    for t in trades:
        bot = str(t.bot or "UNKNOWN").upper()
        sym = str(t.symbol or "").upper()
        meta = meta_by_sym.get(sym) or {}
        feats = meta.get("entry_features") if isinstance(meta.get("entry_features"), dict) else {}
        rv = _safe_float(feats.get("rel_volume"), 0.0)
        by_bot.setdefault(bot, []).append((rv, _wins(t)))

    for bot, rows in by_bot.items():
        if len(rows) < max(min_trades_per_slice, 40):
            continue
        losers = [rv for rv, win in rows if not win]
        winners = [rv for rv, win in rows if win]
        if len(losers) < 15 or len(winners) < 15:
            continue

        # If loser relvol is meaningfully lower than winner relvol, propose a floor.
        lo_med = sorted(losers)[len(losers) // 2]
        wi_med = sorted(winners)[len(winners) // 2]
        if wi_med <= 0:
            continue

        if lo_med < wi_med * 0.75:
            # propose floor at ~winner 25th percentile
            winners_sorted = sorted(winners)
            q25 = winners_sorted[max(0, int(0.25 * len(winners_sorted)) - 1)]
            floor = max(0.5, min(3.0, float(q25)))
            rid = f"min_relvol_{bot.lower()}"
            rationale = f"{bot} losers cluster at low relvol (loser_med={lo_med:.2f}, winner_med={wi_med:.2f})"
            rules.append(
                FilterRule(
                    id=rid,
                    bot=bot,
                    kind="MIN_FEATURE",
                    when={},
                    params={"feature": "rel_volume", "min": floor},
                    rationale=rationale,
                    support={"loser_med": lo_med, "winner_med": wi_med, "min": floor, "n": len(rows)},
                )
            )

    pack = {
        "version": "researcher_v1",
        "trades": len(trades),
        "rules": [asdict(r) for r in rules],
    }
    return pack


def write_rules(pack: Dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True, help="Path to dt_trades.jsonl")
    ap.add_argument("--out", required=True, help="Output rules json")
    ap.add_argument("--min_trades", type=int, default=25)
    args = ap.parse_args()

    pack = propose_filters(dt_trades_path=Path(args.trades), min_trades_per_slice=int(args.min_trades))
    write_rules(pack, Path(args.out))
    print(f"wrote {args.out} rules={len(pack.get('rules') or [])} trades={pack.get('trades')}")


if __name__ == "__main__":
    main()
