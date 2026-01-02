# dt_backend/historical_replay/replay_metrics_dt.py
"""Parse Phase-6 replay artifacts and compute metrics + promotion gates.

Consumes dt_trades.jsonl produced by step_replay_engine_dt and returns:
  - overall PnL, win rate, expectancy, drawdown, hit rate, avg R, trades/day
  - breakdown by bot / regime / micro time window
  - promotion gate result with reasons

This is deliberately *audit-friendly*: it relies on the append-only event log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


@dataclass
class TradeRecord:
    symbol: str
    bot: str
    regime: str
    day_type: str
    micro: str
    entry_ts: str
    exit_ts: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    stop_price: float
    pnl: float
    r_multiple: float
    exit_reason: str
    entry_confidence: float = 0.0


def _calc_r_multiple(side: str, entry: float, exit_px: float, stop_px: float) -> float:
    # Risk per share is distance to stop.
    risk = abs(entry - stop_px)
    if risk <= 0:
        return 0.0
    pnl_per_share = (exit_px - entry) if side == "BUY" else (entry - exit_px)
    return pnl_per_share / risk


def trades_from_events(events: List[Dict[str, Any]]) -> List[TradeRecord]:
    """Reconstruct trade records from dt_trades.jsonl events.

    Assumptions:
    - bracket_set represents an entry (BUY or short entry SELL).
    - exit_signal closes remaining qty.
    - partial_signal realizes partial qty (treated as its own mini-exit, but we also track remaining qty).

    We compute a single TradeRecord per symbol per entry by using the *final* exit_price as the close.
    Partial exits are included into pnl, but the final exit_price is last exit price.
    """

    open_pos: Dict[str, Dict[str, Any]] = {}
    closed: List[TradeRecord] = []

    # We process in file order. If timestamps exist, file order should still be chronological.
    for e in events:
        t = str(e.get("type") or "")
        sym = str(e.get("symbol") or "").upper()
        if not sym:
            continue

        if t == "bracket_set":
            side = str(e.get("side") or "").upper() or "BUY"
            qty = _safe_float(e.get("qty"), 0.0)
            entry_px = _safe_float(e.get("entry"), _safe_float(e.get("entry_price"), 0.0))
            stop_px = _safe_float(e.get("stop"), _safe_float(e.get("stop_price"), 0.0))

            bot = str(e.get("bot") or e.get("meta", {}).get("bot") or "")
            meta = e.get("meta") if isinstance(e.get("meta"), dict) else {}
            regime = str(meta.get("regime") or e.get("regime") or "")
            day_type = str(meta.get("day_type") or e.get("day_type") or "")
            micro = str(meta.get("micro") or e.get("micro") or "")

            open_pos[sym] = {
                "side": side,
                "qty": qty,
                "rem_qty": qty,
                "entry": entry_px,
                "stop": stop_px,
                "pnl": 0.0,
                "bot": bot,
                "regime": regime,
                "day_type": day_type,
                "micro": micro,
                "entry_ts": str(e.get("ts") or ""),
                "last_exit_ts": "",
                "last_exit_px": 0.0,
                "exit_reason": "",
                "confidence": _safe_float(e.get("confidence"), 0.0),
            }
            continue

        if t == "partial_signal":
            pos = open_pos.get(sym)
            if not pos:
                continue
            qty = min(_safe_float(e.get("qty"), 0.0), pos.get("rem_qty", 0.0))
            if qty <= 0:
                continue
            exit_px = _safe_float(e.get("last"), 0.0)
            entry = _safe_float(pos.get("entry"), 0.0)
            side = str(pos.get("side") or "BUY")
            pnl = (exit_px - entry) * qty if side == "BUY" else (entry - exit_px) * qty
            pos["pnl"] = _safe_float(pos.get("pnl"), 0.0) + pnl
            pos["rem_qty"] = _safe_float(pos.get("rem_qty"), 0.0) - qty
            pos["last_exit_ts"] = str(e.get("ts") or "")
            pos["last_exit_px"] = exit_px
            pos["exit_reason"] = str(e.get("reason") or pos.get("exit_reason") or "partial")
            continue

        if t == "exit_signal":
            pos = open_pos.get(sym)
            if not pos:
                continue
            qty = _safe_float(pos.get("rem_qty"), 0.0)
            exit_px = _safe_float(e.get("last"), 0.0)
            entry = _safe_float(pos.get("entry"), 0.0)
            side = str(pos.get("side") or "BUY")
            pnl = (exit_px - entry) * qty if side == "BUY" else (entry - exit_px) * qty
            total_pnl = _safe_float(pos.get("pnl"), 0.0) + pnl

            stop_px = _safe_float(pos.get("stop"), 0.0)
            r = _calc_r_multiple(side=side, entry=entry, exit_px=exit_px, stop_px=stop_px)
            # Scale R by fraction of full size actually realized (partials + final)
            # (If rem_qty shrank due to partials, total_pnl already accounts.)
            full_r = 0.0
            if abs(entry - stop_px) > 0:
                full_r = total_pnl / (abs(entry - stop_px) * max(1e-9, _safe_float(pos.get("qty"), 0.0)))

            closed.append(
                TradeRecord(
                    symbol=sym,
                    bot=str(pos.get("bot") or ""),
                    regime=str(pos.get("regime") or ""),
                    day_type=str(pos.get("day_type") or ""),
                    micro=str(pos.get("micro") or ""),
                    entry_ts=str(pos.get("entry_ts") or ""),
                    exit_ts=str(e.get("ts") or pos.get("last_exit_ts") or ""),
                    side=side,
                    qty=_safe_float(pos.get("qty"), 0.0),
                    entry_price=entry,
                    exit_price=exit_px,
                    stop_price=stop_px,
                    pnl=total_pnl,
                    r_multiple=full_r,
                    exit_reason=str(e.get("reason") or pos.get("exit_reason") or ""),
                    entry_confidence=_safe_float(pos.get("confidence"), 0.0),
                )
            )

            open_pos.pop(sym, None)
            continue

    return closed


def _equity_curve(trades: List[TradeRecord]) -> List[float]:
    eq = 0.0
    curve = []
    for t in trades:
        eq += _safe_float(t.pnl, 0.0)
        curve.append(eq)
    return curve


def _max_drawdown(curve: List[float]) -> float:
    peak = float("-inf")
    max_dd = 0.0
    for x in curve:
        peak = max(peak, x)
        dd = peak - x
        max_dd = max(max_dd, dd)
    return max_dd


def _agg_stats(trades: List[TradeRecord]) -> Dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {
            "trades": 0,
            "pnl": 0.0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "expectancy": 0.0,
        }
    pnl = sum(t.pnl for t in trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    avg_r = sum(t.r_multiple for t in trades) / n
    return {
        "trades": n,
        "pnl": float(pnl),
        "win_rate": float(wins / n),
        "avg_r": float(avg_r),
        "expectancy": float(avg_r),
    }


def compute_replay_metrics(trades: List[TradeRecord]) -> Dict[str, Any]:
    curve = _equity_curve(trades)
    out: Dict[str, Any] = {}
    out.update(_agg_stats(trades))
    out["drawdown"] = float(_max_drawdown(curve)) if curve else 0.0
    out["equity_end"] = float(curve[-1]) if curve else 0.0

    # hit_rate here = % profitable trades
    out["hit_rate"] = float(out.get("win_rate", 0.0))

    # breakouts
    by_bot: Dict[str, List[TradeRecord]] = {}
    by_reg: Dict[str, List[TradeRecord]] = {}
    by_micro: Dict[str, List[TradeRecord]] = {}

    for t in trades:
        by_bot.setdefault(t.bot or "UNKNOWN", []).append(t)
        by_reg.setdefault(t.regime or "UNKNOWN", []).append(t)
        by_micro.setdefault(t.micro or "UNKNOWN", []).append(t)

    out["by_bot"] = {k: _agg_stats(v) for k, v in sorted(by_bot.items(), key=lambda kv: kv[0])}
    out["by_regime"] = {k: _agg_stats(v) for k, v in sorted(by_reg.items(), key=lambda kv: kv[0])}
    out["by_micro"] = {k: _agg_stats(v) for k, v in sorted(by_micro.items(), key=lambda kv: kv[0])}

    return out


def promotion_gate(
    metrics: Dict[str, Any],
    *,
    min_trades: int = 20,
    min_expectancy: float = 0.05,
    max_drawdown: float = 5_000.0,
    min_slice_trades: int = 8,
) -> Dict[str, Any]:
    """Simple, conservative promotion gate.

    Rules (tunable):
      - overall trade count >= min_trades
      - overall expectancy (avg_r) >= min_expectancy
      - drawdown <= max_drawdown (in $ terms from replay sizing)
      - for each bot slice with >= min_slice_trades, expectancy > 0

    This is a *gate*, not the truth — it prevents obviously-bad versions from going live.
    """

    reasons: List[str] = []
    trades = int(metrics.get("trades", 0) or 0)
    exp = _safe_float(metrics.get("expectancy"), 0.0)
    dd = _safe_float(metrics.get("drawdown"), 0.0)

    if trades < min_trades:
        reasons.append(f"too_few_trades({trades}<{min_trades})")
    if exp < min_expectancy:
        reasons.append(f"low_expectancy({exp:.3f}<{min_expectancy})")
    if dd > max_drawdown:
        reasons.append(f"drawdown_too_high({dd:.2f}>{max_drawdown})")

    by_bot = metrics.get("by_bot") if isinstance(metrics.get("by_bot"), dict) else {}
    for bot, stat in by_bot.items():
        if not isinstance(stat, dict):
            continue
        n = int(stat.get("trades", 0) or 0)
        if n < min_slice_trades:
            continue
        bexp = _safe_float(stat.get("expectancy"), 0.0)
        if bexp <= 0:
            reasons.append(f"bot_negative({bot}:{bexp:.3f})")

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "thresholds": {
            "min_trades": min_trades,
            "min_expectancy": min_expectancy,
            "max_drawdown": max_drawdown,
            "min_slice_trades": min_slice_trades,
        },
    }


def metrics_from_dt_trades_file(dt_trades_path: Path) -> Dict[str, Any]:
    events = _read_jsonl(dt_trades_path)
    trades = trades_from_events(events)
    metrics = compute_replay_metrics(trades)
    gate = promotion_gate(metrics)
    return {
        "trades": [asdict(t) for t in trades],
        "metrics": metrics,
        "gate": gate,
        "source": str(dt_trades_path),
    }
