"""dt_backend/historical_replay/backtest_runner_dt.py — Phase 6

Day-by-day intraday replay/backtest harness for dt_backend.

What this is
------------
This runner replays historical intraday bars *through the same policy + execution
stack* used in live mode (strategies -> policy -> execution -> synthetic exits).

Key properties
--------------
* Deterministic artifacts (uses DT_TRUTH_DIR + DT_ROLLING_PATH overrides)
* Deterministic time (sets DT_NOW_UTC per bar)
* Produces measurable metrics + breakdowns (bot/regime/time-window)
* Checkpointed day-by-day execution

Expected inputs
---------------
Raw day files in:  <ml_data_dt>/intraday/replay/raw_days
Each file should be JSON and can be in one of these shapes:

  A) {"date": "YYYY-MM-DD", "symbols": {"AAPL": {"bars": [...]}, ...}}
  B) {"AAPL": [...bars...], "MSFT": [...bars...]}
  C) [{"symbol":"AAPL", "bars": [...]}, ...]

Bars are dicts with at least:
    ts/t: ISO8601 timestamp, and o,h,l,c,v

Usage
-----
python -m dt_backend.historical_replay.backtest_runner_dt \
  --start 2025-11-01 --end 2025-11-28 --tag phase6_v1 --tf 5Min

"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dt_backend.core import DT_PATHS
from dt_backend.core.logger_dt import log
from dt_backend.core.time_override_dt import parse_utc, utc_iso
from dt_backend.core.data_pipeline_dt import save_rolling, ensure_symbol_node

from dt_backend.core.context_state_dt import build_intraday_context
from dt_backend.engines.feature_engineering import build_intraday_features
from dt_backend.core.regime_detector_dt import classify_intraday_regime
from dt_backend.core.meta_controller_dt import ensure_daily_plan
from dt_backend.core.policy_engine_dt import apply_intraday_policy
from dt_backend.core.execution_dt import run_execution_intraday
from dt_backend.engines.trade_executor import execute_from_policy, ExecutionConfig


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _ml_data_root() -> Path:
    root = DT_PATHS.get("dtml_data")
    return root if isinstance(root, Path) else Path("ml_data_dt")


def _raw_days_dir() -> Path:
    return _ml_data_root() / "intraday" / "replay" / "raw_days"


def _runs_dir() -> Path:
    return _ml_data_root() / "intraday" / "replay" / "runs"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _iter_day_files(raw_dir: Path, start: str, end: str) -> List[Path]:
    start_d = datetime.fromisoformat(start).date()
    end_d = datetime.fromisoformat(end).date()
    out: List[Path] = []
    if not raw_dir.exists():
        return out
    for p in sorted(raw_dir.glob("*.json")):
        name = p.stem
        # accept either YYYY-MM-DD.json or raw_YYYY-MM-DD.json etc
        for token in name.split("_"):
            try:
                d = datetime.fromisoformat(token).date()
                if start_d <= d <= end_d:
                    out.append(p)
                break
            except Exception:
                continue
    return sorted(out)


def _extract_symbol_bars(obj: Any) -> Tuple[Optional[str], Dict[str, List[Dict[str, Any]]]]:
    """Return (date_str, {sym: bars})."""
    date_str: Optional[str] = None
    syms: Dict[str, List[Dict[str, Any]]] = {}

    if isinstance(obj, dict):
        # A) {date, symbols}
        if "date" in obj and isinstance(obj.get("date"), str):
            date_str = obj.get("date")
        if isinstance(obj.get("symbols"), dict):
            for sym, node in obj["symbols"].items():
                if not isinstance(sym, str):
                    continue
                if isinstance(node, dict):
                    bars = node.get("bars") or node.get("bars_intraday") or node.get("bars_1m") or []
                else:
                    bars = node
                if isinstance(bars, list):
                    syms[sym.upper()] = [b for b in bars if isinstance(b, dict)]
            return date_str, syms

        # B) {AAPL: [...bars...]}
        for sym, bars in obj.items():
            if not isinstance(sym, str):
                continue
            if sym.lower() in {"date", "symbols", "meta"}:
                continue
            if isinstance(bars, list):
                syms[sym.upper()] = [b for b in bars if isinstance(b, dict)]
        return date_str, syms

    # C) [{symbol, bars}]
    if isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict):
                continue
            sym = item.get("symbol")
            if not isinstance(sym, str):
                continue
            bars = item.get("bars") or item.get("bars_intraday") or []
            if isinstance(bars, list):
                syms[sym.upper()] = [b for b in bars if isinstance(b, dict)]
        return date_str, syms

    return None, {}


def _bar_ts(b: Dict[str, Any]) -> Optional[datetime]:
    return parse_utc(b.get("ts") or b.get("t") or b.get("timestamp"))


def _normalize_bars(bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for b in bars:
        if not isinstance(b, dict):
            continue
        ts = b.get("ts") or b.get("t") or b.get("timestamp")
        if ts is None:
            continue
        # Ensure canonical key "ts" exists
        bb = dict(b)
        bb["ts"] = str(ts)
        out.append(bb)
    out.sort(key=lambda x: str(x.get("ts")))
    return out


def _aggregate_5m(bars_1m: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Best-effort 5-minute aggregation from 1-minute bars."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for b in bars_1m:
        dt = _bar_ts(b)
        if dt is None:
            continue
        # Floor to 5 minute
        floored = dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)
        key = floored.isoformat().replace("+00:00", "Z")
        buckets.setdefault(key, []).append(b)

    out: List[Dict[str, Any]] = []
    for key in sorted(buckets.keys()):
        chunk = buckets[key]
        if not chunk:
            continue
        # Use first/last for o/c, min/max for l/h, sum v
        try:
            o = float(chunk[0].get("o"))
            c = float(chunk[-1].get("c"))
            h = max(float(x.get("h")) for x in chunk)
            l = min(float(x.get("l")) for x in chunk)
        except Exception:
            continue
        v = 0.0
        for x in chunk:
            try:
                v += float(x.get("v") or x.get("volume") or 0.0)
            except Exception:
                pass
        out.append({"ts": key, "o": o, "h": h, "l": l, "c": c, "v": v})
    return out


# ---------------------------------------------------------------------------
# Metrics (Phase 6)
# ---------------------------------------------------------------------------


@dataclass
class CloseEvent:
    ts: datetime
    symbol: str
    side: str
    qty: float
    price: float
    realized_pnl: float


def _read_ledger(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
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


def _close_events_for_day(ledger: Dict[str, Any], day: str) -> List[CloseEvent]:
    out: List[CloseEvent] = []
    fills = ledger.get("fills")
    if not isinstance(fills, list):
        return out
    for f in fills:
        if not isinstance(f, dict):
            continue
        ts = parse_utc(f.get("ts"))
        if ts is None:
            continue
        if ts.date().isoformat() != day:
            continue
        try:
            realized = float(f.get("realized_pnl") or 0.0)
            side = str(f.get("side") or "").upper()
            qty = float(f.get("qty") or 0.0)
            price = float(f.get("price") or 0.0)
            sym = str(f.get("symbol") or "").upper()
        except Exception:
            continue
        # Only count closing actions (SELL for longs, BUY for shorts)
        if realized == 0.0:
            # partial close can still be zero; keep it if it has qty
            pass
        if not sym or qty <= 0 or price <= 0 or not side:
            continue
        out.append(CloseEvent(ts=ts, symbol=sym, side=side, qty=qty, price=price, realized_pnl=realized))
    out.sort(key=lambda x: x.ts)
    return out


def _entry_events(trades: List[Dict[str, Any]], day: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in trades:
        if str(e.get("type") or "") != "bracket_set":
            continue
        ts = parse_utc(e.get("ts"))
        if ts is None or ts.date().isoformat() != day:
            continue
        out.append(e)
    out.sort(key=lambda x: str(x.get("ts") or ""))
    return out


def _compute_day_metrics(
    *,
    day: str,
    ledger: Dict[str, Any],
    trades: List[Dict[str, Any]],
    start_equity: float,
) -> Dict[str, Any]:
    closes = _close_events_for_day(ledger, day)
    entries = _entry_events(trades, day)

    # Build per-symbol queue of open entry risk blocks.
    open_q: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        sym = str(e.get("symbol") or "").upper()
        qty = float(e.get("qty") or 0.0)
        entry = float(e.get("entry_price") or 0.0)
        stop = e.get("stop")
        try:
            stop_f = float(stop) if stop is not None else 0.0
        except Exception:
            stop_f = 0.0
        risk_per_share = abs(entry - stop_f) if (entry > 0 and stop_f > 0) else 0.0
        risk_total = risk_per_share * qty
        meta = e.get("meta") if isinstance(e.get("meta"), dict) else {}
        open_q.setdefault(sym, []).append(
            {
                "remaining_qty": qty,
                "risk_total": risk_total,
                "risk_per_share": risk_per_share,
                "bot": str(e.get("bot") or "UNKNOWN"),
                "regime": str(meta.get("regime_label") or "UNKNOWN"),
                "micro": str(meta.get("micro_label") or "UNKNOWN"),
                "time_window": str(meta.get("time_window") or "UNKNOWN"),
            }
        )

    realized = 0.0
    wins = 0
    closes_n = 0
    r_list: List[float] = []

    by_bot: Dict[str, Dict[str, float]] = {}
    by_regime: Dict[str, Dict[str, float]] = {}
    by_window: Dict[str, Dict[str, float]] = {}

    def _acc(bucket: Dict[str, Dict[str, float]], key: str, pnl: float, r: float) -> None:
        b = bucket.get(key) or {"pnl": 0.0, "r": 0.0, "n": 0.0}
        b["pnl"] = float(b.get("pnl") or 0.0) + pnl
        b["r"] = float(b.get("r") or 0.0) + r
        b["n"] = float(b.get("n") or 0.0) + 1.0
        bucket[key] = b

    for c in closes:
        closes_n += 1
        realized += float(c.realized_pnl)
        if c.realized_pnl > 0:
            wins += 1

        # Attribute to latest open entry for this symbol.
        sym_q = open_q.get(c.symbol) or []
        bot = "UNKNOWN"
        regime = "UNKNOWN"
        tw = "UNKNOWN"
        r_mult = 0.0

        if sym_q:
            ent = sym_q[0]
            bot = str(ent.get("bot") or "UNKNOWN")
            regime = str(ent.get("regime") or "UNKNOWN")
            tw = str(ent.get("time_window") or "UNKNOWN")

            # Pro-rate risk for partial closes.
            rq = float(ent.get("remaining_qty") or 0.0)
            rt = float(ent.get("risk_total") or 0.0)
            if rq > 0 and rt > 0:
                frac = min(1.0, max(0.0, float(c.qty) / rq))
                risk_part = rt * frac
                if risk_part > 0:
                    r_mult = float(c.realized_pnl) / risk_part
            # Decrement remaining qty and pop if done.
            ent["remaining_qty"] = max(0.0, rq - float(c.qty))
            # Reduce risk_total proportionally.
            ent["risk_total"] = max(0.0, rt - (rt * min(1.0, float(c.qty) / max(rq, 1e-9))))
            if float(ent.get("remaining_qty") or 0.0) <= 0.0:
                sym_q.pop(0)
            open_q[c.symbol] = sym_q

        r_list.append(r_mult)
        _acc(by_bot, bot, float(c.realized_pnl), r_mult)
        _acc(by_regime, regime, float(c.realized_pnl), r_mult)
        _acc(by_window, tw, float(c.realized_pnl), r_mult)

    win_rate = (wins / closes_n) if closes_n > 0 else 0.0
    expectancy = (realized / closes_n) if closes_n > 0 else 0.0
    avg_r = (sum(r_list) / len(r_list)) if r_list else 0.0

    end_equity = start_equity + realized

    return {
        "day": day,
        "start_equity": float(start_equity),
        "end_equity": float(end_equity),
        "realized_pnl": float(realized),
        "closes": int(closes_n),
        "win_rate": float(win_rate),
        "expectancy": float(expectancy),
        "avg_r": float(avg_r),
        "by_bot": by_bot,
        "by_regime": by_regime,
        "by_time_window": by_window,
    }


def _max_drawdown(equity_curve: List[float]) -> float:
    peak = float("-inf")
    dd = 0.0
    for x in equity_curve:
        peak = max(peak, x)
        dd = min(dd, x - peak)
    return float(dd)


# ---------------------------------------------------------------------------
# Replay loop
# ---------------------------------------------------------------------------


def _choose_axis(symbol_bars: Dict[str, List[Dict[str, Any]]]) -> str:
    for p in ["SPY", "QQQ"]:
        if p in symbol_bars and symbol_bars[p]:
            return p
    # fallback to longest
    best = ""
    best_n = -1
    for sym, bars in symbol_bars.items():
        if len(bars) > best_n:
            best = sym
            best_n = len(bars)
    return best


def _build_step_rolling(
    *,
    symbol_bars_1m: Dict[str, List[Dict[str, Any]]],
    pointers: Dict[str, int],
    current_ts: datetime,
    tf: str,
) -> Dict[str, Any]:
    rolling: Dict[str, Any] = {}
    for sym, bars in symbol_bars_1m.items():
        # Advance pointer to include bars <= current_ts
        i = pointers.get(sym, 0)
        while i < len(bars):
            bt = _bar_ts(bars[i])
            if bt is None or bt <= current_ts:
                i += 1
                continue
            break
        pointers[sym] = i
        slice_1m = bars[:i]
        node = ensure_symbol_node(rolling, sym)
        node["bars_intraday"] = slice_1m
        if tf.lower().startswith("5"):
            node["bars_intraday_5m"] = _aggregate_5m(slice_1m)
        rolling[sym] = node
    return rolling


def run_backtest(*, start: str, end: str, tag: str, tf: str) -> Path:
    raw_dir = _raw_days_dir()
    day_files = _iter_day_files(raw_dir, start, end)
    if not day_files:
        raise SystemExit(f"No raw day files found in {raw_dir} for range {start}..{end}")

    run_root = _runs_dir() / tag
    run_root.mkdir(parents=True, exist_ok=True)

    # Hermetic run overrides
    os.environ["DT_TRUTH_DIR"] = str(run_root)
    os.environ["DT_ROLLING_PATH"] = str(run_root / "intraday" / "rolling_intraday.json")
    os.environ["DT_BOT_ID"] = f"replay_{tag}"
    os.environ["DT_BOT_LEDGER_PATH"] = str(run_root / "intraday" / "brokers" / f"bot_replay_{tag}.json")

    # Backtests should not depend on ML models unless explicitly enabled.
    os.environ.setdefault("DT_ALLOW_MODEL_FALLBACK", "0")
    os.environ["DT_FEATURE_TF"] = tf
    os.environ.setdefault("DT_FEATURES_MIN_INTERVAL", "0")

    # Safety: keep it conservative in replay until proven.
    os.environ.setdefault("DT_MAX_ORDERS_PER_CYCLE", "3")

    equity = float(os.getenv("DT_BACKTEST_START_EQUITY", "100000") or 100000.0)
    equity_curve: List[float] = []
    day_results: List[Dict[str, Any]] = []

    exec_cfg = ExecutionConfig(
        dry_run=False,
        max_orders_per_cycle=int(float(os.getenv("DT_MAX_ORDERS_PER_CYCLE", "3") or 3)),
        allow_shorts=str(os.getenv("DT_ALLOW_SHORTS", "0") or "0").strip().lower() in {"1","true","yes","y"},
        min_confidence=float(os.getenv("DT_EXEC_MIN_CONF", "0.25") or 0.25),
        default_qty=float(os.getenv("DT_EXEC_DEFAULT_QTY", "1") or 1.0),
    )

    for p in day_files:
        obj = _load_json(p)
        date_hint, symbol_bars = _extract_symbol_bars(obj)
        if not symbol_bars:
            log(f"[backtest] ⚠️ skipping {p.name}: no symbols")
            continue

        # Determine day string.
        day = date_hint
        if not day:
            # Try parse from filename tokens
            for token in p.stem.split("_"):
                try:
                    datetime.fromisoformat(token).date()
                    day = token
                    break
                except Exception:
                    continue
        if not day:
            log(f"[backtest] ⚠️ skipping {p.name}: could not determine date")
            continue

        # Normalize bars
        symbol_bars_1m = {sym: _normalize_bars(bars) for sym, bars in symbol_bars.items()}
        axis_sym = _choose_axis(symbol_bars_1m)
        axis = symbol_bars_1m.get(axis_sym) or []
        if not axis:
            log(f"[backtest] ⚠️ skipping {day}: empty axis symbol")
            continue

        log(f"[backtest] ▶️ {day} axis={axis_sym} symbols={len(symbol_bars_1m)} bars={len(axis)}")

        pointers = {sym: 0 for sym in symbol_bars_1m.keys()}

        # One plan per day.
        plan_forced = False

        for b in axis:
            cur = _bar_ts(b)
            if cur is None:
                continue

            os.environ["DT_NOW_UTC"] = utc_iso(cur)

            rolling = _build_step_rolling(symbol_bars_1m=symbol_bars_1m, pointers=pointers, current_ts=cur, tf=tf)
            save_rolling(rolling)

            # Build state for this moment.
            build_intraday_context(target_date=day, now_utc=cur)
            build_intraday_features(now_utc=cur, ignore_min_interval=True)
            classify_intraday_regime(now_utc=cur)
            if not plan_forced:
                ensure_daily_plan(force=True, date_override=day)
                plan_forced = True

            apply_intraday_policy(max_positions=int(float(os.getenv("DT_MAX_POSITIONS", "3") or 3)))
            run_execution_intraday(now_utc=cur)
            execute_from_policy(exec_cfg, now_utc=cur)

        # Force an EOD flatten tick at 15:59 NY (roughly).
        try:
            close_tick = parse_utc(f"{day}T20:59:00Z") or (parse_utc(f"{day}T20:59:00+00:00"))
        except Exception:
            close_tick = None
        if close_tick is not None:
            os.environ["DT_NOW_UTC"] = utc_iso(close_tick)
            execute_from_policy(exec_cfg, now_utc=close_tick)

        # Compute day metrics from artifacts.
        ledger_path = Path(os.environ["DT_BOT_LEDGER_PATH"])
        trades_path = Path(os.environ["DT_TRUTH_DIR"]) / "intraday" / "dt_trades.jsonl"
        ledger = _read_ledger(ledger_path)
        trades = _read_jsonl(trades_path)
        day_m = _compute_day_metrics(day=day, ledger=ledger, trades=trades, start_equity=equity)
        equity = float(day_m.get("end_equity") or equity)
        equity_curve.append(equity)
        day_results.append(day_m)

        # Persist checkpoint
        (run_root / "checkpoints").mkdir(parents=True, exist_ok=True)
        ck = run_root / "checkpoints" / f"{day}.json"
        ck.write_text(json.dumps(day_m, ensure_ascii=False, indent=2), encoding="utf-8")

    # Summarize
    total_pnl = (equity_curve[-1] - equity_curve[0]) if len(equity_curve) >= 2 else (equity_curve[-1] - float(os.getenv("DT_BACKTEST_START_EQUITY", "100000") or 100000.0) if equity_curve else 0.0)
    mdd = _max_drawdown(equity_curve) if equity_curve else 0.0
    closes_total = sum(int(d.get("closes") or 0) for d in day_results)
    wins_total = 0
    for d in day_results:
        # approximate wins from win_rate
        try:
            wins_total += int(round(float(d.get("win_rate") or 0.0) * int(d.get("closes") or 0)))
        except Exception:
            pass
    win_rate_total = (wins_total / closes_total) if closes_total > 0 else 0.0
    expectancy_total = (total_pnl / closes_total) if closes_total > 0 else 0.0

    summary = {
        "tag": tag,
        "start": start,
        "end": end,
        "tf": tf,
        "days": len(day_results),
        "start_equity": float(os.getenv("DT_BACKTEST_START_EQUITY", "100000") or 100000.0),
        "end_equity": float(equity_curve[-1]) if equity_curve else float(os.getenv("DT_BACKTEST_START_EQUITY", "100000") or 100000.0),
        "total_pnl": float(total_pnl),
        "closes": int(closes_total),
        "win_rate": float(win_rate_total),
        "expectancy": float(expectancy_total),
        "max_drawdown": float(mdd),
        "updated_at": utc_iso(),
    }

    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--tag", required=True, help="run tag/name")
    ap.add_argument("--tf", default="5Min", choices=["1Min", "5Min"], help="feature timeframe")
    args = ap.parse_args()

    run_dir = run_backtest(start=args.start, end=args.end, tag=args.tag, tf=args.tf)
    log(f"[backtest] ✅ completed. run_dir={run_dir}")


if __name__ == "__main__":
    main()
