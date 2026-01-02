# dt_backend/historical_replay/historical_replay_engine.py
"""
Full intraday replay engine:
    raw bars → context_dt → features_dt → predictions_dt → policy_dt → execution_dt
and then computes:
    - per-symbol PnL
    - gross PnL
    - hit rate
    - trades count
    - daily replay summary

Writes results to:
    ml_data_dt/intraday/replay/replay_results/<date>.json

UPDATED:
    • Support .json.gz compressed raw_day files
    • Auto-detect gzip vs plain JSON
"""

from __future__ import annotations

import json
import gzip
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dt_backend.core.config_dt import DT_PATHS
from dt_backend.core.data_pipeline_dt import (
    _read_rolling,
    save_rolling,
    ensure_symbol_node,
    log,
)

# Phase 2 engines
from dt_backend.core.context_state_dt import build_intraday_context
from dt_backend.core.regime_detector_dt import classify_intraday_regime
from dt_backend.core.meta_controller_dt import ensure_daily_plan
from dt_backend.engines.feature_engineering import build_intraday_features

# Phase 1 model scoring (classification)
from dt_backend.ml.ai_model_intraday import score_intraday_batch, load_intraday_models

# Phase 3 logic
from dt_backend.core.policy_engine_dt import apply_intraday_policy
from dt_backend.core.execution_dt import run_execution_intraday


# ---------------------------------------------------------
# Result struct
# ---------------------------------------------------------

@dataclass
class ReplayResult:
    date: str
    n_symbols: int
    n_trades: int
    gross_pnl: float
    avg_pnl_per_trade: float
    hit_rate: float
    meta: Dict[str, Any]


# ---------------------------------------------------------
# Path helpers
# ---------------------------------------------------------

def _paths_for_date(date_str: str) -> Tuple[Path, Path]:
    """
    Resolve:
      • raw day file (prefers .json.gz if present, else .json)
      • replay result output path (.json, uncompressed)
    """
    root = Path(DT_PATHS.get("dtml_data", Path("ml_data_dt")))
    raw_gz = root / "intraday" / "replay" / "raw_days" / f"{date_str}.json.gz"
    raw_json = root / "intraday" / "replay" / "raw_days" / f"{date_str}.json"
    result_file = root / "intraday" / "replay" / "replay_results" / f"{date_str}.json"

    if raw_gz.exists():
        raw_path = raw_gz
    else:
        raw_path = raw_json

    return raw_path, result_file


# ---------------------------------------------------------
# Load raw intraday bars (gzip OR normal)
# ---------------------------------------------------------

def _load_raw_day(date_str: str) -> List[Dict[str, Any]]:
    raw_path, _ = _paths_for_date(date_str)

    if not raw_path.exists():
        log(f"[replay_engine] ⚠️ missing raw day file: {raw_path}")
        return []

    try:
        # Auto-detect gzip by extension
        if raw_path.suffix == ".gz":
            with gzip.open(raw_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with raw_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

        if not isinstance(data, list):
            log(f"[replay_engine] ⚠️ malformed raw file (not a list): {raw_path}")
            return []

        return data

    except Exception as e:
        log(f"[replay_engine] ❌ load failure {raw_path}: {e}")
        return []


# ---------------------------------------------------------
# Rolling bootstrap
# ---------------------------------------------------------

def _inject_bars(raw_day: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Take the raw list:
        [{"symbol": "AAPL", "bars": [...]}, ...]
    and build a fresh rolling dict with bars_intraday per symbol.
    """
    rolling: Dict[str, Any] = {}

    for entry in raw_day:
        sym = str(entry.get("symbol") or "").upper()
        if not sym:
            continue

        bars = entry.get("bars") or []
        node = ensure_symbol_node(rolling, sym)
        node["bars_intraday"] = bars
        rolling[sym] = node

    return rolling


def _extract_prices(bars: List[Dict[str, Any]]) -> List[float]:
    """
    Extract close/last prices from Alpaca-style minute bars.

    We accept:
      • "c"     → close
      • "close" → close
      • "price" → fallback
    """
    out: List[float] = []
    for b in bars:
        if not isinstance(b, dict):
            continue
        try:
            p = b.get("price")
            if p is None:
                p = b.get("c", b.get("close"))
            if p is None:
                continue
            out.append(float(p))
        except Exception:
            continue
    return out


def _parse_any_ts(ts_raw: Any) -> datetime | None:
    """Best-effort parse of bar timestamps into timezone-aware UTC datetime."""
    if ts_raw is None:
        return None
    try:
        # epoch seconds
        if isinstance(ts_raw, (int, float)):
            v = float(ts_raw)
            # handle nanoseconds
            if v > 1e12:
                v = v / 1e9
            elif v > 1e10:
                v = v / 1e6
            return datetime.fromtimestamp(v, tz=timezone.utc)

        s = str(ts_raw).strip()
        if s.isdigit():
            return _parse_any_ts(float(s))
        # ISO string
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _infer_now_utc(raw_day: List[Dict[str, Any]], date_str: str) -> datetime:
    """Pick a representative UTC timestamp for replay day for time-aware engines."""
    # try first symbol's last bar
    try:
        if raw_day and isinstance(raw_day[0], dict):
            bars = raw_day[0].get('bars')
            if isinstance(bars, list) and bars:
                ts = _parse_any_ts((bars[-1] or {}).get('ts') or (bars[-1] or {}).get('t') or (bars[-1] or {}).get('timestamp'))
                if ts:
                    return ts
    except Exception:
        pass

    # fallback: noon NY ~= 17:00 UTC (rough), but we stay simple
    try:
        base = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    except Exception:
        base = datetime.now(timezone.utc)
    return base.replace(hour=17, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------
# PnL computation
# ---------------------------------------------------------

def _symbol_pnl(node: Dict[str, Any]) -> Tuple[float, int, int]:
    """
    Simple 1-shot PnL for the session:

        BUY  → (end/start - 1)
        SELL → (start/end - 1)

    Multiplied by execution_dt["size"].

    Returns:
        (pnl, trades_count, hits_count)
    """
    bars = node.get("bars_intraday") or []
    if len(bars) < 2:
        return 0.0, 0, 0

    prices = _extract_prices(bars)
    if len(prices) < 2:
        return 0.0, 0, 0

    start, end = prices[0], prices[-1]
    exec_dt = node.get("execution_dt") or {}

    side = str(exec_dt.get("side") or "").upper()
    size = float(exec_dt.get("size") or 0.0)

    if side == "BUY":
        ret = (end / start - 1.0) if start > 0 else 0.0
    elif side == "SELL":
        ret = (start / end - 1.0) if end > 0 else 0.0
    else:
        ret = 0.0

    pnl = size * ret
    trades = 1 if side in {"BUY", "SELL"} else 0
    hits = 1 if pnl > 0 else 0

    return pnl, trades, hits


# ---------------------------------------------------------
# Full replay for one day
# ---------------------------------------------------------

def replay_intraday_day(date_str: str) -> ReplayResult | None:
    """
    End-to-end replay for a single trading day:
      1) Load bars from raw_days/
      2) Build context_dt + features_dt
      3) Score with intraday models (classification)
      4) Apply policy_dt + execution_dt
      5) Aggregate PnL + trades + hit rate
      6) Write replay_results/<date>.json
    """
    raw_day = _load_raw_day(date_str)
    if not raw_day:
        return None

    # 1) bars → rolling
    rolling = _inject_bars(raw_day)
    save_rolling(rolling)

    # Infer a stable replay time for time-aware engines
    now_utc = _infer_now_utc(raw_day, date_str)

    # 2) context + features
    build_intraday_context(target_date=date_str, now_utc=now_utc)
    build_intraday_features(now_utc=now_utc)

    # 3) model scoring
    rolling = _read_rolling()
    rows: List[Dict[str, Any]] = []
    index: List[str] = []

    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue
        feats = node.get("features_dt") or {}
        if feats:
            rows.append(feats)
            index.append(sym)

    if not rows:
        log(f"[replay_engine] ⚠️ no features computed for {date_str}")
        return None

    import pandas as pd
    X = pd.DataFrame(rows, index=index)

    models = load_intraday_models()
    proba_df, labels = score_intraday_batch(X, models=models)

    for sym in proba_df.index:
        node = rolling.get(sym) or {}
        node["predictions_dt"] = {
            "label": str(labels.loc[sym]),
            "proba": proba_df.loc[sym].to_dict(),
        }
        rolling[sym] = node

    save_rolling(rolling)

    # 4) regime + meta-controller (Phase 2/2.5 + Phase 4)
    try:
        classify_intraday_regime(now_utc=now_utc)
    except Exception:
        pass
    try:
        ensure_daily_plan(force=True, date_override=date_str)
    except Exception:
        pass

    # 5) policy + execution
    apply_intraday_policy()
    run_execution_intraday(now_utc=now_utc)

    # 5) PnL aggregation
    rolling = _read_rolling()
    gross, trades, hits = 0.0, 0, 0

    for sym, node in rolling.items():
        if sym.startswith("_"):
            continue
        pnl, t, h = _symbol_pnl(node)
        gross += pnl
        trades += t
        hits += h

    avg_trade = gross / trades if trades > 0 else 0.0
    hit_rate = hits / trades if trades > 0 else 0.0

    result = ReplayResult(
        date=date_str,
        n_symbols=len([s for s in rolling if not s.startswith("_")]),
        n_trades=trades,
        gross_pnl=gross,
        avg_pnl_per_trade=avg_trade,
        hit_rate=hit_rate,
        meta={},
    )

    # Attach global context (regime/day plan) for downstream slice metrics
    try:
        g = (_read_rolling() or {}).get("_GLOBAL_DT") or {}
        if isinstance(g, dict):
            result.meta = {
                "regime_dt": g.get("regime_dt"),
                "regime": g.get("regime"),
                "micro_regime_dt": g.get("micro_regime_dt"),
                "daily_plan_dt": g.get("daily_plan_dt"),
            }
    except Exception:
        pass


    # 6) Save output
    _, out_path = _paths_for_date(date_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)

    log(
        f"[replay_engine] ✅ {date_str} → PnL={gross:.4f}, "
        f"trades={trades}, hit_rate={hit_rate:.2f}"
    )
    return result


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main() -> None:
    today = datetime.now(timezone.utc).date()
    replay_intraday_day(today.isoformat())


if __name__ == "__main__":
    main()
