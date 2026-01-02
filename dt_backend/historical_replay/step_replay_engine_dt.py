# dt_backend/historical_replay/step_replay_engine_dt.py
"""Intraday *step* replay engine (Phase 6).

This simulates the live DT loop over a historical day by advancing through
minute bars in fixed increments (e.g., every 5 minutes):

  raw bars → rolling snapshot → context/features → model predictions → policy → execution → trade_executor

Design goals
------------
- Replay-safe: never touches live artifacts when DT_TRUTH_DIR/DT_ROLLING_PATH env overrides are set.
- Deterministic enough for debugging: timestamps come from bar time (now_utc).
- Compatible with live semantics: uses the same pipeline + position manager.

Inputs
------
Expects raw day files produced by historical_replay_fetcher:
  <ml_data_dt>/intraday/replay/raw_days/YYYY-MM-DD.json(.gz)

Outputs
-------
Writes into the *override* directories configured by the caller (recommended):
  DT_TRUTH_DIR/<date>/dt_trades.jsonl
  DT_TRUTH_DIR/<date>/positions_dt.json
  DT_BOT_LEDGER_PATH (ledger file)
  DT_ROLLING_PATH (rolling cache)

The companion replay runner parses dt_trades.jsonl to compute metrics.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from dt_backend.core.config_dt import DT_PATHS
from dt_backend.core.data_pipeline_dt import ensure_symbol_node, save_rolling
from dt_backend.core.logger_dt import log

from dt_backend.core.meta_controller_dt import ensure_daily_plan
from dt_backend.core.context_state_dt import build_intraday_context
from dt_backend.core.regime_detector_dt import classify_intraday_regime
from dt_backend.engines.feature_engineering import build_intraday_features

from dt_backend.ml.ai_model_intraday import load_intraday_models, score_intraday_batch
from dt_backend.core.policy_engine_dt import apply_intraday_policy
from dt_backend.core.execution_dt import run_execution_intraday
from dt_backend.engines.trade_executor import execute_from_policy, ExecutionConfig


_ALLOWED = ("BUY", "HOLD", "SELL")


def _paths_for_date(date_str: str) -> Path:
    root = Path(DT_PATHS.get("dtml_data", Path("ml_data_dt")))
    raw_gz = root / "intraday" / "replay" / "raw_days" / f"{date_str}.json.gz"
    raw_json = root / "intraday" / "replay" / "raw_days" / f"{date_str}.json"
    if raw_gz.exists():
        return raw_gz
    return raw_json


def _load_raw_day(date_str: str) -> List[Dict[str, Any]]:
    path = _paths_for_date(date_str)
    if not path.exists():
        log(f"[dt_step_replay] ⚠️ missing raw day: {path}")
        return []
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        log(f"[dt_step_replay] ❌ failed to read {path}: {e}")
        return []


def _parse_any_ts(ts_raw: Any) -> Optional[datetime]:
    """Parse bar timestamps into tz-aware UTC datetime.

    Supports:
      - epoch seconds/ms/us/ns numbers
      - numeric strings
      - ISO strings (with or without Z)
    """
    if ts_raw is None:
        return None
    try:
        if isinstance(ts_raw, (int, float)):
            v = float(ts_raw)
            if v > 1e12:
                v = v / 1e9
            elif v > 1e10:
                v = v / 1e6
            return datetime.fromtimestamp(v, tz=timezone.utc)
        s = str(ts_raw).strip()
        if s.isdigit():
            return _parse_any_ts(float(s))
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _bar_ts(bar: Dict[str, Any]) -> Optional[datetime]:
    if not isinstance(bar, dict):
        return None
    return _parse_any_ts(bar.get("ts") or bar.get("t") or bar.get("timestamp"))


def _bar_close(bar: Dict[str, Any]) -> Optional[float]:
    if not isinstance(bar, dict):
        return None
    try:
        v = bar.get("c", bar.get("close", bar.get("price")))
        return float(v) if v is not None else None
    except Exception:
        return None


def _normalize_proba(row: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    s = 0.0
    for k in _ALLOWED:
        try:
            out[k] = float(row.get(k, 0.0))
        except Exception:
            out[k] = 0.0
        s += out[k]
    if s <= 0:
        return {"BUY": 0.0, "HOLD": 1.0, "SELL": 0.0}
    return {k: v / s for k, v in out.items()}


def _features_to_row(sym: str, feats: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"symbol": sym}
    for k, v in (feats or {}).items():
        if k in {"ts", "timestamp", "symbol"}:
            continue
        try:
            row[k] = float(v)
        except Exception:
            row[k] = 0.0
    return row


def _attach_predictions_to_rolling(rolling: Dict[str, Any], *, models: Any, now_utc: datetime) -> int:
    syms = [s for s in rolling.keys() if not str(s).startswith("_")]
    syms.sort()

    rows: List[Dict[str, Any]] = []
    used: List[str] = []
    for sym in syms:
        node = ensure_symbol_node(rolling, sym)
        feats = node.get("features_dt")
        if not isinstance(feats, dict) or not feats:
            continue
        rows.append(_features_to_row(sym, feats))
        used.append(sym)

    if not rows:
        return 0

    df = pd.DataFrame.from_records(rows).set_index("symbol")
    proba_df, _ = score_intraday_batch(df, models=models)

    ts = now_utc.isoformat().replace("+00:00", "Z")
    predicted = 0

    for sym in used:
        if sym not in proba_df.index:
            continue
        node = ensure_symbol_node(rolling, sym)
        row = proba_df.loc[sym].to_dict()
        proba = _normalize_proba(row)
        label = max(proba.items(), key=lambda kv: kv[1])[0]
        if label not in _ALLOWED:
            label = "HOLD"
        node["predictions_dt"] = {
            "label": label,
            "proba": proba,
            "confidence": float(max(proba.values())),
            "ts": ts,
            "meta": {
                "source": "score_intraday_batch",
                "lgb_active": bool(getattr(models, "lgb", None) is not None),
                "lstm_active": bool(getattr(models, "lstm", None) is not None),
                "transf_active": bool(getattr(models, "transf", None) is not None),
            },
        }
        rolling[sym] = node
        predicted += 1

    return predicted


def _select_time_index(raw_day: List[Dict[str, Any]]) -> List[datetime]:
    """Pick a canonical timestamp list for stepping.

    Strategy: use the symbol with the most bars.
    """
    best: List[Dict[str, Any]] = []
    for entry in raw_day:
        bars = entry.get("bars") if isinstance(entry, dict) else None
        if isinstance(bars, list) and len(bars) > len(best):
            best = bars

    ts: List[datetime] = []
    for b in best:
        dt = _bar_ts(b)
        if dt is not None:
            ts.append(dt)
    ts = sorted(set(ts))
    return ts


def _subsample_times(times: List[datetime], *, step_minutes: int) -> List[datetime]:
    if not times:
        return []
    step_s = max(1, int(step_minutes)) * 60
    out = [times[0]]
    last = times[0]
    for t in times[1:]:
        if (t - last).total_seconds() >= step_s - 0.5:
            out.append(t)
            last = t
    if out[-1] != times[-1]:
        out.append(times[-1])
    return out


@dataclass
class StepReplaySummary:
    date: str
    steps: int
    symbols: int
    predicted_steps: int


def replay_intraday_day_step(
    *,
    date_str: str,
    step_minutes: int = 5,
    max_symbols: Optional[int] = None,
    exec_cfg: Optional[ExecutionConfig] = None,
) -> StepReplaySummary:
    """Run a step-wise replay over a single day.

    IMPORTANT: Caller should set env overrides so this doesn't touch live files:
      - DT_TRUTH_DIR
      - DT_ROLLING_PATH
      - DT_LOCK_PATH
      - DT_BOT_LEDGER_PATH
    """

    raw_day = _load_raw_day(date_str)
    if not raw_day:
        return StepReplaySummary(date=date_str, steps=0, symbols=0, predicted_steps=0)

    # Cap universe for speed/debug.
    if max_symbols is not None:
        raw_day = raw_day[: max(0, int(max_symbols))]

    # Pre-parse bar timestamps + maintain per-symbol progressive index.
    per_sym: Dict[str, Dict[str, Any]] = {}
    for entry in raw_day:
        sym = str((entry or {}).get("symbol") or "").upper()
        bars = (entry or {}).get("bars")
        if not sym or not isinstance(bars, list):
            continue
        parsed: List[Tuple[datetime, Dict[str, Any]]] = []
        for b in bars:
            dt = _bar_ts(b)
            if dt is None:
                continue
            parsed.append((dt, b))
        parsed.sort(key=lambda x: x[0])
        per_sym[sym] = {"parsed": parsed, "i": 0}

    times = _select_time_index(raw_day)
    times = _subsample_times(times, step_minutes=step_minutes)
    if not times:
        return StepReplaySummary(date=date_str, steps=0, symbols=len(per_sym), predicted_steps=0)

    models = load_intraday_models()
    cfg = exec_cfg or ExecutionConfig(dry_run=False)

    predicted_steps = 0
    for step_dt in times:
        # Build rolling snapshot at this step.
        rolling: Dict[str, Any] = {}
        for sym, st in per_sym.items():
            parsed = st["parsed"]
            i = int(st["i"])
            while i < len(parsed) and parsed[i][0] <= step_dt:
                i += 1
            st["i"] = i

            node = ensure_symbol_node(rolling, sym)
            node["bars_intraday"] = [b for _, b in parsed[:i]]
            # Ensure last_price is sane even before feature pass.
            if node["bars_intraday"]:
                px = _bar_close(node["bars_intraday"][-1])
                if px is not None:
                    node.setdefault("features_dt", {})
                    if isinstance(node["features_dt"], dict):
                        node["features_dt"].setdefault("last_price", float(px))
            rolling[sym] = node

        save_rolling(rolling)

        # Phase 2: daily plan + regime (time-aware)
        ensure_daily_plan(now_utc=step_dt)
        classify_intraday_regime(now_utc=step_dt)

        # Phase 1/2: context + features
        build_intraday_context(target_date=date_str, now_utc=step_dt)
        build_intraday_features(now_utc=step_dt)

        # Reload rolling (features attached) and attach predictions.
        # build_intraday_features writes to rolling file.
        # We read it back via save_rolling in feature engine; easiest is to re-open through file pipeline,
        # but feature engine already saved. We'll pull from disk by reusing our in-memory rolling snapshot
        # and merging feature/prediction steps via save_rolling again.
        # For simplicity we re-read from file using json/gzip is inside data_pipeline_dt; avoid here.
        from dt_backend.core.data_pipeline_dt import _read_rolling  # local import to avoid cycles
        rolling = _read_rolling()

        predicted = _attach_predictions_to_rolling(rolling, models=models, now_utc=step_dt)
        if predicted:
            predicted_steps += 1
        save_rolling(rolling)

        # Phase 3: policy + execution
        apply_intraday_policy()
        run_execution_intraday()

        # Phase 5: execute orders + manage exits
        execute_from_policy(cfg, now_utc=step_dt)

    return StepReplaySummary(date=date_str, steps=len(times), symbols=len(per_sym), predicted_steps=predicted_steps)
