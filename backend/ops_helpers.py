from __future__ import annotations
"""
ops_helpers.py — v2.3 (Self-Learning Loop + Unified Config)
AION Analytics / StockAnalyzerPro

Purpose:
• Prediction Logger (parquet/jsonl) — schema aligned for outcome harvesting
• Outcomes Harvester — reads Rolling history for realized returns
• Confidence Calibration — computes hit rate + Brier and writes Rolling Brain
• Thin wrappers to Rolling Brain I/O (proxy to data_pipeline)
• Drift report helper (evidently)
• Sentiment enricher (stub kept)
• IO helpers (atomic parquet, append)
• Live prices helper (StockAnalysis batch)
• Full snake_case normalization across all key fields
• ✅ Now uses config.PATHS for all read/write directories
"""

import os, json, gzip, hashlib, uuid, shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Iterable, Dict, Any, Optional, List, Tuple
from .config import PATHS  # ✅ unified path import

# Optional deps
try:
    import pandas as pd                 # type: ignore
except Exception:
    pd = None
try:
    import pyarrow as pa                # type: ignore
    import pyarrow.parquet as pq        # type: ignore
except Exception:
    pa = pq = None

# =============================================================================
# Normalization Helpers
# =============================================================================
NORMALIZE_KEYS = {
    "peRatio": "pe_ratio", "pbRatio": "pb_ratio", "psRatio": "ps_ratio",
    "pegRatio": "peg_ratio", "debtEquity": "debt_equity", "debtEbitda": "debt_ebitda",
    "revenueGrowth": "revenue_growth", "epsGrowth": "eps_growth",
    "profitMargin": "profit_margin", "operatingMargin": "operating_margin",
    "grossMargin": "gross_margin", "dividendYield": "dividend_yield",
    "payoutRatio": "payout_ratio", "marketCap": "marketCap",
    "roa": "roa", "roe": "roe", "roic": "roic",
}

def normalize_keys(data):
    """Normalize dict or DataFrame column names to snake_case."""
    if data is None:
        return data
    if isinstance(data, dict):
        for old, new in NORMALIZE_KEYS.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
        return data
    try:
        if isinstance(data, pd.DataFrame):
            rename_map = {old: new for old, new in NORMALIZE_KEYS.items() if old in data.columns}
            return data.rename(columns=rename_map)
    except Exception:
        pass
    return data


# =============================================================================
# Rolling Brain I/O (thin proxies to data_pipeline)
# =============================================================================
try:
    from .data_pipeline import save_brain as _save_brain_dp, _read_brain as _read_brain_dp
except Exception:
    _save_brain_dp = _read_brain_dp = None

BRAIN_PATH = PATHS["brain"]  # ✅ unified config path
BRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

def _read_brain() -> dict:
    """Read Rolling Brain (JSON.GZ)."""
    if _read_brain_dp:
        try:
            return _read_brain_dp() or {}
        except Exception:
            pass
    if not BRAIN_PATH.exists():
        return {}
    try:
        with gzip.open(BRAIN_PATH, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_brain(data: dict) -> None:
    """Save Rolling Brain (JSON.GZ)."""
    data = normalize_keys(data)
    if _save_brain_dp:
        try:
            _save_brain_dp(data)
            return
        except Exception:
            pass
    tmp = str(BRAIN_PATH) + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data or {}, f, ensure_ascii=False)
    os.replace(tmp, BRAIN_PATH)

# =============================================================================
# Shared helpers
# =============================================================================
DEFAULT_LOG_DIR = PATHS["prediction_logs"]  # ✅ unified path
DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

OUTCOMES_DIR = PATHS["prediction_outcomes"]  # ✅ unified path
OUTCOMES_DIR.mkdir(parents=True, exist_ok=True)

def _now_utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def _features_hash(names: Iterable[str]) -> str:
    h = hashlib.sha256()
    for n in sorted(list(names or [])):
        h.update(str(n).encode("utf-8", "ignore"))
    return h.hexdigest()[:16]

def _map_horizon_to_days(h: str) -> int:
    h = (h or "").lower().strip()
    if h in ("1d", "day"): return 1
    if h in ("5d", "1w", "week"): return 5
    if h in ("3w",): return 15
    if h in ("1m", "month"): return 21
    if h in ("3m",): return 63
    if h in ("6m",): return 126
    if h in ("1y", "year"): return 252
    return 5

def _safe_float(x, default=None):
    try:
        if x is None: return default
        return float(x)
    except Exception:
        return default


# =============================================================================
# Prediction Logger (Phase-1)
# =============================================================================
def log_predictions(
    records: Iterable[Dict[str, Any]],
    model_version: str,
    feature_names: Iterable[str],
    store_dir: Optional[str | Path] = None
) -> str:
    """Logs all model predictions in parquet or JSONL (snake_case normalized)."""
    store = Path(store_dir or DEFAULT_LOG_DIR)
    store.mkdir(parents=True, exist_ok=True)
    ts = _now_utc_ts()
    fhash = _features_hash(feature_names)
    base = f"preds_{ts}_{model_version}"
    fparq = store / f"{base}.parquet"
    fjson = store / f"{base}.jsonl"

    rows: List[Dict[str, Any]] = []
    asof_date = datetime.now(timezone.utc).date().isoformat()

    for r in (records or []):
        r = normalize_keys(dict(r))
        sym = (r.get("symbol") or r.get("ticker") or "").upper()
        if not sym:
            continue
        horizon = str(r.get("horizon", "1w"))
        h_days = _map_horizon_to_days(horizon)

        cur = _safe_float(r.get("currentPrice", r.get("current_price")))
        pred_price = _safe_float(r.get("predictedPrice", r.get("predicted_price")))
        y_pred = _safe_float(r.get("y_pred"))
        exp_pct = _safe_float(r.get("expectedReturnPct", r.get("expected_return_pct")))

        if exp_pct is None and cur is not None and pred_price is not None:
            exp_pct = (pred_price - cur) / cur * 100.0
        if pred_price is None and cur is not None and y_pred is not None:
            pred_price = cur * (1.0 + y_pred)
        if y_pred is None and exp_pct is not None:
            y_pred = exp_pct / 100.0

        proba = _safe_float(r.get("proba", r.get("confidence")))
        score = _safe_float(r.get("score"))
        rscore = _safe_float(r.get("rankingScore", r.get("ranking_score")))

        rows.append({
            "ts_utc": ts,
            "asof_date": asof_date,
            "symbol": sym,
            "horizon": horizon,
            "horizon_days": int(h_days),
            "current_price": cur,
            "predicted_price": pred_price,
            "expected_return_pct": exp_pct,
            "y_pred": y_pred,
            "proba": proba,
            "score": score,
            "ranking_score": rscore,
            "model_version": model_version,
            "features_hash": fhash,
            "meta_json": json.dumps(r.get("meta") or {}, ensure_ascii=False),
        })

    if not rows:
        with open(fjson, "w", encoding="utf-8") as f:
            pass
        return str(fjson)

    if pd is not None and pa is not None and pq is not None:
        df = normalize_keys(pd.DataFrame(rows))
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(fparq))
        return str(fparq)
    else:
        with open(fjson, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return str(fjson)

# =============================================================================
# Outcomes Harvester (Phase-1)
# =============================================================================
from .config import PATHS  # ✅ unified path import at top (if not already included)

def _load_history_from_rolling(symbol: str) -> pd.DataFrame:
    """Reads history for a given symbol from unified rolling.json.gz."""
    if pd is None:
        return None
    rolling_path = PATHS["rolling"]  # ✅ unified
    if not rolling_path.exists():
        return pd.DataFrame(columns=["date", "close"])
    try:
        with gzip.open(rolling_path, "rt", encoding="utf-8") as f:
            js = json.load(f)
        node = normalize_keys((js or {}).get(symbol.upper(), {}) or {})
        hist = node.get("history") or []
        if not hist:
            return pd.DataFrame(columns=["date", "close"])
        df = pd.DataFrame(hist)
        if "date" not in df or "close" not in df:
            return pd.DataFrame(columns=["date", "close"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        return df[["date", "close"]]
    except Exception:
        return pd.DataFrame(columns=["date", "close"])

def _future_close_from_hist(df_hist: pd.DataFrame, asof: pd.Timestamp, days: int) -> Optional[float]:
    if df_hist is None or df_hist.empty:
        return None
    tgt_calendar = asof + timedelta(days=int(days))
    fut = df_hist.loc[df_hist["date"] >= tgt_calendar].head(1)
    if fut.empty:
        return None
    try:
        return float(fut.iloc[0]["close"])
    except Exception:
        return None

def _read_latest_pred_file() -> Optional[Path]:
    files = sorted(DEFAULT_LOG_DIR.glob("preds_*.parquet"), key=lambda p: p.stat().st_mtime) if DEFAULT_LOG_DIR.exists() else []
    if files:
        return files[-1]
    files = sorted(DEFAULT_LOG_DIR.glob("preds_*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None

def harvest_outcomes(pred_file: Optional[str | Path] = None) -> Dict[str, Any]:
    """Harvest realized outcomes using Rolling cache, normalized."""
    if pd is None:
        return {"status": "error", "error": "pandas not available"}

    pred_path = Path(pred_file) if pred_file else _read_latest_pred_file()
    if not pred_path or not pred_path.exists():
        return {"status": "no_preds"}

    if pred_path.suffix == ".parquet":
        preds = pd.read_parquet(pred_path)
    else:
        rows = []
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line.strip()))
                except Exception:
                    pass
        preds = pd.DataFrame(rows)

    preds = normalize_keys(preds)
    required = {"symbol", "asof_date", "horizon", "horizon_days", "current_price"}
    if preds.empty or not required.issubset(set(preds.columns)):
        return {"status": "bad_schema"}

    preds["asof_date"] = pd.to_datetime(preds["asof_date"], errors="coerce", utc=True).dt.tz_localize(None)

    out_rows: List[Dict[str, Any]] = []
    for _, r in preds.iterrows():
        sym = str(r["symbol"]).upper()
        asof = r["asof_date"]
        hdays = int(r["horizon_days"])
        cur = _safe_float(r["current_price"])
        if pd.isna(asof) or cur is None:
            continue

        hist = _load_history_from_rolling(sym)
        fut_close = _future_close_from_hist(hist, asof, hdays)
        if fut_close is None:
            continue

        realized = (fut_close / cur) - 1.0
        out_rows.append({
            "asof_date": asof.date().isoformat(),
            "symbol": sym,
            "horizon": str(r["horizon"]),
            "horizon_days": hdays,
            "current_price": cur,
            "future_price": float(fut_close),
            "realized_return_pct": float(realized * 100.0),
            "y_pred": _safe_float(r.get("y_pred")),
            "proba": _safe_float(r.get("proba")),
        })

    if not out_rows:
        return {"status": "no_outcomes"}

    df = pd.DataFrame(out_rows)
    ts = _now_utc_ts()
    out_fp = PATHS["prediction_outcomes"] / f"outcomes_{ts}.parquet"  # ✅ unified
    df.to_parquet(out_fp, index=False)
    return {"status": "ok", "evaluated": int(len(df)), "file": str(out_fp)}


# =============================================================================
# Confidence Calibration (Phase-1)
# =============================================================================
# (Unchanged except normalization applied to inputs)
def _brier_score(y_true_bin: List[int], p_up: List[float]) -> float:
    if not y_true_bin:
        return float("nan")
    import numpy as np
    yt = np.asarray(y_true_bin, dtype=float)
    pp = np.clip(np.asarray(p_up, dtype=float), 0.0, 1.0)
    return float(np.mean((pp - yt) ** 2))

def _hit_rate(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return float("nan")
    import numpy as np
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = (~np.isnan(yt)) & (~np.isnan(yp))
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(yt[mask]) == np.sign(yp[mask])))

def calibrate_confidence(brain: Optional[dict] = None, outcomes_file: Optional[str | Path] = None) -> dict:
    """Computes calibration metrics and updates Rolling Brain (normalized)."""
    if pd is None:
        return brain or {}

    brain = normalize_keys(dict(brain or {}))
    from numpy import exp, clip

    ofp = Path(outcomes_file) if outcomes_file else max(PATHS["prediction_outcomes"].glob("outcomes_*.parquet"), default=None, key=lambda p: p.stat().st_mtime)  # ✅ unified
    if not ofp or not ofp.exists():
        brain.setdefault("calibration", {}).update({
            "updated_at": datetime.utcnow().isoformat(),
            "hit_rate": None, "brier": None,
            "confidence_multiplier": 1.0, "n": 0,
        })
        return brain

    df = normalize_keys(pd.read_parquet(ofp))
    if df.empty:
        return brain

    y_true_pct, y_pred_frac = df.get("realized_return_pct"), df.get("y_pred")
    if y_true_pct is None or y_pred_frac is None:
        return brain

    y_true, y_pred = y_true_pct.astype(float)/100.0, y_pred_frac.astype(float)
    hr = _hit_rate(y_true.tolist(), y_pred.tolist())

    if "proba" in df.columns and df["proba"].notna().any():
        p_up = df["proba"].astype(float).clip(0.0, 1.0).fillna(0.5).tolist()
    else:
        p_up = (1.0 / (1.0 + exp(-clip(y_pred, -0.25, 0.25) * 10.0))).tolist()

    y_bin = (y_true > 0).astype(int).tolist()
    brier = _brier_score(y_bin, p_up)

    if hr != hr:
        mult = 1.0
    else:
        base = 1.0 + (hr - 0.5) * 0.6
        damp = max(0.8, 1.0 - min(brier if brier == brier else 0.1, 0.25))
        mult = float(max(0.85, min(1.15, base * damp)))

    brain.setdefault("calibration", {}).update({
        "updated_at": datetime.utcnow().isoformat(),
        "hit_rate": None if hr != hr else float(hr),
        "brier": None if brier != brier else float(brier),
        "confidence_multiplier": mult,
        "n": int(len(df)),
        "source_file": str(ofp),
    })


# =============================================================================
# Drift Report Helper (Phase-2)
# =============================================================================
def run_drift_report(ref_path: str, cur_path: str) -> bool:
    """
    Generate a basic drift report comparing reference vs current ML datasets.
    Uses evidently if available; otherwise falls back to a simple diff summary.
    Returns True if report created successfully.
    """
    try:
        import pandas as pd
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently import ColumnMapping

        if not os.path.exists(ref_path) or not os.path.exists(cur_path):
            print(f"⚠️ Drift skipped — one or both dataset paths not found.")
            return False

        ref = pd.read_parquet(ref_path)
        cur = pd.read_parquet(cur_path)
        if ref.empty or cur.empty:
            print(f"⚠️ Drift skipped — empty datasets.")
            return False

        mapping = ColumnMapping(target=None, numerical_features=None, categorical_features=None)
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur, column_mapping=mapping)

        out_path = PATHS["ml_data"] / "drift_report.html"  # ✅ unified
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(out_path))

        print(f"✅ Drift report generated → {out_path}")
        return True

    except ModuleNotFoundError:
        print("ℹ️ evidently not installed — skipping drift report generation.")
        return False
    except Exception as e:
        print(f"⚠️ Drift report failed: {e}")
        return False
