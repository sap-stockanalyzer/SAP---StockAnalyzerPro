# dt_backend/ml/ml_data_builder_intraday.py — v5.2 (LABEL-AWARE + SAFE)
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dt_backend.core.config_dt import DT_PATHS  # type: ignore
from dt_backend.core.data_pipeline_dt import _read_rolling, log

def _lazy_pd():
    import pandas as pd
    return pd


# -------------------------------
# Intraday news intel (optional)
# -------------------------------
def _intraday_news_dir() -> Path:
    root = (
        DT_PATHS.get("ml_data_dt")
        or DT_PATHS.get("dtml_data")
        or DT_PATHS.get("root")
        or "ml_data_dt"
    )
    d = Path(root) / "news_intraday"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _latest_news_intraday_file() -> Optional[Path]:
    d = _intraday_news_dir()
    files = sorted(d.glob("news_intraday_*.json"))
    return files[-1] if files else None


def _load_latest_intraday_news_intel() -> Dict[str, Dict[str, Any]]:
    path = _latest_news_intraday_file()
    if not path:
        log("[dt_ml_builder] ℹ️ No intraday news intel found.")
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        syms = raw.get("symbols") or {}
        if not isinstance(syms, dict):
            log("[dt_ml_builder] ⚠️ Invalid intraday news file format.")
            return {}
        log(f"[dt_ml_builder] ✅ Loaded intraday news intel from {path.name}")
        return syms
    except Exception as e:
        log(f"[dt_ml_builder] ⚠️ Failed reading intraday news file: {e}")
        return {}


def _merge_news_into_features(sym: str, feats: Dict[str, Any], intel: Dict[str, Any]) -> None:
    node = intel.get(sym.upper()) or {}

    shock = node.get("shock") or {}
    buzz = node.get("buzz") or {}

    feats["news_shock_score"] = float(shock.get("shock_score", 0.0) or 0.0)
    feats["news_shock_direction"] = float(shock.get("shock_direction", 0.0) or 0.0)
    feats["news_recent_articles"] = float(shock.get("recent_articles", 0.0) or 0.0)

    feats["news_buzz_count"] = float(buzz.get("buzz_count", 0.0) or 0.0)
    feats["news_buzz_score"] = float(buzz.get("buzz_score", 0.0) or 0.0)


# -------------------------------
# Timestamp inference
# -------------------------------
def _infer_ts(node: Dict[str, Any]) -> datetime | None:
    feats = node.get("features_dt") or {}
    ts_raw = feats.get("ts") or feats.get("timestamp")

    if isinstance(ts_raw, str):
        for fmt in ("%Y-%m-%d %H:%M:%S",):
            try:
                return datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except Exception:
                try:
                    return datetime.strptime(ts_raw.split(".")[0], fmt)
                except Exception:
                    pass

    if isinstance(ts_raw, (int, float)):
        try:
            return datetime.utcfromtimestamp(float(ts_raw))
        except Exception:
            pass

    bars = node.get("bars_intraday") or []
    latest_ts = None
    for bar in bars:
        if not isinstance(bar, dict):
            continue
        cand = bar.get("ts") or bar.get("t") or bar.get("timestamp")
        dt = None
        if isinstance(cand, str):
            try:
                dt = datetime.fromisoformat(cand.replace("Z", "+00:00"))
            except Exception:
                try:
                    dt = datetime.strptime(cand.split(".")[0], "%Y-%m-%d %H:%M:%S")
                except Exception:
                    dt = None
        elif isinstance(cand, (int, float)):
            try:
                dt = datetime.utcfromtimestamp(float(cand))
            except Exception:
                dt = None

        if dt is None:
            continue
        if latest_ts is None or dt > latest_ts:
            latest_ts = dt

    return latest_ts or datetime.utcnow()


# -------------------------------
# Dataset paths
# -------------------------------
def _resolve_dataset_paths() -> List[Path]:
    paths: List[Path] = []

    # canonical keys (we accept multiple to stay compatible with refactors)
    if "dtml_intraday_dataset" in DT_PATHS:
        paths.append(Path(DT_PATHS["dtml_intraday_dataset"]))
    if "dtml_data" in DT_PATHS:
        paths.append(Path(DT_PATHS["dtml_data"]) / "training_data_intraday.parquet")
    if "ml_data_dt" in DT_PATHS:
        paths.append(Path(DT_PATHS["ml_data_dt"]) / "training_data_intraday.parquet")

    if not paths:
        paths.append(Path("ml_data_dt") / "training_data_intraday.parquet")

    seen = set()
    out: List[Path] = []
    for p in paths:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


# -------------------------------
# Label helpers (FAST / SAFE)
# -------------------------------
_LABEL2ID = {"SELL": 0, "HOLD": 1, "BUY": 2}

def _env_float(name: str, default: float) -> float:
    try:
        v = (os.environ.get(name, "") or "").strip()
        return float(v) if v else float(default)
    except Exception:
        return float(default)

def _auto_label_from_returns(df) -> None:
    """Add label + label_id using a *cross-sectional* return signal.

    Why this exists:
      - Your current builder is a snapshot builder (one row per symbol).
      - That means we don't have a forward-looking target without a replay store.
      - But your pipeline needs a real LightGBM model file to avoid crashing.
      - So we auto-label using returns across symbols (quantiles) as a bootstrap.

    Source column priority:
      1) pct_chg_from_open
      2) intraday_return
      3) last_price (fallback -> no labels)
    """
    if "label" in df.columns or "label_id" in df.columns:
        return

    src = None
    for cand in ("pct_chg_from_open", "intraday_return"):
        if cand in df.columns:
            src = cand
            break

    if src is None:
        return

    r = df[src].astype(float)

    # Quantile labeling produces all 3 classes for most real distributions.
    # Safety thresholds allow you to widen/narrow HOLD band.
    q_lo = _env_float("DT_LABEL_Q_LO", 0.33)
    q_hi = _env_float("DT_LABEL_Q_HI", 0.67)
    q_lo = max(0.05, min(0.49, q_lo))
    q_hi = max(0.51, min(0.95, q_hi))

    lo = float(r.quantile(q_lo))
    hi = float(r.quantile(q_hi))

    # If distribution is degenerate (lo==hi), fall back to sign-based.
    if not (hi > lo):
        mild = _env_float("DT_LABEL_MILD", 0.0015)
        def _lab(x: float) -> str:
            if x >= mild:
                return "BUY"
            if x <= -mild:
                return "SELL"
            return "HOLD"
        labs = r.apply(lambda x: _lab(float(x)))
    else:
        def _lab(x: float) -> str:
            if x <= lo:
                return "SELL"
            if x >= hi:
                return "BUY"
            return "HOLD"
        labs = r.apply(lambda x: _lab(float(x)))

    df["label"] = labs.astype(str)
    df["label_id"] = df["label"].map(_LABEL2ID).astype(int)


def build_intraday_dataset(max_symbols: int | None = None) -> Dict[str, Any]:
    pd = _lazy_pd()

    rolling = _read_rolling()
    if not rolling:
        log("[dt_ml_builder] ⚠️ rolling empty, nothing to build.")
        return {"status": "empty", "rows": 0, "symbols": 0}

    intraday_news = _load_latest_intraday_news_intel()

    items = [(sym, node) for sym, node in rolling.items() if not str(sym).startswith("_")]
    items.sort(key=lambda kv: kv[0])

    if max_symbols is not None:
        items = items[: max(0, int(max_symbols))]

    rows: List[Dict[str, Any]] = []

    for sym, node in items:
        if not isinstance(node, dict):
            continue

        feats_raw = node.get("features_dt") or {}
        if not isinstance(feats_raw, dict) or not feats_raw:
            continue

        # copy so we do NOT mutate rolling[sym]["features_dt"]
        feats = dict(feats_raw)

        _merge_news_into_features(sym, feats, intraday_news)

        ts = _infer_ts(node)
        row: Dict[str, Any] = {"symbol": sym, "ts": ts}

        for k, v in feats.items():
            if k in {"symbol", "ts"}:
                continue
            row[k] = v

        rows.append(row)

    if not rows:
        log("[dt_ml_builder] ⚠️ no feature rows to write.")
        return {"status": "no_rows", "rows": 0, "symbols": 0}

    df = pd.DataFrame.from_records(rows)

    # ✅ add labels (bootstrap) if they aren't already present
    _auto_label_from_returns(df)

    paths = _resolve_dataset_paths()

    try:
        primary = paths[0]
        cols = list(df.columns)
        for i, p in enumerate(paths):
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p, index=False)
            if i == 0:
                log(f"[dt_ml_builder] ✅ wrote primary dataset → {p} (rows={len(df)}, symbols={df['symbol'].nunique()})")
            else:
                log(f"[dt_ml_builder] ↳ mirrored dataset → {p}")

        return {
            "status": "ok",
            "rows": int(len(df)),
            "symbols": int(df["symbol"].nunique()),
            "path": str(primary),
            "columns": cols,
            "labeled": bool(("label" in df.columns) or ("label_id" in df.columns)),
        }
    except Exception as e:
        log(f"[dt_ml_builder] ⚠️ failed to write dataset(s) {paths}: {e}")
        return {"status": "error", "error": str(e), "rows": 0, "symbols": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build intraday ML dataset (snapshot) from rolling features_dt.")
    parser.add_argument("--max_symbols", type=int, default=None)
    args = parser.parse_args()

    res = build_intraday_dataset(max_symbols=args.max_symbols)
    log(f"[dt_ml_builder] 📊 Result: {res}")
