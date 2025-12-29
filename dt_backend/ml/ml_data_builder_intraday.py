# dt_backend/ml/ml_data_builder_intraday.py — v5.1 (NEWS-AWARE + SAFE)
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import json

from dt_backend.core.config_dt import DT_PATHS  # type: ignore
from dt_backend.core.data_pipeline_dt import _read_rolling, log

def _lazy_pd():
    import pandas as pd
    return pd

def _intraday_news_dir() -> Path:
    root = DT_PATHS.get("ml_data_dt") or DT_PATHS.get("root") or "ml_data_dt"
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


def _infer_ts(node: Dict[str, Any]) -> datetime | None:
    feats = node.get("features_dt") or {}
    ts_raw = feats.get("ts") or feats.get("timestamp")

    if isinstance(ts_raw, str):
        try:
            return datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except Exception:
            try:
                return datetime.strptime(ts_raw.split(".")[0], "%Y-%m-%d %H:%M:%S")
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
        if isinstance(cand, str):
            try:
                dt = datetime.fromisoformat(cand.replace("Z", "+00:00"))
            except Exception:
                try:
                    dt = datetime.strptime(cand.split(".")[0], "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
        elif isinstance(cand, (int, float)):
            try:
                dt = datetime.utcfromtimestamp(float(cand))
            except Exception:
                continue
        else:
            continue

        if latest_ts is None or dt > latest_ts:
            latest_ts = dt

    return latest_ts or datetime.utcnow()


def _resolve_dataset_paths() -> List[Path]:
    paths: List[Path] = []

    if "dtml_intraday_dataset" in DT_PATHS:
        paths.append(Path(DT_PATHS["dtml_intraday_dataset"]))

    if "dtml_data" in DT_PATHS:
        paths.append(Path(DT_PATHS["dtml_data"]) / "training_data_intraday.parquet")

    if not paths:
        paths.append(Path("ml_data_dt") / "training_data_intraday.parquet")

    seen = set()
    out: List[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def build_intraday_dataset(max_symbols: int | None = None) -> Dict[str, Any]:
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

        # ✅ IMPORTANT: copy so we do NOT mutate rolling[sym]["features_dt"]
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
    paths = _resolve_dataset_paths()

    try:
        primary = paths[0]
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
        }
    except Exception as e:
        log(f"[dt_ml_builder] ⚠️ failed to write dataset(s) {paths}: {e}")
        return {"status": "error", "error": str(e), "rows": 0, "symbols": 0}