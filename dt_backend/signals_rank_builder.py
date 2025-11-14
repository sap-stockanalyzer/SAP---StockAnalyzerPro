"""
signals_rank_builder.py — AION Intraday Rank Generator
------------------------------------------------------
Ranks all tickers based on AI predictions and saves to
ml_data_dt/signals/prediction_rank_fetch.json.gz
"""

import os, gzip, json
from datetime import datetime, timezone
import pandas as pd
from backend.data_pipeline import log
from dt_backend.config_dt import DT_PATHS

SIGNALS_PATH = DT_PATHS["dtsignals"] / "prediction_rank_fetch.json.gz"

def build_intraday_signals(predictions: dict | list):
    """
    predictions: { "AAPL": {"predicted": 0.016, "confidence": 0.91}, ... }
    Builds rank file based on predicted × confidence.
    """
    if not predictions:
        log("[signals_rank_builder] ⚠️ no predictions found.")
        return None

    if isinstance(predictions, list):
        predictions = {row.get("symbol"): row for row in predictions if row.get("symbol")}

    rows = []
    for sym, vals in (predictions or {}).items():
        if not sym:
            continue
        predicted = vals.get("predicted")
        score = vals.get("score", predicted)
        if score is None:
            score = vals.get("confidence")
        confidence = vals.get("confidence") or vals.get("proba")
        rows.append({
            "symbol": sym,
            "predicted": float(predicted) if predicted is not None else 0.0,
            "confidence": float(confidence) if confidence is not None else 0.0,
            "label": vals.get("label"),
            "score": float(score) if score is not None else 0.0,
        })

    # Convert to DataFrame for easy sorting
    df = pd.DataFrame(rows)
    if df.empty:
        log("[signals_rank_builder] ⚠️ No rows available for ranking.")
        return None

    df.sort_values("score", ascending=False, inplace=True)
    df["rank"] = range(1, len(df) + 1)

    data = {
        "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "owned": [],
        "ranks": df[["symbol", "rank", "predicted", "confidence", "label", "score"]].to_dict(orient="records"),
    }

    os.makedirs(DT_PATHS["dtsignals"], exist_ok=True)
    with gzip.open(SIGNALS_PATH, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    log(f"[signals_rank_builder] ⚡ {len(df)} tickers ranked and saved → {SIGNALS_PATH}")
    return SIGNALS_PATH
