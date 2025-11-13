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

def build_intraday_signals(predictions: dict):
    """
    predictions: { "AAPL": {"predicted": 0.016, "confidence": 0.91}, ... }
    Builds rank file based on predicted × confidence.
    """
    if not predictions:
        log("[signals_rank_builder] ⚠️ no predictions found.")
        return None

    # Convert to DataFrame for easy sorting
    df = pd.DataFrame([
        {"symbol": sym,
         "predicted": vals.get("predicted", 0),
         "confidence": vals.get("confidence", 0),
         "score": vals.get("predicted", 0) * vals.get("confidence", 0)}
        for sym, vals in predictions.items()
    ])

    df.sort_values("score", ascending=False, inplace=True)
    df["rank"] = range(1, len(df) + 1)

    data = {
        "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "owned": [],
        "ranks": df[["symbol", "rank", "predicted", "confidence"]].to_dict(orient="records"),
    }

    os.makedirs(DT_PATHS["dtsignals"], exist_ok=True)
    with gzip.open(SIGNALS_PATH, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    log(f"[signals_rank_builder] ⚡ {len(df)} tickers ranked and saved → {SIGNALS_PATH}")
    return SIGNALS_PATH
