# dt_backend/signals_builder.py — v1.0 (Intraday signals board)
# Aggregates the latest intraday predictions into ranked BUY/SELL signal boards.
# Output → ml_data_dt/signals/intraday_predictions.json (already written by ai_model_intraday)
#           + ranked boards (top_buy.json, top_sell.json)

from __future__ import annotations
import os, sys, json
from pathlib import Path
from datetime import datetime
import pandas as pd

# safe import shim
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from backend.data_pipeline import log  # type: ignore
except Exception:
    def log(msg: str): print(msg, flush=True)

from dt_backend.config_dt import DT_PATHS

SIGNALS_DIR = DT_PATHS["dtsignals"]
RAW_PATH = SIGNALS_DIR / "intraday_predictions.json"
TOP_BUY_PATH = SIGNALS_DIR / "top_buy.json"
TOP_SELL_PATH = SIGNALS_DIR / "top_sell.json"

def _safe_load_json(path: Path) -> list[dict]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                js = json.load(f)
                if isinstance(js, dict) and "rows" in js:
                    return js["rows"]
                elif isinstance(js, list):
                    return js
    except Exception as e:
        log(f"[signals_builder] ⚠️ failed to load {path.name}: {e}")
    return []

def _rank_signals(rows: list[dict], side: str = "BUY", top_n: int = 20) -> pd.DataFrame:
    """Rank signals of a given side by confidence, momentum_score, orderflow_score."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df["label"].str.upper() == side.upper()]
    if df.empty:
        return df
    # rank key
    df["score_rank"] = (
        df["confidence"].fillna(0) * 0.6 +
        df.get("momentum_score", 0) * 0.25 +
        df.get("orderflow_score", 0) * 0.15
    )
    df = df.sort_values("score_rank", ascending=False).head(top_n)
    df["rank"] = range(1, len(df) + 1)
    df["generated_at"] = datetime.utcnow().isoformat()
    return df[["rank","symbol","confidence","momentum_score","orderflow_score","currentPrice","generated_at"]]

def build_intraday_signals():
    """
    Read predictions_intraday.json.gz and emit a simple
    (symbol -> {signal, score}) dict + write the rank file.
    """
    from dt_backend.config_dt import DT_PATHS
    import gzip, json
    sig_dir = DT_PATHS["ml_data_dt"] / "signals"
    preds_f = sig_dir / "predictions_intraday.json.gz"
    if not preds_f.exists():
        return {"status": "skipped_no_predictions"}

    with gzip.open(preds_f, "rt", encoding="utf-8") as f:
        preds = json.load(f)

    out_map = {}
    for sym, rec in preds.items():
        score = float(rec.get("score", 0.0))
        sig = "BUY" if score > 0 else "SELL" if score < 0 else "HOLD"
        out_map[sym] = {"signal": sig, "score": score}

    # also write the rank file used by rank_fetch_scheduler
    rank_file = sig_dir / "prediction_rank_fetch.json.gz"
    ranked = dict(sorted(out_map.items(), key=lambda kv: kv[1]["score"], reverse=True))
    with gzip.open(rank_file, "wt", encoding="utf-8") as g:
        json.dump({s: {"rank": i+1} for i, s in enumerate(ranked.keys())}, g)

    return {"status": "ok", "signals": len(out_map)}

def write_intraday_signals(top_n: int = 20) -> Path | None:
    """
    Reads the latest intraday_predictions.json → writes ranked boards:
        - top_buy.json
        - top_sell.json
    Returns the main board folder path.
    """
    rows = _safe_load_json(RAW_PATH)
    if not rows:
        log("[signals_builder] ⚠️ no predictions found.")
        return None

    df_buy = _rank_signals(rows, "BUY", top_n=top_n)
    df_sell = _rank_signals(rows, "SELL", top_n=top_n)

    try:
        out_buy = df_buy.to_dict(orient="records") if not df_buy.empty else []
        out_sell = df_sell.to_dict(orient="records") if not df_sell.empty else []

        with open(TOP_BUY_PATH, "w", encoding="utf-8") as f:
            json.dump({"generated_at": datetime.utcnow().isoformat(), "rows": out_buy}, f, indent=2)
        with open(TOP_SELL_PATH, "w", encoding="utf-8") as f:
            json.dump({"generated_at": datetime.utcnow().isoformat(), "rows": out_sell}, f, indent=2)

        log(f"[signals_builder] ✅ wrote {len(out_buy)} BUYs / {len(out_sell)} SELLs")
    except Exception as e:
        log(f"[signals_builder] ⚠️ failed to write boards: {e}")
        return None

    return SIGNALS_DIR

if __name__ == "__main__":
    out = write_intraday_signals()
    print(f"boards → {out}")
