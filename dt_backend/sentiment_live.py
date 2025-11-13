# dt_backend/sentiment_live.py — v1.0
# Rapid headline sentiment aggregator for intraday ticks.
# Tries dt caches first; falls back to backend news cache if available.

from __future__ import annotations
import os, sys, json, datetime
from pathlib import Path
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from backend.data_pipeline import log  # type: ignore
except Exception:
    def log(msg: str): print(msg, flush=True)

from dt_backend.config_dt import DT_PATHS

# Optional: use backend PATHS for existing news caches, if present
BACKEND_NEWS_DIR: Path | None = None
try:
    from backend.config import PATHS as BACKEND_PATHS  # type: ignore
    BACKEND_NEWS_DIR = BACKEND_PATHS.get("news")  # type: ignore
except Exception:
    BACKEND_NEWS_DIR = None

def _today_tag() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d")

def _safe_load_json(path: Path) -> dict | list | None:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _extract_ticker_sentiment(rows: List[dict]) -> Dict[str, float]:
    """
    Very simple aggregation: average sentiment per ticker across recent items.
    Expects article rows with fields: {'tickers': [], 'sentiment': <float  -1..+1>, ...}
    """
    sums: Dict[str, float] = {}
    cnts: Dict[str, int] = {}
    for r in rows or []:
        tickers = r.get("tickers") or []
        s = r.get("sentiment")
        try:
            s = float(s) if s is not None else None
        except Exception:
            s = None
        if s is None:
            continue
        for t in tickers:
            t = str(t).upper().strip()
            if not t:
                continue
            sums[t] = sums.get(t, 0.0) + s
            cnts[t] = cnts.get(t, 0) + 1
    return {k: (sums[k] / max(cnts.get(k, 1), 1)) for k in sums.keys()}

def load_recent_sentiment() -> Dict[str, float]:
    """
    Tries in order:
      1) dt_daily news (future extension) under data_dt/
      2) backend news raw cache: news_raw_YYYYMMDD.json
      3) returns {}
    """
    # 1) (reserved for dt-specific future cache)
    # dt_path = DT_PATHS["dtmetrics"].parent / f"dt_news_{_today_tag()}.json"
    # js = _safe_load_json(dt_path)
    # if isinstance(js, dict):
    #     rows = list(js.values())
    #     return _extract_ticker_sentiment(rows)

    # 2) fallback to backend news raw (if present)
    if BACKEND_NEWS_DIR:
        raw_path = BACKEND_NEWS_DIR / f"news_raw_{_today_tag()}.json"
        js = _safe_load_json(raw_path)
        if isinstance(js, dict):
            rows = list(js.values())
            return _extract_ticker_sentiment(rows)

    return {}

def attach_sentiment(df, symbol_col="symbol", join_on: str | None = None):
    """
    Utility: map sentiment to a dataframe of symbols.
    If join_on provided, it uses that column (e.g., last snapshot with symbol).
    """
    senti_map = load_recent_sentiment()
    if not isinstance(df, (list, tuple)) and hasattr(df, "assign"):
        if symbol_col in df.columns:
            return df.assign(sentiment_live=df[symbol_col].map(lambda s: senti_map.get(str(s).upper(), 0.0)))
    return df
