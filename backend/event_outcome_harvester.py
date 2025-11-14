"""Event Outcome Harvester — updates priors via Bayesian-lite blend."""
from __future__ import annotations
import os, json, glob, time, datetime
from typing import Dict, List
import pandas as pd
from .data_pipeline import log, _read_rolling
from .config import PATHS  # ✅ unified path import

OUT_DIR = PATHS["news"]  # ✅ replaces 'news_cache'
PRIORS_PATH = PATHS["backend_service"].parent / "event_priors.json"  # ✅ inside backend/

def _latest_events_paths(n: int = 14) -> List[str]:
    pats = sorted(
        glob.glob(str(OUT_DIR / "news_events_*.parquet"))
        + glob.glob(str(OUT_DIR / "news_events_*.json")),
        reverse=True
    )
    return pats[:n]

def _load_events(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            if p.endswith(".parquet"):
                frames.append(pd.read_parquet(p))
            else:
                with open(p, "r", encoding="utf-8") as f:
                    frames.append(pd.DataFrame(json.load(f)))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _ret_over(hist: List[Dict], pub_date: str, days: int) -> float:
    if not hist:
        return 0.0
    df = pd.DataFrame(hist)
    if "date" not in df or "close" not in df:
        return 0.0
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    try:
        t0 = pd.to_datetime(pub_date).normalize()
    except Exception:
        return 0.0
    row = df[df["date"] >= t0]
    if row.empty:
        return 0.0
    c0 = float(row.iloc[0]["close"])
    end_i = min(row.index[0] + days, df.index[-1])
    c1 = float(df.loc[end_i, "close"])
    if c0 == 0:
        return 0.0
    return (c1 - c0) / c0

def _update_priors(priors: Dict, stats: Dict[str, Dict[str, float]]) -> Dict:
    alpha = float(priors.get("alpha_update", 0.15))
    for et, m in stats.items():
        priors["priors"].setdefault(et, {"short": 0.0, "mid": 0.0, "long": 0.0})
        for h in ("short", "mid", "long"):
            old = float(priors["priors"][et].get(h, 0.0))
            new = float(m.get(h, old))
            priors["priors"][et][h] = (1 - alpha) * old + alpha * new
    priors["updated_at"] = datetime.datetime.utcnow().isoformat()
    return priors

def run_event_harvest() -> Dict:
    start = time.time()
    paths = _latest_events_paths(14)
    if not paths:
        log("[event_outcomes] ℹ️ no event files found.")
        return {}
    ev = _load_events(paths)
    if ev.empty:
        log("[event_outcomes] ℹ️ empty events dataframe.")
        return {}

    rolling = _read_rolling() or {}
    rows = []
    for _, r in ev.iterrows():
        sym = r.get("ticker")
        if not sym or sym not in rolling:
            continue
        hist = (rolling.get(sym) or {}).get("history") or []
        pub = r.get("published_at")
        realized_1d = _ret_over(hist, pub, 1)
        realized_5d = _ret_over(hist, pub, 5)
        realized_20d = _ret_over(hist, pub, 20)
        rows.append({
            "ticker": sym,
            "event_type": r.get("event_type"),
            "stance": r.get("stance"),
            "published_at": pub,
            "realized_1d": realized_1d,
            "realized_5d": realized_5d,
            "realized_20d": realized_20d
        })

    if not rows:
        log("[event_outcomes] ℹ️ no realized rows computed.")
        return {}

    df = pd.DataFrame(rows)
    df["adj_1d"] = df["realized_1d"] * df["stance"].fillna(0)
    df["adj_5d"] = df["realized_5d"] * df["stance"].fillna(0)
    df["adj_20d"] = df["realized_20d"] * df["stance"].fillna(0)
    stats = (
        df.groupby("event_type")[["adj_1d", "adj_5d", "adj_20d"]].mean()
        .rename(columns={"adj_1d": "short", "adj_5d": "mid", "adj_20d": "long"})
    ).to_dict(orient="index")

    try:
        with open(PRIORS_PATH, "r", encoding="utf-8") as f:
            pri = json.load(f)
    except Exception:
        pri = {"priors": {}}
    pri = _update_priors(pri, stats)
    with open(PRIORS_PATH, "w", encoding="utf-8") as f:
        json.dump(pri, f, indent=2, ensure_ascii=False)

    out_path = OUT_DIR / f"event_outcomes_{datetime.datetime.utcnow().strftime('%Y%m%d')}.parquet"  # ✅ unified
    try:
        df.to_parquet(out_path, index=False)
    except Exception:
        out_path = out_path.with_suffix(".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    dur = time.time() - start
    log(f"[event_outcomes] ✅ {len(df)} rows; priors updated; snapshot → {out_path} ({dur:.1f}s)")
    return {"path": str(out_path)}

if __name__ == "__main__":
    run_event_harvest()
