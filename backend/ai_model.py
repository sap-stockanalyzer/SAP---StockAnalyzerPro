"""
ai_model.py — v2.4 (Phase-2 Rolling-Native Inference + Unified Ranking Engine v3 + Dynamic Feature Fallback)
Author: AION Analytics (StockAnalyzerPro)

Adds:
✅ Auto-computation of missing technical features from history (volume, returns, momentum, volatility)
"""

from __future__ import annotations
import os, json, time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from .data_pipeline import get_symbol_data, _read_rolling, log
from .config import PATHS  # ✅ unified config import

# ---------------------------------------------------------------------
# Globals (use config.py)
# ---------------------------------------------------------------------
RANK_HISTORY_FILE = PATHS["rank_history"]
MODELS_DIR = PATHS["ml_models"]
HORIZONS = ["1w", "2w", "4w", "52w"]

# ---------------------------------------------------------------------
# Normalization Helpers
# ---------------------------------------------------------------------
NORMALIZE_KEYS = {
    "peRatio": "pe_ratio", "pbRatio": "pb_ratio", "psRatio": "ps_ratio",
    "pegRatio": "peg_ratio", "debtEquity": "debt_equity", "debtEbitda": "debt_ebitda",
    "revenueGrowth": "revenue_growth", "epsGrowth": "eps_growth",
    "profitMargin": "profit_margin", "operatingMargin": "operating_margin",
    "grossMargin": "gross_margin", "dividendYield": "dividend_yield",
    "payoutRatio": "payout_ratio", "marketCap": "marketCap",
    "roa": "roa", "roe": "roe", "roic": "roic"
}

def normalize_keys(node: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize ticker node fields from camelCase → snake_case."""
    if not isinstance(node, dict):
        return node
    for old, new in NORMALIZE_KEYS.items():
        if old in node and new not in node:
            node[new] = node.pop(old)
    return node


# ---------------------------------------------------------------------
# NEW: Compute missing technical features dynamically
# ---------------------------------------------------------------------
def _compute_features_from_history(hist: list[dict]) -> dict:
    """Derive momentum / volatility / returns from raw history if absent in node."""
    if not hist or len(hist) < 10:
        return {}
    try:
        df = pd.DataFrame(hist)[["date", "close", "volume"]].dropna()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
        df["ret1"] = df["close"].pct_change(1)
        df["ret5"] = df["close"].pct_change(5)
        df["ret10"] = df["close"].pct_change(10)
        df["volatility_10d"] = df["ret1"].rolling(10).std()
        df["momentum_5d"] = (df["close"] / df["close"].shift(5)) - 1
        latest = df.iloc[-1]
        return {
            "volume": float(latest.get("volume", 0.0) or 0.0),
            "volatility_10d": float(latest.get("volatility_10d", 0.0) or 0.0),
            "momentum_5d": float(latest.get("momentum_5d", 0.0) or 0.0),
            "ret1": float(latest.get("ret1", 0.0) or 0.0),
            "ret5": float(latest.get("ret5", 0.0) or 0.0),
            "ret10": float(latest.get("ret10", 0.0) or 0.0),
        }
    except Exception:
        return {}

# ---------------------------------------------------------------------
# Rank persistence
# ---------------------------------------------------------------------
def save_rank_history(current_rankings: Dict[str, int]) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d")
    data = {"timestamp": ts, "ranks": current_rankings}
    RANK_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RANK_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_rank_history() -> Dict[str, int]:
    if not RANK_HISTORY_FILE.exists():
        return {}
    try:
        with open(RANK_HISTORY_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        ranks = d.get("ranks", {})
        return {k: int(v) for k, v in ranks.items() if isinstance(k, str)}
    except Exception:
        return {}

# ---------------------------------------------------------------------
# Context loader
# ---------------------------------------------------------------------
def _load_context_safe_from_node(node: dict) -> dict:
    """Reads Phase-2 enrichment context fields from rolling."""
    return {
        "macro_breadth": float(node.get("macro_breadth", 0.5) or 0.5),
        "sector_momentum_1m": float(node.get("sector_momentum_1m", 0.0) or 0.0),
        "drift_score": {},
        "hit_ratio": {},
        "info_coeff": 0.0,
    }

# ---------------------------------------------------------------------
# Ranking v3 (unchanged)
# ---------------------------------------------------------------------
def calculate_ranking_score_v3(
    row: dict,
    horizon: str = "1w",
    context: Optional[dict] = None,
    sector_norm_map: Optional[Dict[str, Tuple[float, float]]] = None,
    prev_ranks: Optional[Dict[str, int]] = None,
) -> float:
    context = context or {}
    er = float(row.get("expectedReturnPct", 0.0) or 0.0)
    conf = float(row.get("confidence", 0.0) or 0.0)
    model_score = float(row.get("score", 0.0) or 0.0)
    vol = float(row.get("volatility_10d", 0.05) or 0.05)
    trend = (row.get("trend") or "unknown").lower()
    sector = (row.get("sector") or "").lower()

    sentiment = float(row.get("sentiment_score", 0.0) or 0.0)
    buzz = float(row.get("buzz", 0.0) or 0.0)

    pe = row.get("pe_ratio")
    peg = row.get("peg_ratio")
    div_yield = row.get("dividend_yield")

    drift = float((context.get("drift_score", {}) or {}).get(sector, 0.0) or 0.0)
    realized_acc = float((context.get("hit_ratio", {}) or {}).get(sector, 0.5) or 0.5)
    macro_breadth = float(context.get("macro_breadth", 0.5) or 0.5)
    ic_score = float(context.get("info_coeff", 0.0) or 0.0)

    conf_w = min(1.0, max(0.0, conf)) ** 1.4
    vol_penalty = 1.0 / (1.0 + 10.0 * abs(vol))
    trend_boost = 1.2 if trend == "bullish" else (0.9 if trend == "bearish" else 1.0)

    sentiment_adj = np.tanh(sentiment / 5.0) * 0.05
    buzz_boost = min(buzz / 50.0, 0.05)

    drift_penalty = 1.0 - min(0.5, drift)
    acc_boost = 0.8 + 0.4 * max(0.0, min(1.0, realized_acc))
    ic_boost = 1.0 + np.tanh(ic_score * 5.0) * 0.1
    macro_adj = 1.0 + (macro_breadth - 0.5) * 0.2

    fundamentals = 1.0
    try:
        if pe and peg:
            pe = float(pe)
            peg = float(peg)
            val = max(0.5, min(2.0, peg)) / max(5.0, pe)
            fundamentals = float(np.tanh(val * 2.0) + 1.0)
        elif div_yield:
            dy = float(div_yield)
            fundamentals += min(0.2, max(0.0, dy) / 10.0)
    except Exception:
        pass

    if horizon in ("1d", "1w"):
        weights = dict(er=0.45, conf=0.25, model=0.10, tech=0.10, sent=0.10)
    elif horizon in ("1m",):
        weights = dict(er=0.40, conf=0.25, model=0.15, tech=0.10, sent=0.10)
    else:
        weights = dict(er=0.30, conf=0.20, model=0.15, tech=0.20, sent=0.15)

    er_norm = float(np.tanh(er / 15.0))
    model_norm = float(np.tanh(model_score / 100.0))
    technical = trend_boost * vol_penalty
    sentiment_factor = 1.0 + sentiment_adj + buzz_boost
    health_factor = drift_penalty * acc_boost * ic_boost * macro_adj

    composite = (
        weights["er"]   * er_norm   * conf_w +
        weights["conf"] * conf_w +
        weights["model"]* model_norm +
        weights["tech"] * technical +
        weights["sent"] * sentiment_factor
    ) * health_factor * fundamentals

    if sector_norm_map:
        sec = sector
        if sec in sector_norm_map:
            mean, std = sector_norm_map[sec]
            std = std if std and std > 1e-9 else 1.0
            composite = (composite - mean) / std

    if prev_ranks and isinstance(prev_ranks, dict):
        prev_rank = prev_ranks.get(row.get("ticker", ""))
        if isinstance(prev_rank, int):
            delta = abs(prev_rank - 50) / 50.0
            stability_boost = 1.0 - (0.10 * delta)
            composite *= stability_boost

    # --- Event Boost + Smarter Confidence Adjustment ---
    event_boost = 1.0 + min(0.05, max(-0.05, row.get("event_short_impulse", 0.0)))

    # Base model-derived confidence
    base_conf = float(row.get("confidence", 0.5))

    # Event/news volatility penalty
    buzz = float(row.get("buzz", 0.0))
    event_volatility = (
        abs(row.get("event_short_impulse", 0.0))
        + abs(row.get("event_mid_impulse", 0.0))
        + abs(row.get("event_long_impulse", 0.0))
    )
    news_penalty = min(0.2, (buzz * 0.001) + (event_volatility * 0.3))

    # Combine model confidence with volatility penalty
    adjusted_conf = max(0.0, min(1.0, base_conf * (1.0 - news_penalty)))

    # Weight the final composite score
    composite *= event_boost * (0.75 + 0.5 * adjusted_conf)

    row["explain_factors"] = {
        "expectedReturnPct_norm": round(er_norm, 4),
        "confidence_weight": round(conf_w, 4),
        "model_quality_norm": round(model_norm, 4),
        "technical": round(technical, 4),
        "sentiment_factor": round(sentiment_factor, 4),
        "fundamentals": round(fundamentals, 4),
        "health_factor": round(health_factor, 4),
        "event_boost": round(event_boost, 4),
        "adjusted_conf": round(adjusted_conf, 4),
    }
    return round(float(composite), 5)

# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------
def _build_feature_vector(node: dict, feature_names: List[str]) -> Optional[np.ndarray]:
    """Extract a strictly-ordered vector matching feature_names (fill missing with 0)."""
    node = normalize_keys(node)
    vals: List[float] = []
    for name in feature_names:
        v = node.get(name, 0.0)
        try:
            vals.append(float(v) if v is not None else 0.0)
        except Exception:
            vals.append(0.0)
    return np.array(vals, dtype=float).reshape(1, -1)

# ---------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------
def _load_model(h: str):
    """Load LightGBM model + feature list for horizon h."""
    try:
        import lightgbm as lgb
        dirp = MODELS_DIR / h
        model_path = dirp / "model.txt"
        flist_path = dirp / "feature_list.json"
        if not (model_path.exists() and flist_path.exists()):
            return None, None
        with open(flist_path, "r", encoding="utf-8") as f:
            feature_list = json.load(f)
        booster = lgb.Booster(model_file=str(model_path))
        return booster, feature_list
    except Exception as e:
        log(f"⚠️ load_model[{h}] failed: {e}")
        return None, None

# ---------------------------------------------------------------------
# Public Inference APIs
# ---------------------------------------------------------------------
def predict_for_symbol(symbol: str) -> Dict[str, Any]:
    """Run model inference for a single ticker."""
    sym = (symbol or "").upper().strip()
    if not sym:
        raise ValueError("ticker symbol required")

    node = get_symbol_data(sym) or {}
    node = normalize_keys(node)

    # 🔧 Compute missing technicals from history (dynamic patch)
    if "history" in node and node.get("history"):
        derived = _compute_features_from_history(node["history"])
        if derived:
            for k, v in derived.items():
                if k not in node or node[k] is None:
                    node[k] = v

    # 🔧 One-hot encode sector based on Rolling sector field
    sector_name = (node.get("sector") or "").strip()
    sector_map = {
        "Healthcare": "sector_healthcare",
        "Financials": "sector_financials",
        "Technology": "sector_technology",
        "Industrials": "sector_industrials",
        "Consumer Discretionary": "sector_consumer_discretionary",
        "Communication Services": "sector_communication_services",
        "Materials": "sector_materials",
        "Real Estate": "sector_real_estate",
        "Energy": "sector_energy",
        "Consumer Staples": "sector_consumer_staples",
        "Utilities": "sector_utilities",
        "Financial Services": "sector_financial_services",
        "Consumer Cyclical": "sector_consumer_cyclical",
        "Software": "sector_software",
        "Consumer Defensive": "sector_consumer_defensive",
    }

    # Initialize all known sector dummies as 0
    for col in sector_map.values():
        node[col] = 0.0

    # Activate the one that matches this ticker's sector
    if sector_name in sector_map:
        node[sector_map[sector_name]] = 1.0

    current_price = node.get("close") or node.get("price")
    try:
        current_price = float(current_price) if current_price is not None else None
    except Exception:
        current_price = None

    out: Dict[str, Any] = {"currentPrice": current_price, "predictions": {}}

    for h in HORIZONS:
        booster, features = _load_model(h)
        if booster is None or not features:
            continue

        X = _build_feature_vector(node, features)
        if X is None or X.shape[1] != len(features):
            continue

        try:
            y = booster.predict(X)
            y = float(y[0])
            conf = float(node.get("prediction_confidence", 0.0) or 0.0)
            score = float(y * 100.0)

            if current_price is None:
                continue

            predicted_price = current_price * (1.0 + y)
            expected_return_pct = (predicted_price - current_price) / current_price * 100.0
            context = _load_context_safe_from_node(node)

            row = {
                "ticker": sym,
                "currentPrice": current_price,
                "predictedPrice": predicted_price,
                "expectedReturnPct": expected_return_pct,
                "confidence": conf,
                "score": score,
                "trend": node.get("trend") or "unknown",
                "sector": node.get("sector"),
                "volatility_10d": node.get("volatility_10d", 0.03),
                "sentiment_score": node.get("sentiment_score", 0.0),
                "buzz": node.get("buzz", 0.0),
                "pe_ratio": node.get("pe_ratio"),
                "peg_ratio": node.get("peg_ratio"),
                "dividend_yield": node.get("dividend_yield"),
            }
            ranking_score = calculate_ranking_score_v3(row, h, context=context)
            out["predictions"][h] = {
                "predictedPrice": round(predicted_price, 4),
                "expectedReturnPct": round(expected_return_pct, 4),
                "confidence": round(conf, 4),
                "score": round(score, 3),
                "rankingScore": float(ranking_score),
                "reason": "lgbm",
            }
        except Exception as e:
            log(f"⚠️ model_predict failed for {sym} ({h}): {e}")
            continue

    return out

# ---------------------------------------------------------------------
# score_all_tickers (unchanged)
# ---------------------------------------------------------------------
def score_all_tickers(limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Batch predict across rolling with dynamic progress bar.
    Only symbols with valid model predictions are returned for each horizon.
    """
    rolling = _read_rolling() or {}
    syms = list(rolling.keys())
    if isinstance(limit, int) and limit > 0:
        syms = syms[:limit]

    results: Dict[str, Any] = {}
    total = len(syms)
    if total == 0:
        log("⚠️ No symbols available for scoring.")
        return results

    log(f"⚙️ Starting AI model predictions across {total:,} symbols...")
    start = time.time()

    # ✅ Dynamic loading bar
    with tqdm(total=total, ncols=80, desc="Predicting", unit="tick", ascii=True) as bar:
        for i, sym in enumerate(syms, 1):
            try:
                res = predict_for_symbol(sym)
                if res.get("predictions"):
                    results[sym] = res
            except Exception as e:
                log(f"⚠️ Prediction failed for {sym}: {e}")
            finally:
                bar.update(1)

    dur = time.time() - start
    rate = total / max(dur, 1)
    log(f"✅ Predictions complete — {len(results):,}/{total:,} symbols scored in {dur:.1f}s ({rate:.1f}/s)")
    return results

# ---------------------------------------------------------------------
# Standalone test entry (unchanged)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from backend.data_pipeline import _read_rolling, log

    log("🔍 Running standalone AI model test...")
    rolling = _read_rolling() or {}
    log(f"📦 Loaded {len(rolling)} tickers from rolling.json.gz")

    # Run predictions for a limited sample to test quickly
    preds = score_all_tickers(limit=50)
    log(f"✅ Generated predictions for {len(preds)} tickers.")

    # Optionally show a preview of the first few
    sample_items = list(preds.items())[:3]
    for sym, res in sample_items:
        print(f"\n🔹 {sym}")
        for h, p in res.get("predictions", {}).items():
            print(f"   {h}: price={p.get('predictedPrice')} "
                  f"→ return={p.get('expectedReturnPct')}%, "
                  f"conf={p.get('confidence')}")
