"""
ml_data_builder.py — v2.3 (Rolling-Native + Unified Config)
Author: AION Analytics / StockAnalyzerPro

Purpose:
• Build ML-ready datasets directly from rolling.json.gz
• Derive technical, momentum, volatility & fundamental features
• Integrate sector (and one-hot encode it) for model training
• Normalize all field names (snake_case)
• Output Parquet datasets only (no large JSON)
• ✅ Uses config.PATHS for all read/write locations
"""

import os, pandas as pd, numpy as np
from datetime import datetime
from typing import Dict, Any, List
from .config import PATHS  # ✅ unified path import

from .data_pipeline import (
    _read_rolling,
    ensure_symbol_fields,
    log,
)

# ------------------- Normalization Map -------------------
NORMALIZE_KEYS = {
    "peRatio": "pe_ratio", "pbRatio": "pb_ratio", "psRatio": "ps_ratio",
    "pegRatio": "peg_ratio", "debtEquity": "debt_equity", "debtEbitda": "debt_ebitda",
    "revenueGrowth": "revenue_growth", "epsGrowth": "eps_growth",
    "profitMargin": "profit_margin", "operatingMargin": "operating_margin",
    "grossMargin": "gross_margin", "dividendYield": "dividend_yield",
    "payoutRatio": "payout_ratio", "marketCap": "marketCap"
}


def normalize_keys(node: dict) -> dict:
    """Convert any camelCase keys in a node to snake_case for Rolling consistency."""
    if not isinstance(node, dict):
        return node
    for old, new in NORMALIZE_KEYS.items():
        if old in node and new not in node:
            node[new] = node.pop(old)
    return node


ML_DATA_DIR = PATHS["ml_data"]  # ✅ unified config path
os.makedirs(ML_DATA_DIR, exist_ok=True)


# -------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------
def _compute_features(hist: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute technical/momentum features from a ticker’s history."""
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
            "close": float(latest["close"]),
            "volume": float(latest["volume"]),
            "volatility_10d": float(latest["volatility_10d"]),
            "momentum_5d": float(latest["momentum_5d"]),
            "ret1": float(latest["ret1"]),
            "ret5": float(latest["ret5"]),
            "ret10": float(latest["ret10"]),
        }
    except Exception:
        return {}


# -------------------------------------------------------------
# Core Builder
# -------------------------------------------------------------
def build_ml_dataset(horizon: str = "daily") -> pd.DataFrame:
    """
    Generate a fully enriched dataset from Rolling.
    Returns a DataFrame and saves Parquet only.
    """
    log(f"[ml_data_builder] 🚀 Building ML dataset from Rolling (horizon={horizon})...")
    rolling = _read_rolling()
    if not rolling:
        log("⚠️ No rolling data found.")
        return pd.DataFrame()

    rows = []
    for sym, node in rolling.items():
        node = normalize_keys(node)  # ✅ ensure snake_case per symbol
        hist = node.get("history") or []
        if len(hist) < 30:
            node = ensure_symbol_fields(sym)
            node = normalize_keys(node)
            hist = node.get("history") or []
            if not hist:
                continue

        feats = _compute_features(hist)
        if not feats:
            continue

        enriched = {
            "symbol": sym,
            "sector": node.get("sector"),
            "marketCap": node.get("marketCap"),
            "pe_ratio": node.get("pe_ratio"),
            "pb_ratio": node.get("pb_ratio"),
            "ps_ratio": node.get("ps_ratio"),
            "beta": node.get("beta"),
            "rsi": node.get("rsi") or node.get("rsi_14"),
            "ma50": node.get("ma50"),
            "ma200": node.get("ma200"),
            "sentiment_score": node.get("sentiment_score"),
            "shares_oustanding": node.get("shares_oustanding"),
            "public_float": node.get("public_float"),
            "float_ratio": (node.get("public_float") or 0) / max((node.get("shares_oustanding") or 1), 1),
            "cap_per_share": (node.get("marketCap") or 0) / max((node.get("shares_oustanding") or 1), 1),
            **feats,
        }
        rows.append(enriched)

    df = pd.DataFrame(rows).dropna(subset=["close"])
    if df.empty:
        log("⚠️ No valid data to build ML dataset.")
        return df

    # ------------------- Sector One-Hot Encoding -------------------
    if "sector" in df.columns and df["sector"].notna().any():
        sector_dummies = pd.get_dummies(df["sector"], prefix="sector", dtype=float)
        df = pd.concat([df.drop(columns=["sector"]), sector_dummies], axis=1)
        log(f"✅ One-hot encoded {sector_dummies.shape[1]} sector features")
    else:
        log("ℹ️ No sector info found to encode")

    # ------------------- Target Engineering -------------------
    df["target_1w"] = df["ret5"]
    df["target_2w"] = df["ret10"]
    df["target_4w"] = df["momentum_5d"]
    df["target_52w"] = df["momentum_5d"] * 10

    # ------------------- Save Outputs -------------------
    parquet_path = ML_DATA_DIR / f"training_data_{horizon}.parquet"  # ✅ unified path
    try:
        df.to_parquet(parquet_path, index=False)
        log(f"✅ ML dataset built — {len(df)} rows → {parquet_path}")
    except Exception as e:
        log(f"⚠️ Failed to write parquet: {e}")

    return df


if __name__ == "__main__":
    build_ml_dataset("daily")
    build_ml_dataset("weekly")
