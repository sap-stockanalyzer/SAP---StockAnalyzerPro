# verify_dataset_features.py
# Quick post-build check for TA coverage and targets
import pandas as pd

df = pd.read_parquet("ml_data/training_data_daily.parquet")
print("Rows:", len(df))
cols = ["rsi_14","macd","volatility_10d","momentum_5d","close_lag1"]
print("Non-null TA counts:", {c:int(df[c].notna().sum()) for c in cols if c in df.columns})
print("Targets present:", [c for c in df.columns if c.startswith("target_")][:12])
print(df[["symbol","date","rsi_14","volatility_10d","momentum_5d"]].head(10))
