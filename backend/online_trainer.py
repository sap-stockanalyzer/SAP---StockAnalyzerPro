from __future__ import annotations
import os, glob, json
from typing import List, Optional
from datetime import datetime
try:
    import pandas as pd
except Exception:
    pd=None

MODEL_DIR=os.environ.get("MODEL_DIR","ml_data/models")
OUTC_DIR=os.environ.get("OUTCOMES_DIR","ml_data/prediction_outcomes")
REGISTRY=os.environ.get("MODEL_REGISTRY","ml_data/model_registry.jsonl")
os.makedirs(MODEL_DIR, exist_ok=True)

def _read_all_outcomes():
    files=sorted(glob.glob(os.path.join(OUTC_DIR,"outcomes_*.parquet")))
    if not files or pd is None: 
        return None if pd is None else pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).dropna(subset=["y_true","label"])

def train_incremental(feature_cols: List[str], base_model_path: Optional[str]=None):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score
    df=_read_all_outcomes()
    if df is None or df.empty: return {"status":"no_outcomes"}
    X_cols=[c for c in feature_cols if c in df.columns]
    if not X_cols: X_cols=[c for c in ["y_pred","proba"] if c in df.columns]
    if not X_cols: return {"status":"no_features"}
    y=df["label"].astype(int).values; X=df[X_cols].values
    clf=LogisticRegression(max_iter=1000); clf.fit(X,y)
    acc=float((clf.predict(X)==y).mean())
    try: auc=float(roc_auc_score(y, clf.predict_proba(X)[:,1]))
    except Exception: auc=None
    ts=datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"); path=os.path.join(MODEL_DIR,f"online_meta_clf_{ts}.joblib")
    joblib.dump({"model":clf,"features":X_cols}, path)
    with open(REGISTRY,"a",encoding="utf-8") as f: f.write(json.dumps({"ts":ts,"artifact":path,"metrics":{"acc":acc,"auc":auc,"n":int(len(df))}})+"\n")
    return {"status":"ok","path":path,"acc":acc,"auc":auc,"n":int(len(df))}
