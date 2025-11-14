import numpy as np
import pandas as pd

def simple_long_only(preds: pd.Series, rets: pd.Series, top_q=0.2):
    thr = np.quantile(preds.dropna(), 1-top_q)
    mask = preds >= thr
    strat = (rets * mask.astype(float)).fillna(0.0)
    cum = (1+strat).cumprod()
    ann_ret = cum.iloc[-1]**(252/len(cum)) - 1 if len(cum)>0 else 0
    ann_vol = strat.std() * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-9)
    return {"ann_return": float(ann_ret), "ann_vol": float(ann_vol), "sharpe": float(sharpe)}
