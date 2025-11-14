"""Simple intraday backtesting harness for the DT pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json

import pandas as pd

from dt_backend.config_dt import DT_PATHS

DATASET_PATH = DT_PATHS["dtml_data"] / "training_data_intraday.parquet"
LOG_PATH = DT_PATHS["dtlogs"] / "backtest_runs.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_dataset(dataset_path: Path | None = None) -> pd.DataFrame:
    path = Path(dataset_path or DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp"])


def run_intraday_backtest(dataset_path: Path | None = None, *, threshold: float = 0.0) -> Dict[str, Any]:
    """Compute naive PnL by following BUY signals and holding for the label horizon."""
    df = _load_dataset(dataset_path)
    df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    trades = df[df["target_label_15m"].str.upper() == "BUY"].copy()
    trades = trades[trades["target_ret_15m"].notna()]
    trades = trades[trades.get("momentum_score", 0).fillna(0) >= threshold]

    pnl = trades["target_ret_15m"].sum()
    win_rate = (trades["target_ret_15m"] > 0).mean() if not trades.empty else 0.0

    result = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "trades": int(len(trades)),
        "pnl": float(pnl),
        "win_rate": float(win_rate) if trades.shape[0] else 0.0,
        "dataset": str(dataset_path or DATASET_PATH),
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    return result


if __name__ == "__main__":  # pragma: no cover
    print(run_intraday_backtest())
