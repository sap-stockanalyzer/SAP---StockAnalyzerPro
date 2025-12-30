# dt_backend/jobs/daytrading_job.py — v3.1 (SINGLE-ROLLING + SAFE DEFAULT LOCK)
"""Main intraday trading loop for AION dt_backend.

This job wires together the intraday pipeline:

    rolling → context_dt → features_dt → predictions_dt
            → regime → policy_dt → execution_dt → (optional) broker execution

Architecture (Linux server)
---------------------------
We use a **single** rolling cache file (DT_PATHS['rolling_intraday_file']).

* Live bars are a bounded sliding window and are expected to overwrite.
* Policy/execution/learning state is written into the same rolling.
* Long-lived learning should live in a separate dt_brain artifact (see
  dt_backend/core/dt_brain.py), not in rolling.

Safety notes
------------
- We do NOT embed model-training code in this job file.
- Rolling writes are atomic, but atomic replace does not prevent "lost updates"
  when multiple processes read-modify-write at the same time. So we default the
  rolling lock ON for this process unless the environment explicitly disables it.

    DT_USE_LOCK=1   (default here)
    DT_USE_LOCK=0   (disable if you truly know it's single-writer)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from dt_backend.core import (
    log,
    build_intraday_context,
    classify_intraday_regime,
    apply_intraday_policy,
)
from dt_backend.core.execution_dt import run_execution_intraday
from dt_backend.engines.feature_engineering import build_intraday_features
from dt_backend.ml import score_intraday_tickers, build_intraday_signals
from dt_backend.engines.trade_executor import ExecutionConfig, execute_from_policy


def run_daytrading_cycle(
    execute: bool = False,
    max_symbols: Optional[int] = None,
    max_positions: int = 50,
    execution_cfg: ExecutionConfig | None = None,
) -> Dict[str, Any]:
    """Run one full intraday cycle."""
    log("[daytrading_job] 🚀 starting intraday cycle")

    # Single rolling file: disallow per-process split unless explicitly set elsewhere.
    os.environ.pop("DT_ROLLING_PATH", None)

    # Default rolling lock ON for this job (prevents read-modify-write stomps across processes).
    os.environ.setdefault("DT_USE_LOCK", (os.getenv("DT_USE_LOCK") or "1"))

    ctx_summary = build_intraday_context()
    feat_summary = build_intraday_features(max_symbols=max_symbols)
    score_summary = score_intraday_tickers(max_symbols=max_symbols)
    regime_summary = classify_intraday_regime()
    policy_summary = apply_intraday_policy(max_positions=max_positions)

    exec_dt_summary = run_execution_intraday()
    signals_summary = build_intraday_signals()

    exec_summary: Dict[str, Any] | None = None
    if execute:
        exec_summary = execute_from_policy(execution_cfg)

    log("[daytrading_job] ✅ intraday cycle complete")
    return {
        "context": ctx_summary,
        "features": feat_summary,
        "scoring": score_summary,
        "regime": regime_summary,
        "policy": policy_summary,
        "execution_dt": exec_dt_summary,
        "signals": signals_summary,
        "execution": exec_summary,
    }


def main() -> None:
    run_daytrading_cycle(execute=False)


if __name__ == "__main__":
    main()
