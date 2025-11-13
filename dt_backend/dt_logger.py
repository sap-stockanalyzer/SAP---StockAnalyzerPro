# dt_backend/dt_logger.py — v1.0
# Wrapper for unified logging across DT modules with a [DT] prefix.

from __future__ import annotations
import sys, os
from datetime import datetime

# fallback if backend log() isn't available
def _fallback(msg: str):
    ts = datetime.utcnow().strftime("%H:%M:%S")
    print(f"[DT][{ts}] {msg}", flush=True)

try:
    from dt_backend.data_pipeline_dt import log as _core_log  # type: ignore
    def dt_log(msg: str):
        _core_log(f"[DT] {msg}")
except Exception:
    dt_log = _fallback
