# utils/live_log.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "backend_live.log"
MAX_AGE_SECONDS = 24 * 60 * 60  # 24 hours


def ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def append_log(line: str):
    ensure_log_dir()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {line.rstrip()}\n")


def prune_old_logs():
    if not LOG_FILE.exists():
        return

    cutoff = time.time() - MAX_AGE_SECONDS
    kept: List[str] = []

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ts_str = line.split("]")[0].strip("[")
                ts = time.mktime(time.strptime(ts_str, "%Y-%m-%d %H:%M:%S"))
                if ts >= cutoff:
                    kept.append(line)
            except Exception:
                # Keep malformed lines
                kept.append(line)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(kept)


def tail_lines(limit: int = 300) -> List[str]:
    if not LOG_FILE.exists():
        return []

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return f.readlines()[-limit:]
