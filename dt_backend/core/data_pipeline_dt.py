# dt_backend/core/data_pipeline_dt.py — v1.2
"""
Lightweight I/O helpers for dt_backend intraday engine.

Guarantees:
  • never raises in normal use (best-effort)
  • atomic writes for rolling cache
  • optional lock to prevent cross-process read-modify-write stomps
  • stable node schema via ensure_symbol_node

Locking
-------
Atomic rename prevents gzip corruption, but it does NOT prevent lost updates when
multiple processes do: read → modify → save.

So we support a simple lock file:
  - Enabled when DT_USE_LOCK is truthy ("1", "true", "yes", "on").
  - Default is ON unless you explicitly set DT_USE_LOCK=0.

Stale-lock handling:
  - If the lock file exists but the recorded PID is not alive, we remove it.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config_dt import DT_PATHS
from .logger_dt import log


def _rolling_path() -> Path:
    override = os.getenv("DT_ROLLING_PATH", "").strip()
    if override:
        return Path(override)
    return Path(DT_PATHS["rolling_intraday_file"])


def _lock_path() -> Path:
    override = os.getenv("DT_LOCK_PATH", "").strip()
    if override:
        return Path(override)
    return Path(
        DT_PATHS.get("rolling_dt_lock_file")
        or (Path(DT_PATHS["rolling_intraday_dir"]) / ".rolling_intraday_dt.lock")
    )


def _should_lock() -> bool:
    # Default ON (safer for multi-process schedulers). Set DT_USE_LOCK=0 to disable.
    return str(os.getenv("DT_USE_LOCK", "1")).strip().lower() in ("1", "true", "yes", "y", "on")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return True


def _read_lock_pid(lock: Path) -> int:
    try:
        raw = lock.read_text(encoding="utf-8", errors="ignore").strip()
        parts = raw.split()
        return int(parts[0]) if parts else -1
    except Exception:
        return -1


def _acquire_lock(timeout_s: float = 30.0) -> bool:
    if not _should_lock():
        return True

    lock = _lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + max(0.0, float(timeout_s))

    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                payload = f"{os.getpid()} {time.time():.3f}"
                os.write(fd, payload.encode("utf-8", errors="ignore"))
            finally:
                os.close(fd)
            return True

        except FileExistsError:
            pid = _read_lock_pid(lock)
            if pid > 0 and not _pid_alive(pid):
                try:
                    lock.unlink(missing_ok=True)  # type: ignore[arg-type]
                    continue
                except Exception:
                    pass

            if time.time() >= deadline:
                return False
            time.sleep(0.15)

        except Exception:
            return False


def _release_lock() -> None:
    if not _should_lock():
        return
    try:
        _lock_path().unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass


def _read_rolling() -> Dict[str, Any]:
    path = _rolling_path()
    if not path.exists():
        return {}

    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        log(f"⚠️ failed to read rolling cache {path}: {e}")
        return {}


def save_rolling(rolling: Dict[str, Any]) -> None:
    path = _rolling_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    try:
        if not _acquire_lock(timeout_s=float(os.getenv("DT_LOCK_TIMEOUT", "60"))):
            log(f"⚠️ rolling lock timeout; skipping save: {path}")
            return

        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(rolling or {}, f, ensure_ascii=False, indent=2)

        tmp.replace(path)

    except Exception as e:
        log(f"⚠️ failed to save rolling cache {path}: {e}")
    finally:
        _release_lock()


def _norm_sym(sym: str) -> str:
    return (sym or "").strip().upper()


def load_universe() -> List[str]:
    path = Path(DT_PATHS["universe_file"])
    if not path.exists():
        log(f"⚠️ universe file missing at {path} — using empty universe.")
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"⚠️ failed to parse universe file {path}: {e}")
        return []

    if isinstance(raw, dict) and "symbols" in raw:
        items: Iterable[str] = raw.get("symbols", [])
    elif isinstance(raw, list):
        items = raw
    else:
        log(f"⚠️ unexpected universe schema in {path}, expected list or dict['symbols'].")
        return []

    out: List[str] = []
    seen = set()
    for item in items:
        s = _norm_sym(str(item))
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def ensure_symbol_node(rolling: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    sym = _norm_sym(symbol)
    node = rolling.get(sym)
    if not isinstance(node, dict):
        node = {}

    node.setdefault("bars_intraday", [])
    node.setdefault("features_dt", {})
    node.setdefault("predictions_dt", {})
    node.setdefault("context_dt", {})
    node.setdefault("policy_dt", {})

    rolling[sym] = node
    return node
