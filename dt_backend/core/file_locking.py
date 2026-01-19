"""dt_backend/core/file_locking.py — v1.0

File locking utilities for atomic writes and preventing race conditions.

Implements a saga pattern foundation with proper file locking to ensure:
- No overlapping writes to state files
- Atomic updates (all-or-nothing)
- Deadlock prevention through consistent lock ordering
- Support for both exclusive and shared locks (future)

Lock Ordering (to prevent deadlocks):
  1. positions_dt.json
  2. dt_trades.jsonl
  3. dt_execution_ledger.jsonl
  4. dt_state.json
  5. dt_metrics.json

Usage:
    with AcquireLock("/path/to/file.json", timeout=5.0) as acquired:
        if acquired:
            # safe to write
            pass
"""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional, Union

from dt_backend.core.logger_dt import log, debug

PathLike = Union[str, Path]


@dataclass
class FileLock:
    """Represents a file lock with its metadata."""
    
    path: Path
    fd: Optional[int] = None
    acquired: bool = False
    lock_type: str = "exclusive"  # exclusive or shared
    
    def __post_init__(self):
        """Ensure parent directory exists."""
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)


def _safe_path(p: PathLike) -> Path:
    """Convert path-like to Path, creating parent dirs."""
    try:
        path = Path(p) if not isinstance(p, Path) else p
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        log(f"[file_locking] ⚠️ Failed to create path {p}: {e}")
        raise


@contextmanager
def AcquireLock(
    file_path: PathLike,
    *,
    timeout: float = 5.0,
    shared: bool = False,
) -> Generator[bool, None, None]:
    """Context manager for acquiring file locks.
    
    Implements POSIX advisory locks using fcntl. These locks are:
    - Advisory (cooperating processes must use the same locking mechanism)
    - Process-level (released automatically when process exits)
    - Compatible with NFS (unlike flock)
    
    Args:
        file_path: Path to the file to lock
        timeout: Maximum seconds to wait for lock acquisition
        shared: If True, acquire shared (read) lock; if False, acquire exclusive (write) lock
    
    Yields:
        bool: True if lock was acquired, False otherwise
        
    Example:
        with AcquireLock("/path/to/file.json", timeout=5.0) as acquired:
            if acquired:
                # Critical section - safe to write
                with open("/path/to/file.json", "w") as f:
                    json.dump(data, f)
            else:
                log("Failed to acquire lock")
    """
    lock = FileLock(path=_safe_path(file_path), lock_type="shared" if shared else "exclusive")
    fd = None
    
    try:
        # Open file for locking (create if doesn't exist)
        fd = os.open(str(lock.path), os.O_RDWR | os.O_CREAT, 0o644)
        lock.fd = fd
        
        # Determine lock type
        lock_flag = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        
        # Try to acquire lock with timeout
        deadline = time.time() + max(0.0, float(timeout))
        acquired = False
        
        while time.time() < deadline:
            try:
                # Try non-blocking lock
                fcntl.flock(fd, lock_flag | fcntl.LOCK_NB)
                acquired = True
                lock.acquired = True
                break
            except (IOError, OSError) as e:
                # Lock is held by another process
                if e.errno not in (11, 35):  # EAGAIN, EWOULDBLOCK
                    log(f"[file_locking] ⚠️ Unexpected error acquiring lock for {lock.path}: {e}")
                    break
                # Wait a bit and retry
                time.sleep(0.05)
        
        if acquired:
            debug(f"[file_locking] ✅ Acquired {lock.lock_type} lock: {lock.path.name}")
        else:
            log(f"[file_locking] ⚠️ Timeout acquiring {lock.lock_type} lock: {lock.path.name}")
        
        yield acquired
        
    except Exception as e:
        log(f"[file_locking] ⚠️ Error in lock context for {lock.path}: {e}")
        yield False
        
    finally:
        # Release lock and close file descriptor
        if fd is not None:
            try:
                if lock.acquired:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    debug(f"[file_locking] 🔓 Released lock: {lock.path.name}")
                os.close(fd)
            except Exception as e:
                log(f"[file_locking] ⚠️ Error releasing lock for {lock.path}: {e}")


def WriteLocked(file_path: PathLike, data: str, *, timeout: float = 5.0) -> bool:
    """Write data to file with exclusive lock.
    
    Atomically writes data to file using:
    1. Acquire exclusive lock
    2. Write to temporary file
    3. Atomic rename (replace)
    4. Release lock
    
    Args:
        file_path: Path to target file
        data: String data to write
        timeout: Lock timeout in seconds
        
    Returns:
        bool: True if write succeeded, False otherwise
    """
    path = _safe_path(file_path)
    
    with AcquireLock(path, timeout=timeout) as acquired:
        if not acquired:
            log(f"[file_locking] ⚠️ Failed to acquire lock for write: {path.name}")
            return False
        
        try:
            # Write to temp file first (atomic)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(data, encoding="utf-8")
            # Atomic rename
            tmp.replace(path)
            return True
            
        except Exception as e:
            log(f"[file_locking] ⚠️ Failed to write {path.name}: {e}")
            return False


def AppendLocked(file_path: PathLike, line: str, *, timeout: float = 5.0) -> bool:
    """Append line to file with exclusive lock.
    
    Atomically appends a line to a file (typically .jsonl) using:
    1. Acquire exclusive lock
    2. Append line
    3. Flush to disk
    4. Release lock
    
    Args:
        file_path: Path to target file
        line: Line to append (newline added automatically if missing)
        timeout: Lock timeout in seconds
        
    Returns:
        bool: True if append succeeded, False otherwise
    """
    path = _safe_path(file_path)
    
    with AcquireLock(path, timeout=timeout) as acquired:
        if not acquired:
            log(f"[file_locking] ⚠️ Failed to acquire lock for append: {path.name}")
            return False
        
        try:
            # Ensure line ends with newline
            if not line.endswith("\n"):
                line = line + "\n"
            
            # Append with explicit flush
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())  # Force OS to write to disk
            
            return True
            
        except Exception as e:
            log(f"[file_locking] ⚠️ Failed to append to {path.name}: {e}")
            return False


def ReadLocked(file_path: PathLike, *, timeout: float = 5.0, default: Any = None) -> Any:
    """Read file content with shared lock.
    
    Acquires a shared (read) lock to ensure consistent reads during writes.
    
    Args:
        file_path: Path to file
        timeout: Lock timeout in seconds
        default: Value to return if file doesn't exist or read fails
        
    Returns:
        File content as string, or default value
    """
    path = _safe_path(file_path)
    
    if not path.exists():
        return default
    
    with AcquireLock(path, timeout=timeout, shared=True) as acquired:
        if not acquired:
            log(f"[file_locking] ⚠️ Failed to acquire shared lock for read: {path.name}")
            return default
        
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            log(f"[file_locking] ⚠️ Failed to read {path.name}: {e}")
            return default


@contextmanager
def AcquireMultipleLocks(
    file_paths: list[PathLike],
    *,
    timeout: float = 5.0,
) -> Generator[bool, None, None]:
    """Acquire multiple locks in order to prevent deadlocks.
    
    Always acquires locks in the same order based on lock ordering convention:
      1. positions_dt.json
      2. dt_trades.jsonl
      3. dt_execution_ledger.jsonl
      4. dt_state.json
      5. dt_metrics.json
    
    Args:
        file_paths: List of file paths to lock
        timeout: Total timeout for acquiring all locks
        
    Yields:
        bool: True if all locks were acquired, False otherwise
        
    Example:
        files = [positions_file, trades_file, ledger_file]
        with AcquireMultipleLocks(files, timeout=10.0) as acquired:
            if acquired:
                # All files locked, safe to update
                update_positions()
                append_trade()
                update_ledger()
    """
    # Define lock priority order (lower = acquire first)
    LOCK_ORDER = {
        "dt_positions.json": 1,
        "positions_dt.json": 1,
        "dt_trades.jsonl": 2,
        "dt_execution_ledger.jsonl": 3,
        "dt_state.json": 4,
        "dt_metrics.json": 5,
    }
    
    # Sort paths by lock order
    paths = [_safe_path(p) for p in file_paths]
    paths.sort(key=lambda p: LOCK_ORDER.get(p.name, 999))
    
    # Track acquired locks for cleanup
    acquired_fds = []
    all_acquired = False
    
    try:
        # Acquire each lock in order
        deadline = time.time() + max(0.0, float(timeout))
        
        for path in paths:
            remaining = deadline - time.time()
            if remaining <= 0:
                log(f"[file_locking] ⚠️ Timeout acquiring multiple locks")
                break
            
            fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
            
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired_fds.append((path, fd))
            except (IOError, OSError):
                # Failed to acquire, release all and fail
                os.close(fd)
                log(f"[file_locking] ⚠️ Failed to acquire lock: {path.name}")
                break
        
        # Check if we got all locks
        all_acquired = len(acquired_fds) == len(paths)
        
        if all_acquired:
            debug(f"[file_locking] ✅ Acquired {len(acquired_fds)} locks")
        
        yield all_acquired
        
    except Exception as e:
        log(f"[file_locking] ⚠️ Error in multi-lock context: {e}")
        yield False
        
    finally:
        # Release all acquired locks in reverse order
        for path, fd in reversed(acquired_fds):
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                debug(f"[file_locking] 🔓 Released lock: {path.name}")
            except Exception as e:
                log(f"[file_locking] ⚠️ Error releasing lock {path.name}: {e}")
