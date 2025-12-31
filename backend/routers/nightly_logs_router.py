# backend/routers/nightly_logs_router.py
"""Nightly / Scheduler log browser API.

The UI needs to:
  • List available log files (scheduled + manual runs).
  • Read the full content of a selected log.

This router is intentionally file-based and best-effort. It should never
crash the backend due to a missing folder or a single unreadable file.

Security posture:
  • We only allow reading files from a small allowlist of directories derived
    from unified PATHS (repo-root config.py).
  • The read endpoint accepts an opaque id (base64url of absolute path). That
    path is re-validated to be inside an allowed directory before reading.
"""

from __future__ import annotations

import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query

from backend.core.config import PATHS

router = APIRouter(prefix="/api/logs", tags=["logs"])


def _allowed_roots() -> List[Path]:
    roots: List[Path] = []
    for key in ("nightly_logs", "scheduler_logs", "backend_logs", "logs"):
        try:
            p = PATHS.get(key)
            if p:
                roots.append(Path(p))
        except Exception:
            continue

    # Legacy location (some deployments still redirect stdout here)
    try:
        root = Path(PATHS.get("root") or Path.cwd())
        roots.append(root / "backend" / "jobs" / "logs")
    except Exception:
        pass

    # De-dup + keep existing only
    out: List[Path] = []
    seen = set()
    for r in roots:
        try:
            rr = r.resolve()
        except Exception:
            rr = r
        if str(rr) in seen:
            continue
        seen.add(str(rr))
        out.append(rr)
    return out


def _is_within(path: Path, root: Path) -> bool:
    try:
        path = path.resolve()
        root = root.resolve()
        return path == root or root in path.parents
    except Exception:
        return False


def _encode_id(p: Path) -> str:
    raw = str(p).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_id(id_: str) -> Path:
    pad = "=" * ((4 - (len(id_) % 4)) % 4)
    raw = base64.urlsafe_b64decode((id_ + pad).encode("ascii"))
    return Path(raw.decode("utf-8"))


def _candidate_files(scope: str) -> List[Tuple[str, Path]]:
    """Return [(kind, path), ...] candidates."""
    roots = _allowed_roots()
    nightly_root = Path(PATHS.get("nightly_logs") or Path(PATHS.get("logs") or "logs") / "nightly")
    scheduler_root = Path(PATHS.get("scheduler_logs") or Path(PATHS.get("logs") or "logs") / "scheduler")
    backend_root = Path(PATHS.get("backend_logs") or Path(PATHS.get("logs") or "logs") / "backend")
    logs_root = Path(PATHS.get("logs") or "logs")

    # Normalize scope
    scope = (scope or "nightly").strip().lower()

    picks: List[Tuple[str, Path]] = []

    def add_dir(kind: str, d: Path, recursive: bool = False) -> None:
        try:
            if not d.exists() or not d.is_dir():
                return
            globber = d.rglob if recursive else d.glob
            for ext in ("*.log", "*.txt", "*.out", "*.json"):
                for f in globber(ext):
                    if f.is_file():
                        picks.append((kind, f))
        except Exception:
            return

    if scope in ("nightly", "all"):
        add_dir("nightly", nightly_root, recursive=True)
        # Some schedulers log nightly runs under scheduler logs
        add_dir("scheduler", scheduler_root, recursive=True)
        # ...and some jobs still write to a legacy folder
        try:
            root = Path(PATHS.get("root") or Path.cwd())
            add_dir("legacy", root / "backend" / "jobs" / "logs", recursive=True)
        except Exception:
            pass

        # Many deployments still write nightly output into the root daily log
        # (logs/YYYY-MM-DD.log). Include those here so the page is never empty.
        add_dir("daily", logs_root, recursive=False)

    if scope in ("backend", "all"):
        add_dir("backend", backend_root, recursive=True)

    if scope in ("daily", "all"):
        # Daily rolling log produced by utils.logger
        add_dir("daily", logs_root, recursive=False)

    # Final safety: filter to allowed roots only
    allowed = roots
    out: List[Tuple[str, Path]] = []
    for kind, f in picks:
        if any(_is_within(f, r) for r in allowed):
            out.append((kind, f))
    return out


@router.get("/nightly/runs")
def list_nightly_runs(scope: str = Query(default="nightly")) -> Dict[str, Any]:
    """List log files that are relevant for nightly debugging.

    scope:
      - nightly (default): nightly_logs + scheduler_logs + legacy job logs
      - daily: root daily logs only
      - backend: backend_logs only
      - all: everything in allowed roots
    """
    files = _candidate_files(scope)
    runs: List[Dict[str, Any]] = []
    for kind, p in files:
        try:
            stat = p.stat()
            runs.append(
                {
                    "id": _encode_id(p),
                    "name": p.name,
                    "kind": kind,
                    "size_bytes": int(stat.st_size),
                    "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "rel": str(p),
                }
            )
        except Exception:
            continue

    # newest first
    def _key(x: Dict[str, Any]) -> float:
        try:
            return datetime.fromisoformat(x.get("mtime") or "1970-01-01").timestamp()
        except Exception:
            return 0.0

    runs.sort(key=_key, reverse=True)
    return {"scope": scope, "count": len(runs), "runs": runs}


@router.get("/nightly/run/{run_id}")
def read_nightly_run(
    run_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=1_000_000, ge=1, le=10_000_000),
) -> Dict[str, Any]:
    """Read a log file by id.

    Returns up to `limit` bytes starting at `offset`.
    The UI can request additional chunks if `truncated` is true.
    """
    try:
        p = _decode_id(run_id)
    except Exception:
        raise HTTPException(status_code=400, detail="bad_id")

    allowed = _allowed_roots()
    if not any(_is_within(p, r) for r in allowed):
        raise HTTPException(status_code=403, detail="forbidden")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="not_found")

    try:
        size = p.stat().st_size
        with p.open("rb") as f:
            f.seek(int(offset))
            data = f.read(int(limit))
        text = data.decode("utf-8", errors="replace")
        next_offset = offset + len(data)
        truncated = next_offset < size
        return {
            "id": run_id,
            "name": p.name,
            "path": str(p),
            "size_bytes": int(size),
            "offset": int(offset),
            "limit": int(limit),
            "next_offset": int(next_offset) if truncated else None,
            "truncated": bool(truncated),
            "content": text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"read_failed: {type(e).__name__}")
