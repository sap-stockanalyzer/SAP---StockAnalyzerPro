"""backend/services/supabase_sync.py

Manual artifact sync to Supabase Storage.

Design goals:
  - No heavy dependencies (uses urllib).
  - Upload-only (no deletes).
  - Safe-by-default with per-file status reporting.

Environment (from admin_keys.py):
  - SUPABASE_URL
  - SUPABASE_SERVICE_ROLE_KEY  (recommended for server-side uploads)
  - SUPABASE_BUCKET (default: "aion")
"""

from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from admin_keys import SUPABASE_BUCKET, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL
from config import ROOT, PATHS


@dataclass
class SyncResult:
    uploaded: List[str]
    skipped: List[str]
    missing: List[str]
    errors: List[Dict[str, str]]


def supabase_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)


def _guess_content_type(path: Path) -> str:
    # json.gz should still be sent as application/gzip for clarity.
    if path.name.endswith(".json.gz") or path.suffix == ".gz":
        return "application/gzip"
    if path.suffix == ".json":
        return "application/json"
    ctype, _ = mimetypes.guess_type(str(path))
    return ctype or "application/octet-stream"


def _rel_remote_key(path: Path) -> str:
    """Remote object key in Supabase bucket.

    We preserve repo-relative structure when possible.
    """
    try:
        rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
        return rel
    except Exception:
        return path.name


def _storage_object_url(bucket: str, remote_key: str) -> str:
    base = SUPABASE_URL.rstrip("/")
    # Supabase Storage object endpoint.
    # Using PUT with x-upsert:true to overwrite.
    return f"{base}/storage/v1/object/{bucket}/{remote_key}"


def _upload_one(path: Path, bucket: str, remote_key: str) -> Tuple[bool, str]:
    url = _storage_object_url(bucket, remote_key)
    body = path.read_bytes()

    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": _guess_content_type(path),
        "x-upsert": "true",
    }

    req = Request(url=url, data=body, headers=headers, method="PUT")
    with urlopen(req, timeout=30) as resp:
        status = getattr(resp, "status", 200)
        if 200 <= status < 300:
            return True, f"{status}"
        return False, f"HTTP {status}"


def default_sync_paths() -> List[Path]:
    """Default set of files to sync, best-effort.

    We only include paths that *might* exist; missing ones are reported.
    """
    root = Path(PATHS.get("root") or ROOT)
    ml_data = Path(PATHS.get("ml_data") or (root / "ml_data"))
    ml_data_dt = Path(PATHS.get("ml_data_dt") or (root / "ml_data_dt"))
    nightly_logs = Path(PATHS.get("nightly_logs") or (root / "logs" / "nightly"))
    da_brains = Path(PATHS.get("da_brains") or (root / "da_brains"))

    candidates = [
        # Rolling brains
        Path(PATHS.get("rolling") or (da_brains / "rolling_body.json.gz")),
        Path(PATHS.get("rolling_nervous") or (da_brains / "rolling_nervous.json.gz")),
        Path(PATHS.get("aion_brain") or (da_brains / "core" / "aion_brain.json.gz")),
        da_brains / "system_perf.json",
        # Bots UI/config stores
        ml_data / "config" / "bots_config.json",
        ml_data / "config" / "bots_ui_overrides.json",
        ml_data_dt / "config" / "intraday_bots_ui.json",
        # Last-nightly summary (if present)
        nightly_logs / "last_nightly_summary.json",
        nightly_logs / "last_nightly_summary.json.gz",
        # Swing replay state
        root / "data" / "replay" / "swing" / "replay_state.json",
    ]

    # De-dupe while preserving order
    seen = set()
    out: List[Path] = []
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def sync_to_supabase(*, dry_run: bool = False, targets: Optional[List[str]] = None) -> SyncResult:
    """Upload selected artifacts to Supabase storage.

    targets:
      - If None: uses default_sync_paths().
      - If provided: list of repo-relative paths OR named shortcuts.
        Named shortcuts:
          - "defaults" => default_sync_paths()
    """
    uploaded: List[str] = []
    skipped: List[str] = []
    missing: List[str] = []
    errors: List[Dict[str, str]] = []

    if not supabase_configured():
        return SyncResult(
            uploaded=[],
            skipped=[],
            missing=[],
            errors=[{"error": "Supabase is not configured (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)."}],
        )

    bucket = (SUPABASE_BUCKET or "aion").strip() or "aion"

    # Resolve targets
    paths: List[Path] = []
    if not targets:
        paths = default_sync_paths()
    else:
        for t in targets:
            t = (t or "").strip()
            if not t:
                continue
            if t.lower() == "defaults":
                paths.extend(default_sync_paths())
                continue
            # repo-relative path
            candidate = (ROOT / t).resolve() if not os.path.isabs(t) else Path(t).resolve()
            paths.append(candidate)

    # De-dupe
    unique: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)

    for p in unique:
        if not p.exists() or not p.is_file():
            missing.append(str(p))
            continue

        remote_key = _rel_remote_key(p)
        if dry_run:
            skipped.append(f"DRY_RUN: {p} -> {bucket}/{remote_key}")
            continue

        try:
            ok, detail = _upload_one(p, bucket, remote_key)
            if ok:
                uploaded.append(f"{p} -> {bucket}/{remote_key}")
            else:
                errors.append({"path": str(p), "error": detail})
        except HTTPError as e:
            errors.append({"path": str(p), "error": f"HTTPError {e.code}: {getattr(e, 'reason', '')}"})
        except URLError as e:
            errors.append({"path": str(p), "error": f"URLError: {e}"})
        except Exception as e:
            errors.append({"path": str(p), "error": str(e)})

    return SyncResult(uploaded=uploaded, skipped=skipped, missing=missing, errors=errors)
