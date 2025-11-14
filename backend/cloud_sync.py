"""
cloud_sync.py
v2 — Version check + incremental data/model backups to Supabase every 3 hours.

Uploads ONLY new or changed files (hash-checked) from these folders:
  - stock_cache/
  - fundamentals_cache/
  - news_cache/
  - ml_data/
  - dashboard_cache/   <-- added

Skips code files (.py, .pyc) and logs. Prints one line per folder:
  [cloud_sync] 📦 ml_data Updated successfully (4 files changed)
  [cloud_sync] 📦 stock_cache Already up to date
"""

from __future__ import annotations
import os
import json
import time
import threading
import hashlib
from contextlib import suppress
from datetime import datetime
from typing import Dict, Optional

# .env (backend_service already loads it, but this is safe if called standalone)
with suppress(Exception):
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()

# --- Env + Client ------------------------------------------------------------
def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
)
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "aion-cache")

SUPABASE_READY = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)

supabase = None
if SUPABASE_READY:
    try:
        from supabase import create_client  # type: ignore
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    except Exception as e:
        print(f"[cloud_sync] ⚠️ Supabase client init failed: {e}")
        SUPABASE_READY = False
else:
    print(
        "[cloud_sync] ⚠️ Version check failed: Missing SUPABASE_URL or "
        "SUPABASE_SERVICE_ROLE_KEY / SUPABASE_KEY / SUPABASE_ANON_KEY"
    )

# --- Config ------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FOLDERS_TO_SYNC = [
    "stock_cache",
    "fundamentals_cache",
    "news_cache",
    "ml_data",
    "dashboard_cache",   # ✅ added for Dashboard Intelligence
]
# Deny-list (we'll sync everything except these)
DENY_EXT = {".py", ".pyc", ".log"}

# Manifest cache location
_MANIFEST_PATH = os.path.join(ROOT, "logs", "cloudsync_manifest.json")

# Sync cadence (3 hours)
SYNC_INTERVAL_SECONDS = 3 * 60 * 60

# --- Helpers -----------------------------------------------------------------
def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()

def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _load_manifest() -> Dict[str, str]:
    try:
        with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_manifest(manifest: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(_MANIFEST_PATH), exist_ok=True)
    with open(_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

def _is_sync_candidate(path: str) -> bool:
    ext = _ext(path)
    if ext in DENY_EXT:
        return False
    # Skip hidden and temp files
    base = os.path.basename(path).lower()
    if base.startswith(".") or base.endswith(".tmp"):
        return False
    return True

# --- Core sync ---------------------------------------------------------------
def sync_folder(folder: str, manifest: Optional[Dict[str, str]] = None) -> int:
    """
    Upload ONLY changed/new files in `folder` to Supabase bucket, using hash diff.
    Returns number of files changed (uploaded).
    """
    if not SUPABASE_READY or not supabase:
        return 0

    local_dir = os.path.join(ROOT, folder)
    if not os.path.isdir(local_dir):
        return 0

    if manifest is None:
        manifest = _load_manifest()

    changed = 0
    for root, _dirs, files in os.walk(local_dir):
        for name in files:
            local_path = os.path.join(root, name)
            if not _is_sync_candidate(local_path):
                continue

            rel_path = os.path.relpath(local_path, local_dir).replace("\\", "/")
            key = f"{folder}/{rel_path}"
            sha1 = _sha1_file(local_path)
            prev = manifest.get(key)

            if prev == sha1:
                continue  # unchanged

            # Upload (replace if exists)
            bucket = supabase.storage.from_(SUPABASE_BUCKET)
            try:
                bucket.remove([key])  # emulate upsert
            except Exception:
                pass

            # Skip oversized files (>50 MB)
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            if size_mb > 50:
                print(f"[cloud_sync] ⚠️ Skipped large file: {key} ({size_mb:.1f} MB)")
                continue

            # Upload (replace if exists)
            bucket = supabase.storage.from_(SUPABASE_BUCKET)
            try:
                bucket.remove([key])  # emulate upsert
            except Exception:
                pass

            with open(local_path, "rb") as fh:
                data = fh.read()
            bucket.upload(key, data)

            manifest[key] = sha1
            changed += 1

    # Persist manifest after folder processed
    _save_manifest(manifest)
    return changed


def sync_all() -> None:
    """
    Runs one full sync pass over all configured folders.
    Prints a single line per folder.
    """
    if not SUPABASE_READY or not supabase:
        return

    manifest = _load_manifest()
    total_changes = 0
    for folder in FOLDERS_TO_SYNC:
        n = sync_folder(folder, manifest=manifest)
        total_changes += n
        if n > 0:
            print(f"[cloud_sync] 📦 {folder} Updated successfully ({n} files changed)")
        else:
            print(f"[cloud_sync] 📦 {folder} Already up to date")

    if total_changes == 0:
        print(f"[cloud_sync] ℹ️ No changes to upload at {_now()}")
    else:
        print(f"[cloud_sync] ✅ Sync complete — {total_changes} files uploaded at {_now()}")

# --- Version check -----------------------------------------------------------
def version_check_once() -> None:
    """
    Writes a small /version.json to the bucket root, including timestamp and local version.
    """
    if not SUPABASE_READY or not supabase:
        return

    payload = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "backend_version": os.getenv("SAP_BACKEND_VERSION", "unknown"),
        "folders": FOLDERS_TO_SYNC,
    }

    data = json.dumps(payload, indent=2).encode("utf-8")
    bucket = supabase.storage.from_(SUPABASE_BUCKET)
    try:
        # Remove any old version.json (since "upsert" isn't supported)
        bucket.remove(["version.json"])
    except Exception:
        pass

    bucket.upload("version.json", data)
    print("[cloud_sync] 🏷️  version.json updated in bucket")
    return payload

# --- Background loop ---------------------------------------------------------
def _background_loop():
    print(f"[cloud_sync] 🕒 Sync interval: {SYNC_INTERVAL_SECONDS // 3600}h")
    while True:
        try:
            version_check_once()
            sync_all()
        except Exception as e:
            print(f"[cloud_sync] ⚠️ background sync error: {e}")
        time.sleep(SYNC_INTERVAL_SECONDS)

def start_background_sync():
    """Starts the background sync thread."""
    if not SUPABASE_READY:
        return
    t = threading.Thread(target=_background_loop, name="cloud-sync", daemon=True)
    t.start()
    print("[cloud_sync] ✅ Background cloud sync started.")

# --- Compatibility alias for backend_service.py ---
start_background_tasks = start_background_sync

# -------------------------------------------------------------
# 🧩 Command-line Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    print("[cloud_sync] 🚀 Manual sync triggered...", flush=True)
    try:
        version_check_once()
        sync_all()
        print("[cloud_sync] ✅ Manual sync complete.")
    except Exception as e:
        print(f"[cloud_sync] ❌ Manual sync failed: {e}")

