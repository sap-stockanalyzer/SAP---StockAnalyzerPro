# backend/routers/settings_router.py
"""
Settings Router

Provides a simple API for managing important API keys and settings
stored in the project's .env file.

Intended for use by the frontend "Tools" section where the user can:
  - View current API keys (broker, news, macro, social).
  - Edit keys in text boxes and save.
  - Optionally run a basic "test" to check presence/format.

This router:
  - Reads .env from PATHS["root"] / ".env"
  - Only exposes a curated subset of keys (SETTINGS_KEYS).
  - Preserves comments and unknown keys when updating.

NOTE: This does not attempt to hot-reload the process env; if you want
      live reload behavior you can add a small helper to re-read .env
      into your config layer after updates.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.core.config import PATHS
from utils.logger import log

router = APIRouter(prefix="/api/settings", tags=["settings"])

# Keys we expose in the UI.
SETTINGS_KEYS: List[str] = [
    # --- Trading ---
    "ALPACA_API_KEY_ID",
    "ALPACA_API_SECRET_KEY",
    "ALPACA_PAPER_BASE_URL",

    # --- Market / News ---
    "PERIGON_KEY",
    "FINNHUB_API_KEY",
    "NEWSAPI_KEY",
    "RSS2JSON_KEY",
    "MARKETAUX_API_KEY",

    # --- Social ---
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_USER_AGENT",
    "TWITTER_BEARER",
]

def _env_path() -> Path:
    root = PATHS.get("root")
    if not root:
        # Fallback to current working directory
        root = Path(os.getcwd())
    return Path(root) / ".env"


class SettingsUpdate(BaseModel):
    updates: Dict[str, str]


def _parse_env(content: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, val = stripped.split("=", 1)
        env[key.strip()] = val.strip()
    return env


def _load_env_file(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        log(f"[settings] ⚠️ Failed to read {path}: {e}")
        return ""


def _write_env_file(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as e:
        log(f"[settings] ⚠️ Failed to write {path}: {e}")
        raise


def _apply_updates_to_content(content: str, updates: Dict[str, str]) -> str:
    """
    Update KEY=VALUE lines in .env content while preserving comments and
    unknown keys. Adds new keys at the bottom if they don't exist yet.
    """
    lines = content.splitlines() if content else []
    existing_keys = set()
    updated_keys = set()

    new_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue

        key, _ = stripped.split("=", 1)
        key = key.strip()
        existing_keys.add(key)

        if key in updates:
            # Update this key
            new_val = updates[key]
            new_lines.append(f"{key}={new_val}")
            updated_keys.add(key)
        else:
            new_lines.append(line)

    # Append any new keys that didn't exist in file
    for key, val in updates.items():
        if key not in existing_keys:
            new_lines.append(f"{key}={val}")
            updated_keys.add(key)

    log(f"[settings] ✅ Updated keys in .env: {sorted(updated_keys)}")
    return "\n".join(new_lines) + "\n"


@router.get("/keys", summary="Get selected API keys and settings from .env")
def get_settings_keys() -> Dict[str, Any]:
    """
    Returns a subset of .env keys for UI editing.

    Response:
    {
      "keys": {
         "ALPACA_API_KEY_ID": "...",
         "ALPACA_PAPER_BASE_URL": "...",
         ...
      }
    }
    """
    path = _env_path()
    content = _load_env_file(path)
    env = _parse_env(content)

    result = {k: env.get(k, "") for k in SETTINGS_KEYS}
    return {"keys": result}


@router.post("/update", summary="Update API keys / settings in .env")
def update_settings(payload: SettingsUpdate) -> Dict[str, Any]:
    """
    Apply updates to .env.

    Payload:
    {
      "updates": {
        "ALPACA_API_KEY_ID": "xxx",
        "ALPACA_API_SECRET_KEY": "yyy",
        ...
      }
    }
    """
    path = _env_path()
    content = _load_env_file(path)

    try:
        new_content = _apply_updates_to_content(content, payload.updates)
        _write_env_file(path, new_content)
    except Exception as e:
        log(f"[settings] ⚠️ Failed to update .env: {e}")
        raise HTTPException(status_code=500, detail="Failed to update .env")

    # Optionally: reload process env here if desired
    return {"status": "ok"}


@router.get("/status", summary="Basic status / sanity check for settings")
def settings_status() -> Dict[str, Any]:
    """
    Basic health check on critical keys — does NOT hit external APIs,
    just verifies keys exist and are non-empty.
    """
    path = _env_path()
    content = _load_env_file(path)
    env = _parse_env(content)

    status: Dict[str, Any] = {}
    for key in SETTINGS_KEYS:
        val = env.get(key)
        status[key] = {
            "present": val is not None and val != "",
            "value_length": len(val) if val else 0,
        }

    return {"status": status}


@router.post("/test", summary="Lightweight validation of settings")
def test_settings(payload: SettingsUpdate | None = None) -> Dict[str, Any]:
    """
    Optional endpoint to validate keys.

    Current behavior:
      - If payload is provided, validate those values.
      - Otherwise, validate what's currently in .env.
    """
    if payload and payload.updates:
        env_view = payload.updates
        source = "payload"
    else:
        path = _env_path()
        content = _load_env_file(path)
        env_view = _parse_env(content)
        source = ".env"

    required_for_broker = ["ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"]
    missing_broker = [k for k in required_for_broker if not env_view.get(k)]

    ok = not missing_broker

    return {
        "source": source,
        "ok": ok,
        "missing_broker_keys": missing_broker,
    }


# ==========================================================
# Knobs Configuration Endpoints
# ==========================================================

class KnobsContent(BaseModel):
    content: str


@router.get("/knobs", summary="Get knobs.env content")
def get_knobs() -> Dict[str, Any]:
    """
    Get knobs.env content for EOD/Nightly configuration.
    
    Returns:
        {"content": "...file content..."}
    """
    root = PATHS.get("root")
    if not root:
        root = Path(os.getcwd())
    
    knobs_path = Path(root) / "knobs.env"
    if not knobs_path.exists():
        log(f"[settings] ⚠️ knobs.env not found at {knobs_path}")
        raise HTTPException(status_code=404, detail="knobs.env not found")
    
    try:
        content = knobs_path.read_text(encoding="utf-8")
        return {"content": content}
    except Exception as e:
        log(f"[settings] ⚠️ Failed to read knobs.env: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read knobs.env: {e}")


@router.post("/knobs", summary="Save knobs.env content")
def save_knobs(payload: KnobsContent) -> Dict[str, Any]:
    """
    Save knobs.env content for EOD/Nightly configuration.
    
    Payload:
        {"content": "...file content..."}
    """
    root = PATHS.get("root")
    if not root:
        root = Path(os.getcwd())
    
    knobs_path = Path(root) / "knobs.env"
    
    try:
        # Ensure parent directory exists
        knobs_path.parent.mkdir(parents=True, exist_ok=True)
        knobs_path.write_text(payload.content, encoding="utf-8")
        log(f"[settings] ✅ Saved knobs.env to {knobs_path}")
        return {"ok": True, "message": "Saved successfully"}
    except Exception as e:
        log(f"[settings] ⚠️ Failed to write knobs.env: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save knobs.env: {e}")


@router.get("/dt-knobs", summary="Get dt_knobs.env content")
def get_dt_knobs() -> Dict[str, Any]:
    """
    Get dt_knobs.env content for Intraday/day-trading configuration.
    
    Returns:
        {"content": "...file content..."}
    """
    root = PATHS.get("root")
    if not root:
        root = Path(os.getcwd())
    
    dt_knobs_path = Path(root) / "dt_knobs.env"
    if not dt_knobs_path.exists():
        log(f"[settings] ⚠️ dt_knobs.env not found at {dt_knobs_path}")
        raise HTTPException(status_code=404, detail="dt_knobs.env not found")
    
    try:
        content = dt_knobs_path.read_text(encoding="utf-8")
        return {"content": content}
    except Exception as e:
        log(f"[settings] ⚠️ Failed to read dt_knobs.env: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read dt_knobs.env: {e}")


@router.post("/dt-knobs", summary="Save dt_knobs.env content")
def save_dt_knobs(payload: KnobsContent) -> Dict[str, Any]:
    """
    Save dt_knobs.env content for Intraday/day-trading configuration.
    
    Payload:
        {"content": "...file content..."}
    """
    root = PATHS.get("root")
    if not root:
        root = Path(os.getcwd())
    
    dt_knobs_path = Path(root) / "dt_knobs.env"
    
    try:
        # Ensure parent directory exists
        dt_knobs_path.parent.mkdir(parents=True, exist_ok=True)
        dt_knobs_path.write_text(payload.content, encoding="utf-8")
        log(f"[settings] ✅ Saved dt_knobs.env to {dt_knobs_path}")
        return {"ok": True, "message": "Saved successfully"}
    except Exception as e:
        log(f"[settings] ⚠️ Failed to write dt_knobs.env: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save dt_knobs.env: {e}")
