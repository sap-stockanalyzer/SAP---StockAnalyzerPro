# backend/core/data_pipeline.py — v1.3 (Aligned: context is source-of-truth for news/social + AION brain support)
"""
Data Pipeline — AION Analytics (Rolling Engine)

Responsibilities:
    • Load & save rolling.json.gz and rolling_brain.json.gz safely.
    • Normalize core fields:
        symbol, sector, predictions, context, news, social, policy.
    • Guarantee every prediction horizon exists with safe default values.
    • Guarantee safe float conversion for all numeric fields.
    • Maintain backups of rolling + brain before overwriting.

UPDATED (v1.2):
    ✅ rolling_brain moved to brains folder (PATHS["rolling_brain"])
    ✅ Removes “split-brain” fields:
         - If context has ctx["news"] / ctx["social"], those become canonical node["news"]/node["social"]
         - If context lacks them but node has news/social, we inject into context (keeps everything unified)

UPDATED (v1.3):
    ✅ Adds AION brain support:
         - _read_aion_brain()
         - save_aion_brain()
      Uses canonical path via PATHS["brain"] (aion_brain.json.gz)

NOTE:
    This version is aligned with backend.core.config.PATHS:

        PATHS["rolling"]         → (your canonical rolling path)
        PATHS["rolling_brain"]   → da_brains/core/rolling_brain.json.gz
        PATHS["brain"]           → da_brains/core/aion_brain.json.gz
        PATHS["rolling_backups"] → data/stock_cache/master/backups
"""

from __future__ import annotations

import json
import gzip
import shutil
import os
from pathlib import Path
from typing import Dict, Any

from backend.core.config import PATHS
from utils.logger import log as _log  # shared logger

# IMPORTANT:
# ai_model imports `log` from this module.
# NOTE: backend.core.ai_model legacy alias has been removed; callers must import explicit ai_model submodules.
log = _log

# -------------------------------------------------------------
# Paths (aligned with config.PATHS)
# -------------------------------------------------------------

ROLLING_BODY_PATH: Path = (PATHS.get("rolling_body") or PATHS["rolling"])
ROLLING_NERVOUS_PATH: Path | None = PATHS.get("rolling_nervous")
BRAIN_PATH: Path = (PATHS.get("rolling_brain") or PATHS["rolling_brain"])

# ✅ NEW (v1.3): AION brain (behavioral memory)
AION_BRAIN_PATH: Path = PATHS.get("brain")  # may be absent in older configs
if AION_BRAIN_PATH is None:
    # fall back safely to a conventional location (won't crash import)
    AION_BRAIN_PATH = Path(PATHS["root"]) / "da_brains" / "core" / "aion_brain.json.gz"

BACKUP_DIR: Path = PATHS["rolling_backups"]
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = ["1d", "3d", "1w", "2w", "4w", "13w", "26w", "52w"]

# Log read summaries (counts) can be noisy when UI polls /api/system/status.
# Set AION_LOG_READ_SUMMARY=1 to re-enable these per-read logs.
AION_LOG_READ_SUMMARY = os.getenv("AION_LOG_READ_SUMMARY", "0").strip().lower() in {"1", "true", "yes", "y", "on"}


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def safe_float(x) -> float:
    """Convert to float safely with fallback."""
    try:
        if x is None:
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def _load_json_gz(path: Path) -> Dict[str, Any]:
    """Safe JSON.gz loader with logging."""
    if not path.exists():
        log(f"[data_pipeline] ℹ️ {path} does not exist — returning empty dict.")
        return {}
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            log(f"[data_pipeline] ⚠️ {path} did not contain a dict — coercing to empty.")
            return {}
        return data
    except Exception as e:
        log(f"[data_pipeline] ⚠️ Failed to load {path}: {e}")
        # If this is a critical canonical file, attempt restore from backups and retry once.
        try:
            if path == ROLLING_BODY_PATH:
                if _restore_latest_backup(ROLLING_BODY_PATH, PATHS.get("rolling_backups", ROLLING_BODY_PATH.parent)):
                    with gzip.open(path, "rt", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        return data
            if path == BRAIN_PATH:
                if _restore_latest_backup(BRAIN_PATH, PATHS.get("rolling_backups", BRAIN_PATH.parent)):
                    with gzip.open(path, "rt", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        return data
        except Exception:
            pass
        return {}


def _save_json_gz(path: Path, data: Dict[str, Any]):
    """
    Atomic-ish write of JSON.gz with a temp file.

    Windows note:
    - Use a temp file in the SAME directory
    - os.replace is atomic on the same filesystem
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(str(tmp), str(path))
    except Exception as e:
        log(f"[data_pipeline] ❌ Failed to write {path}: {e}")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass



def _restore_latest_backup(target: Path, backups_dir: Path) -> bool:
    """Best-effort restore of latest backup file for target into target."""
    try:
        if not backups_dir.exists():
            return False
        # backups are stored as <name>.YYYYMMDD_HHMMSS.bak (or similar). pick latest by mtime
        candidates = sorted(backups_dir.glob(target.name + ".*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return False
        src = candidates[0]
        backups_dir.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        # copy then replace atomically-ish
        tmp = target.with_name(target.name + ".restore_tmp")
        tmp.write_bytes(src.read_bytes())
        os.replace(str(tmp), str(target))
        log(f"[data_pipeline] 🛟 Restored {target.name} from backup: {src.name}")
        return True
    except Exception as e:
        log(f"[data_pipeline] ⚠️ Backup restore failed for {target}: {e}")
        return False

def _backup_file(path: Path):
    """Copy current file into BACKUP_DIR with mtime-based suffix."""
    if not path.exists():
        return
    ts = path.stat().st_mtime
    name = path.name.replace(".json.gz", "")
    out = BACKUP_DIR / f"{name}_{int(ts)}.json.gz"
    try:
        shutil.copy2(path, out)
        log(f"[data_pipeline] 📦 Backup → {out.name}")
    except Exception as e:
        log(f"[data_pipeline] ⚠️ Failed backup: {e}")


def _norm_dict_block(x: Any) -> Dict[str, Any]:
    """
    Normalize a dict-like block:
      - ensures dict
      - converts numeric primitives to float (preserves strings, bools, nested dict/list)
    """
    if not isinstance(x, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in x.items():
        if isinstance(v, (int, float)):
            out[k] = safe_float(v)
        else:
            out[k] = v
    return out


# -------------------------------------------------------------
# Public Loaders
# -------------------------------------------------------------

def _read_rolling() -> Dict[str, Any]:
    """Load the canonical rolling snapshot."""
    data = _load_json_gz(ROLLING_BODY_PATH)
    if AION_LOG_READ_SUMMARY:
        log(f"[data_pipeline] ℹ️ _read_rolling → {len(data)} keys from {ROLLING_BODY_PATH}")
    return data


def _read_brain() -> Dict[str, Any]:
    """
    Load the canonical rolling brain snapshot.

    NOTE:
        This now lives under:
            ROOT/da_brains/core/rolling_brain.json.gz
    """
    data = _load_json_gz(BRAIN_PATH)
    if AION_LOG_READ_SUMMARY:
        log(f"[data_pipeline] ℹ️ _read_brain → {len(data)} keys from {BRAIN_PATH}")
    return data


def _read_aion_brain() -> Dict[str, Any]:
    """
    Load the canonical AION brain snapshot.

    This brain is global behavior/memory:
      - confidence calibration knobs
      - risk bias knobs
      - aggressiveness knobs
      - regime-level modifiers
    """
    data = _load_json_gz(AION_BRAIN_PATH)
    if AION_LOG_READ_SUMMARY:
        log(f"[data_pipeline] ℹ️ _read_aion_brain → {len(data)} keys from {AION_BRAIN_PATH}")
    return data


# -------------------------------------------------------------
# Normalization logic
# -------------------------------------------------------------

def _ensure_prediction_block(block: dict) -> dict:
    """
    Guarantee the fields required by prediction_logger + UI.
    Normalizes BOTH model outputs AND missing predictions.

    Regression fields supported:
      - score, confidence, label, predicted_return, target_price
      - rating, rating_score, components (preserved if present)
    """
    out = {
        "score": safe_float(block.get("score")),
        "confidence": safe_float(block.get("confidence")),
        "label": int(block.get("label", 0) or 0),
        "predicted_return": safe_float(block.get("predicted_return")),
        "base_return": safe_float(block.get("base_return")),
        "target_price": block.get("target_price"),
        "rating": block.get("rating"),
        "rating_score": block.get("rating_score"),
        "components": block.get("components"),
    }

    try:
        if out["target_price"] is not None:
            out["target_price"] = float(out["target_price"])
    except Exception:
        out["target_price"] = None

    try:
        if out["rating_score"] is not None:
            out["rating_score"] = int(out["rating_score"])
    except Exception:
        out["rating_score"] = None

    if out["components"] is not None and not isinstance(out["components"], dict):
        out["components"] = None

    return out


def _ensure_predictions(preds: dict) -> dict:
    """Ensure predictions contain all required horizons."""
    out: Dict[str, Any] = {}

    if not isinstance(preds, dict):
        preds = {}

    for h in HORIZONS:
        block = preds.get(h)
        if isinstance(block, dict):
            out[h] = _ensure_prediction_block(block)
        else:
            out[h] = {
                "score": 0.0,
                "confidence": 0.0,
                "label": 0,
                "predicted_return": 0.0,
                "base_return": 0.0,
                "target_price": None,
                "rating": None,
                "rating_score": None,
                "components": None,
            }

    return out


def _normalize_symbol(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Guarantees required subfields exist: predictions, context,
    news, social, policy.

    Canon rules (v1.2):
      - context is the primary fusion layer
      - if context includes ctx["news"]/ctx["social"], those become node["news"]/node["social"]
      - if context does NOT include them but node does, we inject them into context
    """
    node = dict(node)

    # ---------- Sector ----------
    sector = node.get("sector") or (node.get("fundamentals") or {}).get("sector")
    if not isinstance(sector, str):
        sector = ""
    node["sector"] = sector.upper().strip()

    # ---------- Predictions ----------
    node["predictions"] = _ensure_predictions(node.get("predictions") or {})

    # ---------- Context (canonical fusion block) ----------
    ctx = _norm_dict_block(node.get("context") or {})
    ctx_news = ctx.get("news")
    ctx_social = ctx.get("social")

    # ---------- Legacy/top-level news/social ----------
    news_top = _norm_dict_block(node.get("news") or {})
    social_top = _norm_dict_block(node.get("social") or {})

    # Prefer context-state structured blocks if present
    if isinstance(ctx_news, dict) and ctx_news:
        news = _norm_dict_block(ctx_news)
    else:
        news = news_top

    if isinstance(ctx_social, dict) and ctx_social:
        social = _norm_dict_block(ctx_social)
    else:
        social = social_top

    # Inject into context if missing (prevents divergence over time)
    if not isinstance(ctx.get("news"), dict) or not ctx.get("news"):
        if news:
            ctx["news"] = dict(news)

    if not isinstance(ctx.get("social"), dict) or not ctx.get("social"):
        if social:
            ctx["social"] = dict(social)

    # Keep top-level mirrors for older services/UI expectations
    node["context"] = ctx
    node["news"] = news
    node["social"] = social

    # ---------- Policy ----------
    pol = node.get("policy") or {}
    node["policy"] = pol if isinstance(pol, dict) else {}

    return node


def _normalize_rolling(rolling: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for sym, node in rolling.items():
        if sym.startswith("_"):
            out[sym] = node
            continue
        out[sym] = _normalize_symbol(node) if isinstance(node, dict) else {}
    return out


# -------------------------------------------------------------
# Backwards-compatible helper
# -------------------------------------------------------------

def ensure_symbol_fields(node: Dict[str, Any]) -> Dict[str, Any]:
    """Alias for legacy services."""
    return _normalize_symbol(node)


# -------------------------------------------------------------
# Save operations (with backup)
# -------------------------------------------------------------


def _read_rolling_nervous() -> Dict[str, Any]:
    """Read ML/metrics rolling (rolling_nervous). Returns {} on failure."""
    if not ROLLING_NERVOUS_PATH:
        return {}
    if not Path(ROLLING_NERVOUS_PATH).exists():
        return {}
    try:
        with gzip.open(ROLLING_NERVOUS_PATH, "rt", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if AION_LOG_READ_SUMMARY:
                log(f"[data_pipeline] ℹ️ _read_rolling_nervous → {len(data)} keys from {ROLLING_NERVOUS_PATH}")
            return data
    except Exception as e:
        log(f"[data_pipeline] ⚠️ Failed to load {ROLLING_NERVOUS_PATH}: {e}")
    return {}


def save_rolling_nervous(data: Dict[str, Any]) -> bool:
    """Atomic write for rolling_nervous."""
    if not ROLLING_NERVOUS_PATH:
        return False
    out_path = Path(ROLLING_NERVOUS_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
            json.dump(data, f)
        tmp_path.replace(out_path)
        return True
    except Exception as e:
        log(f"[data_pipeline] ⚠️ Failed to save nervous rolling {out_path}: {e}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        return False

def save_rolling(rolling: Dict[str, Any], *, allow_empty: bool = False):
    """Normalize & backup rolling before overwriting.

    Safety:
        Never wipe the canonical rolling cache with an empty dict unless explicitly forced.
        This prevents a single failed read from cascading into a total rolling wipe.
    """
    if not isinstance(rolling, dict):
        return

    if (not allow_empty) and len(rolling) == 0 and ROLLING_BODY_PATH.exists():
        log("[data_pipeline] 🛑 Refusing to overwrite rolling with empty dict (allow_empty=False).")
        return

    rolling = _normalize_rolling(rolling)
    _backup_file(ROLLING_BODY_PATH)
    _save_json_gz(ROLLING_BODY_PATH, rolling)
    log(f"[data_pipeline] 💾 rolling.json.gz updated ({len(rolling)} symbols)")


def save_brain(brain: Dict[str, Any]):
    """Backup + save rolling brain snapshot."""
    if not isinstance(brain, dict):
        return

    _backup_file(BRAIN_PATH)
    _save_json_gz(BRAIN_PATH, brain)
    log(f"[data_pipeline] 🧠 rolling_brain.json.gz updated ({len(brain)} entries)")


def save_aion_brain(brain: Dict[str, Any]):
    """Save AION brain snapshot (behavioral brain)."""
    if not isinstance(brain, dict):
        return

    # AION brain is small; we keep it stable and do not spam backups into rolling_backups.
    # If you want backups later, we can add PATHS["brains_backups"] in config.
    _save_json_gz(AION_BRAIN_PATH, brain)
    log(f"[data_pipeline] 🧠 aion_brain.json.gz updated ({len(brain)} keys)")
