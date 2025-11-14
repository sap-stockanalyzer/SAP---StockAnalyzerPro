"""Vector index for semantic similarity (fallbacks if deps missing, unified config)."""
from __future__ import annotations
import os, json, math, hashlib
from typing import List, Dict
from .config import PATHS  # ✅ unified path import

# ---------------------------------------------------------------------
# Config-aware paths
# ---------------------------------------------------------------------
INDEX_PATH = PATHS["news"] / "vector_index.jsonl"  # ✅ replaces news_cache/
MODEL_NAME = os.getenv("AION_EMBED_MODEL", "all-MiniLM-L6-v2")

_MODEL = None

def _get_model():
    """Load sentence transformer model (or fallback to hashing)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _MODEL = SentenceTransformer(MODEL_NAME)
    except Exception:
        _MODEL = None
    return _MODEL


def _embed(text: str) -> List[float]:
    """Return 384d vector embedding or deterministic hash fallback."""
    model = _get_model()
    if model is None:
        h = hashlib.sha256((text or "").encode("utf-8")).digest()
        return [((b - 128) / 128.0) for b in h[:32]]
    try:
        v = model.encode([text])[0]
        return v.tolist()
    except Exception:
        h = hashlib.sha256((text or "").encode("utf-8")).digest()
        return [((b - 128) / 128.0) for b in h[:32]]


def add(doc_id: str, text: str, meta: Dict):
    """Append new document to vector index."""
    os.makedirs(INDEX_PATH.parent, exist_ok=True)
    row = {"id": doc_id, "vec": _embed(text), "meta": meta}
    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_all():
    """Load all rows from the vector index file."""
    if not INDEX_PATH.exists():
        return []
    rows = []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _cosine(a, b):
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    s = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return s / (na * nb)


def search_similar(text: str, k: int = 3, score_threshold: float = 0.9):
    """Return top-k similar docs with cosine similarity ≥ threshold."""
    target = _embed(text)
    rows = _load_all()
    scored = []
    for r in rows:
        sim = _cosine(target, r.get("vec") or [])
        if sim >= score_threshold:
            scored.append({"id": r.get("id"), "score": sim, "meta": r.get("meta")})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]
