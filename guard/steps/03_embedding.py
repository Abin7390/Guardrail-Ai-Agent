"""Stage 3: local text embeddings for RAG retrieval.

Primary engine is sentence-transformers MiniLM-L6-v2 (384-dim, mean-pooled,
L2-normalized) running on the pinned torch CPU wheel. Falls back to a
deterministic 384-bucket hashing vectorizer over character trigrams when
sentence-transformers or the model is unavailable, so retrieval keeps working
offline (engine tag "fallback"). Vectors from the two engines are NOT
comparable: the ask flow gates scoring on matching ``embedding_engine`` tags.
"""

import hashlib
import logging
import math
import os
import threading

logger = logging.getLogger("guard.embedding")

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_DIM = 384
FALLBACK_MODEL = "char-trigram-hash-384"
MINILM_ENGINE = "minilm"
FALLBACK_ENGINE = "fallback"

_lock = threading.Lock()
_model = None
_engine_mode: str | None = None


def _fallback_vector(text: str) -> list[float]:
    """Deterministic hashed char-trigram TF vector, L2-normalized."""
    logger.info("embedding | fallback embedding text: %s", text)
    vector = [0.0] * FALLBACK_DIM
    lowered = (text or "").lower()
    for start in range(max(len(lowered) - 2, 0)):
        trigram = lowered[start : start + 3]
        digest = hashlib.md5(trigram.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % FALLBACK_DIM
        vector[bucket] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _ensure_engine() -> None:
    global _model, _engine_mode
    if _engine_mode is not None:
        return
    with _lock:
        if _engine_mode is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            model_id = os.environ.get("GUARD_EMBED_MODEL", DEFAULT_EMBED_MODEL)
            logger.info("embedding | loading model: %s (first run downloads it)", model_id)
            _model = SentenceTransformer(model_id)
            _engine_mode = MINILM_ENGINE
            get_dims = getattr(_model, "get_embedding_dimension", None)
            if get_dims is None:
                get_dims = _model.get_sentence_embedding_dimension
            logger.info("embedding | engine ready: minilm (%d dims)", int(get_dims()))
        except Exception as exc:
            _model = None
            _engine_mode = FALLBACK_ENGINE
            logger.warning(
                "embedding | model unavailable (%s): hashed-trigram fallback active", exc
            )


def get_engine_mode() -> str:
    """Current embedding engine: 'minilm', 'fallback', or 'unloaded'."""
    return _engine_mode or "unloaded"


def get_model_id() -> str:
    if _engine_mode == MINILM_ENGINE:
        return os.environ.get("GUARD_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    return FALLBACK_MODEL


def embed(text: str) -> tuple[list[float], str, str]:
    """Embed one text; returns (vector, engine, model), L2-normalized."""
    _ensure_engine()
    if _engine_mode == MINILM_ENGINE:
        logger.info("embedding | minilm embedding text: %s", text)
        vector = _model.encode(text, normalize_embeddings=True).tolist()
        return vector, MINILM_ENGINE, get_model_id()
    return _fallback_vector(text), FALLBACK_ENGINE, FALLBACK_MODEL
