"""ABAC-filtered RAG retrieval core used by the unified chat orchestrator.

Retrieval-only: no LLM call. The ABAC predicate is applied inside the SQL
query (pre-filter, deny-by-default), similarity is cosine in Python over the
permitted rows, and every retrieved chunk is re-scanned by the Prompt-Guard
classifier before it can enter the context (indirect-injection defense).

Unauthorized-access detection: the query is also scored against the rows the
ABAC policy withholds from the requesting user. When the best restricted
similarity reaches ``GUARD_ABAC_MATCH_THRESHOLD`` the outcome reports an
``unauthorized_attempt`` and the caller rejects the request. Restricted chunk
text is never logged.

All JSONL rows use the unified schema from :mod:`guard.flags` — ids, counts,
labels, and the masked query only.
"""

import logging
import os
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from guard.abac import retrieval_predicate, subject_attributes
from guard.db import Chunk, Document, User
from guard.flags import (
    EVENT_RAG_INJECTION,
    LAYER_PROMPT_GUARD,
    append_flag,
    user_block,
)
from guard.steps.embedding import embed
from guard.steps.prompt_guard import PromptGuardVerdict, classify

logger = logging.getLogger("guard.rag")

RULE_RAG_INJECTION = "RAG_CONTEXT_INJECTION"
DEFAULT_TOP_K = 5
DEFAULT_ABAC_MATCH_THRESHOLD = 0.65
RESTRICTED_MATCH_LIMIT = 5


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_id: int
    ordinal: int
    title: str
    text: str
    score: float


@dataclass(frozen=True)
class Citation:
    document_id: int
    chunk_id: int
    title: str
    score: float


@dataclass(frozen=True)
class RetrievalOutcome:
    """Result of the shared ABAC-filtered retrieval core (no screening, no audit row)."""

    chunks: list[RetrievedChunk]
    permitted: int
    dropped_chunk_ids: list[int]
    embedding_engine: str | None
    engine_mismatch: bool
    assembled_context: str
    citations: list[Citation]
    unauthorized_attempt: bool = False
    restricted_top_score: float = 0.0
    restricted_match_ids: list[int] = field(default_factory=list)


def get_top_k() -> int:
    return int(os.environ.get("GUARD_RAG_TOP_K", str(DEFAULT_TOP_K)))


def get_abac_threshold() -> float:
    return float(os.environ.get("GUARD_ABAC_MATCH_THRESHOLD", DEFAULT_ABAC_MATCH_THRESHOLD))


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _permitted_rows(session: Session, predicate) -> list:
    statement: Select = (
        select(
            Chunk.id,
            Chunk.ordinal,
            Chunk.text,
            Chunk.embedding,
            Chunk.embedding_engine,
            Document.id.label("document_id"),
            Document.title,
        )
        .join(Document, Chunk.document_id == Document.id)
        .where(predicate)
    )
    return list(session.execute(statement))


def _scan_chunk(row) -> PromptGuardVerdict:
    """Indirect-injection scan of one retrieved chunk (label + score only in logs)."""
    return classify(row.text)


def _restricted_attempt(
    session: Session,
    predicate,
    vector: list[float],
    engine: str,
) -> tuple[bool, float, list[int]]:
    """Score the query against rows the ABAC policy withholds from the user.

    Returns ``(unauthorized_attempt, restricted_top_score, restricted_match_ids)``
    where match ids are document ids (never chunk text) of restricted documents
    the query strongly matches.
    """
    threshold = get_abac_threshold()
    restricted_rows = [
        row
        for row in _permitted_rows(session, ~predicate)
        if row.embedding_engine == engine and row.embedding
    ]
    if not restricted_rows:
        return False, 0.0, []
    scored = sorted(
        ((row, _cosine(vector, row.embedding)) for row in restricted_rows),
        key=lambda pair: pair[1],
        reverse=True,
    )
    top_score = round(scored[0][1], 4)
    match_ids: list[int] = []
    for row, score in scored:
        if score >= threshold and row.document_id not in match_ids:
            match_ids.append(row.document_id)
        if len(match_ids) >= RESTRICTED_MATCH_LIMIT:
            break
    return top_score >= threshold, top_score, match_ids


def retrieve(
    session: Session,
    user: User,
    query_text: str,
    top_k: int | None = None,
) -> RetrievalOutcome:
    """Embed ``query_text`` (already masked by the caller) and retrieve chunks.

    Used by the unified chat orchestrator. The ABAC predicate applies to ``user``'s DB row (deny-by-default), similarity is cosine over
    permitted rows, and every candidate chunk is re-scanned by the Prompt-Guard
    classifier (flagged chunks are dropped and audited). The query is also
    scored against restricted rows to detect unauthorized-access attempts
    (``unauthorized_attempt`` on the outcome). Does not write a ``RAG_QUERY``
    row - callers own their own auditing.
    """
    if top_k is None:
        top_k = get_top_k()

    vector, engine, _model = embed(query_text)
    subject = subject_attributes(user)
    predicate = retrieval_predicate(subject)

    rows = _permitted_rows(session, predicate)
    permitted = len(rows)
    engine_rows = [row for row in rows if row.embedding_engine == engine and row.embedding]
    engine_mismatch = permitted > 0 and not engine_rows

    unauthorized_attempt, restricted_top_score, restricted_match_ids = _restricted_attempt(
        session, predicate, vector, engine
    )
    if unauthorized_attempt:
        logger.warning(
            "rag | unauthorized access attempt: query matches restricted documents %s "
            "(top_score=%.4f >= threshold %.2f)",
            restricted_match_ids,
            restricted_top_score,
            get_abac_threshold(),
        )

    scored = [(row, _cosine(vector, row.embedding)) for row in engine_rows]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    kept = []
    dropped_chunk_ids: list[int] = []
    for row, score in scored[:top_k]:
        verdict = _scan_chunk(row)
        if verdict.flagged:
            dropped_chunk_ids.append(row.id)
            append_flag(
                {
                    "event": EVENT_RAG_INJECTION,
                    "layer": LAYER_PROMPT_GUARD,
                    "disposition": "DROP_CHUNK",
                    "severity": "high",
                    "reason": "suspected prompt injection in retrieved chunk",
                    "rules": [RULE_RAG_INJECTION],
                    "user": user_block(user),
                    "prompt_masked": "",
                    "details": {
                        "document_id": row.document_id,
                        "chunk_id": row.id,
                        "label": verdict.label,
                        "suspicious_score": verdict.suspicious_score,
                    },
                    "audit_id": None,
                }
            )
            logger.info(
                "rag | dropped chunk %d from context: injection suspected "
                "(label=%s score=%.4f engine=%s)",
                row.id,
                verdict.label,
                verdict.suspicious_score,
                verdict.engine,
            )
            continue
        kept.append((row, score))

    chunks = [
        RetrievedChunk(
            chunk_id=row.id,
            document_id=row.document_id,
            ordinal=row.ordinal,
            title=row.title,
            text=row.text,
            score=round(score, 4),
        )
        for row, score in kept
    ]
    assembled = "\n\n".join(
        f"[{index}] {chunk.text}" for index, chunk in enumerate(chunks, start=1)
    )
    citations = [
        Citation(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            score=chunk.score,
        )
        for chunk in chunks
    ]
    return RetrievalOutcome(
        chunks=chunks,
        permitted=permitted,
        dropped_chunk_ids=dropped_chunk_ids,
        embedding_engine=engine,
        engine_mismatch=engine_mismatch,
        assembled_context=assembled,
        citations=citations,
        unauthorized_attempt=unauthorized_attempt,
        restricted_top_score=restricted_top_score,
        restricted_match_ids=restricted_match_ids,
    )
