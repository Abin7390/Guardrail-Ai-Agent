"""Unified chat orchestrator: guard -> mask -> route -> retrieve -> answer -> demask.

One call runs the full chain per request:

1. ``screen(prompt, reversible=True)`` - REJECT halts everything; MASKED and
   CLEAN prompts continue with the masked text (PII redaction happens first,
   the flag row is written by ``screen``).
2. Router LLM call on the masked prompt plus the requester's ABAC context,
   returning strict JSON ``{"is_flagged", "flag_reason", "severity",
   "needs_rag", "search_query"}``; any failure falls back to
   ``needs_rag=False`` (recorded in the audit row). A flagged decision halts
   the request (``LLM_REJECT``) and writes a flag row.
3. If needed, ``retrieve()`` - ABAC-filtered, injection-screened chunks for
   the requesting user. When the query strongly matches content the user is
   not allowed to see, the request is rejected (``ABAC_REJECT``) and the
   attempt is flagged.
4. Answer LLM call with the masked prompt (plus assembled context when
   retrieval produced chunks).
5. ``demask()`` restores the indexed placeholders in the LLM output.

Every request writes one ``audit_log`` row (masked content only): never the
raw prompt, never the demasked answer, never mapping values. Security events
are appended to ``flags.jsonl`` via :mod:`guard.flags` with layer, user, the
full masked prompt, severity, and reason.
"""

import logging
import re
import time
from dataclasses import dataclass, replace

from sqlalchemy.orm import Session

from guard.abac import POLICY_VERSION, subject_attributes
from guard.db import AuditLog, User
from guard.flags import LAYER_ABAC, LAYER_LLM_ROUTER, write_flag
from guard.llm import LLMClientError, LLMResponse, get_client
from guard.pipeline import GuardResult, screen
from guard.steps.rag import Citation, RetrievalOutcome, RetrievedChunk, retrieve
from guard.steps.masking import MaskingResult, demask
from guard.steps.prompt_guard import PromptGuardVerdict, get_threshold
from guard.steps.llm_flagging import RULE_LLM_ROUTER, RouterDecision, llm_flagging

logger = logging.getLogger("guard.chat")

STATUS_ANSWER = "ANSWER"
STATUS_REJECTED = "REJECTED"
STATUS_LLM_ERROR = "LLM_ERROR"

ANSWER_TEMPERATURE = 0.2
ANSWER_MAX_TOKENS = 1024

_ANSWER_SYSTEM_BASE = (
    "You are a helpful, concise assistant. The user's message may contain "
    "redaction placeholders like [REDACTED_7] where sensitive information was "
    "removed. Keep every such token verbatim in your answer; never guess, "
    "reconstruct, or fill in redacted content."
)
_ANSWER_SYSTEM_CONTEXT = (
    _ANSWER_SYSTEM_BASE
    + "\n\nAnswer using the numbered context below. Whenever you rely on the "
    "context, cite the relevant entries inline as [n] matching their numbers. "
    "If the context does not contain the answer, say so plainly.\n\nContext:\n"
)

_PLACEHOLDER_TOKEN = re.compile(r"\[REDACTED_(\d+)\]")


@dataclass(frozen=True)
class ChatResult:
    status: str
    answer_demasked: str | None
    answer_masked: str | None
    masked_prompt: str | None
    disposition: str
    verdict: PromptGuardVerdict | None
    masking: MaskingResult | None
    router: RouterDecision | None
    chunks: list[RetrievedChunk]
    citations: list[Citation]
    llm_model: str | None
    llm_usage: dict | None
    latency_ms: int
    audit_id: int | None
    error: str | None = None


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _rejected_result(
    started: float,
    guarded: GuardResult,
    masked_prompt: str | None,
    router_decision: RouterDecision | None,
    masking: MaskingResult | None,
    audit_id: int,
) -> ChatResult:
    return ChatResult(
        status=STATUS_REJECTED,
        answer_demasked=None,
        answer_masked=None,
        masked_prompt=masked_prompt,
        disposition=guarded.disposition,
        verdict=guarded.verdict,
        masking=masking,
        router=router_decision,
        chunks=[],
        citations=[],
        llm_model=None,
        llm_usage=None,
        latency_ms=_elapsed_ms(started),
        audit_id=audit_id,
    )


def _write_audit(
    session: Session,
    *,
    user: User,
    subject: dict,
    status: str,
    masked_prompt: str | None,
    guarded: GuardResult,
    router_decision: RouterDecision | None,
    outcome: RetrievalOutcome | None,
    llm_response: LLMResponse | None,
    llm_error: str | None,
    llm_latency_ms: int,
    restored_count: int,
    unmatched_count: int,
) -> int:
    """Persist one audit_log row (masked content only) and return its id."""
    verdict = guarded.verdict
    guard_json = {
        "disposition": guarded.disposition,
        "rules": guarded.rules,
        "label": verdict.label if verdict else None,
        "suspicious_score": verdict.suspicious_score if verdict else None,
        "threshold": get_threshold(),
        "engine": verdict.engine if verdict else None,
    }
    masking_json = None
    if guarded.masking is not None:
        masking_json = {
            "engine": guarded.masking.engine,
            "entities": guarded.masking.entities,
            "placeholder_count": len(guarded.masking.mapping or []),
        }
    router_json = None
    if router_decision is not None:
        router_json = {
            "needs_rag": router_decision.needs_rag,
            "search_query": router_decision.search_query,
            "is_flagged": router_decision.is_flagged,
            "severity": router_decision.severity,
        }
        if router_decision.flag_reason:
            router_json["flag_reason"] = router_decision.flag_reason
        if router_decision.fallback_reason:
            router_json["fallback_reason"] = router_decision.fallback_reason
    rag_json = None
    if outcome is not None:
        rag_json = {
            "policy_version": POLICY_VERSION,
            "permitted_chunks": outcome.permitted,
            "chunk_ids": [
                {
                    "id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "score": chunk.score,
                }
                for chunk in outcome.chunks
            ],
            "dropped_chunk_ids": outcome.dropped_chunk_ids,
            "embedding_engine": outcome.embedding_engine,
            "engine_mismatch": outcome.engine_mismatch,
            "unauthorized_attempt": outcome.unauthorized_attempt,
            "restricted_top_score": outcome.restricted_top_score,
            "restricted_match_ids": outcome.restricted_match_ids,
        }
    llm_json = None
    if llm_response is not None or llm_error is not None:
        llm_json = {
            "model": llm_response.model if llm_response else None,
            "finish_reason": llm_response.finish_reason if llm_response else None,
            "usage": llm_response.usage if llm_response else None,
            "latency_ms": llm_latency_ms,
            "answer_masked": llm_response.content if llm_response else None,
        }
        if llm_error:
            llm_json["error"] = llm_error
    demasking_json = {
        "restored_count": restored_count,
        "unmatched_count": unmatched_count,
    }

    row = AuditLog(
        username=user.username,
        role=subject.get("role"),
        status=status,
        masked_prompt=masked_prompt,
        guard=guard_json,
        masking=masking_json,
        router=router_json,
        rag=rag_json,
        llm=llm_json,
        demasking=demasking_json,
    )
    session.add(row)
    session.commit()
    logger.info("chat | audit row id=%s written (status=%s)", row.id, status)
    return row.id


def chat(
    session: Session,
    user: User,
    prompt: str,
    top_k: int | None = None,
) -> ChatResult:
    """Run the full guard -> mask -> route -> retrieve -> answer -> demask chain."""
    started = time.perf_counter()
    subject = subject_attributes(user)

    guarded = screen(prompt, reversible=True, user=user)
    if guarded.disposition == "REJECT":
        logger.info("chat | REJECT: pipeline halted before any LLM call")
        audit_id = _write_audit(
            session,
            user=user,
            subject=subject,
            status=STATUS_REJECTED,
            masked_prompt=None,
            guarded=guarded,
            router_decision=None,
            outcome=None,
            llm_response=None,
            llm_error=None,
            llm_latency_ms=0,
            restored_count=0,
            unmatched_count=0,
        )
        return _rejected_result(started, guarded, None, None, None, audit_id)

    masked_prompt = guarded.masked_prompt or prompt
    mapping = guarded.masking.mapping if guarded.masking is not None else None

    router_decision = llm_flagging(masked_prompt, subject)
    if router_decision.is_flagged:
        logger.info(
            "chat | REJECT: router flagged the prompt (severity=%s, reason=%s)",
            router_decision.severity,
            router_decision.flag_reason,
        )
        guarded = replace(
            guarded,
            disposition="LLM_REJECT",
            flagged=True,
            rules=[*guarded.rules, RULE_LLM_ROUTER],
        )
        audit_id = _write_audit(
            session,
            user=user,
            subject=subject,
            status=STATUS_REJECTED,
            masked_prompt=masked_prompt,
            guarded=guarded,
            router_decision=router_decision,
            outcome=None,
            llm_response=None,
            llm_error=None,
            llm_latency_ms=0,
            restored_count=0,
            unmatched_count=0,
        )
        details = {}
        if router_decision.fallback_reason:
            details["fallback_reason"] = router_decision.fallback_reason
        write_flag(
            layer=LAYER_LLM_ROUTER,
            disposition="LLM_REJECT",
            severity=router_decision.severity,
            reason=router_decision.flag_reason or "flagged by LLM router",
            rules=guarded.rules,
            user=user,
            prompt_masked=masked_prompt,
            details=details,
            audit_id=audit_id,
        )
        return _rejected_result(
            started, guarded, masked_prompt, router_decision, guarded.masking, audit_id
        )

    outcome = None
    if router_decision.needs_rag:
        outcome = retrieve(
            session, user, router_decision.search_query or masked_prompt, top_k
        )
        logger.info(
            "chat | retrieval: kept=%d dropped=%d permitted=%d mismatch=%s",
            len(outcome.chunks),
            len(outcome.dropped_chunk_ids),
            outcome.permitted,
            outcome.engine_mismatch,
        )
        if outcome.unauthorized_attempt:
            logger.warning(
                "chat | REJECT: query matches restricted content outside the "
                "user's scope (top_score=%.4f, docs=%s)",
                outcome.restricted_top_score,
                outcome.restricted_match_ids,
            )
            guarded = replace(
                guarded,
                disposition="ABAC_REJECT",
                flagged=True,
                rules=[*guarded.rules, "ABAC_UNAUTHORIZED_ATTEMPT"],
            )
            audit_id = _write_audit(
                session,
                user=user,
                subject=subject,
                status=STATUS_REJECTED,
                masked_prompt=masked_prompt,
                guarded=guarded,
                router_decision=router_decision,
                outcome=outcome,
                llm_response=None,
                llm_error=None,
                llm_latency_ms=0,
                restored_count=0,
                unmatched_count=0,
            )
            write_flag(
                layer=LAYER_ABAC,
                disposition="ABAC_REJECT",
                severity="high",
                reason=(
                    "query strongly matches restricted content outside the "
                    "requester's authorized scope"
                ),
                rules=guarded.rules,
                user=user,
                prompt_masked=masked_prompt,
                details={
                    "restricted_top_score": outcome.restricted_top_score,
                    "restricted_match_ids": outcome.restricted_match_ids,
                    "permitted_chunks": outcome.permitted,
                },
                audit_id=audit_id,
            )
            return _rejected_result(
                started, guarded, masked_prompt, router_decision, guarded.masking, audit_id
            )

    system = _ANSWER_SYSTEM_BASE
    if outcome is not None and outcome.chunks:
        system = _ANSWER_SYSTEM_CONTEXT + outcome.assembled_context

    answer_started = time.perf_counter()
    try:
        response = get_client().chat(
            [{"role": "user", "content": masked_prompt}],
            system=system,
            temperature=ANSWER_TEMPERATURE,
            max_tokens=ANSWER_MAX_TOKENS,
        )
    except LLMClientError as exc:
        llm_latency_ms = _elapsed_ms(answer_started)
        total_ms = _elapsed_ms(started)
        logger.warning("chat | answer LLM call failed after %dms", llm_latency_ms)
        audit_id = _write_audit(
            session,
            user=user,
            subject=subject,
            status=STATUS_LLM_ERROR,
            masked_prompt=masked_prompt,
            guarded=guarded,
            router_decision=router_decision,
            outcome=outcome,
            llm_response=None,
            llm_error=str(exc),
            llm_latency_ms=llm_latency_ms,
            restored_count=0,
            unmatched_count=0,
        )
        return ChatResult(
            status=STATUS_LLM_ERROR,
            answer_demasked=None,
            answer_masked=None,
            masked_prompt=masked_prompt,
            disposition=guarded.disposition,
            verdict=guarded.verdict,
            masking=guarded.masking,
            router=router_decision,
            chunks=outcome.chunks if outcome else [],
            citations=outcome.citations if outcome else [],
            llm_model=None,
            llm_usage=None,
            latency_ms=total_ms,
            audit_id=audit_id,
            error=str(exc),
        )

    llm_latency_ms = _elapsed_ms(answer_started)
    answer_masked = response.content
    answer_demasked = demask(answer_masked, mapping)
    tokens = [match.group(0) for match in _PLACEHOLDER_TOKEN.finditer(answer_masked)]
    placed = {entity.placeholder for entity in mapping or []}
    restored_count = sum(1 for token in tokens if token in placed)
    unmatched_count = len(placed - set(tokens))

    total_ms = _elapsed_ms(started)
    audit_id = _write_audit(
        session,
        user=user,
        subject=subject,
        status=STATUS_ANSWER,
        masked_prompt=masked_prompt,
        guarded=guarded,
        router_decision=router_decision,
        outcome=outcome,
        llm_response=response,
        llm_error=None,
        llm_latency_ms=llm_latency_ms,
        restored_count=restored_count,
        unmatched_count=unmatched_count,
    )
    logger.info(
        "chat | answer complete: status=%s citations=%d restored=%d/%d latency=%dms",
        STATUS_ANSWER,
        len(outcome.citations) if outcome else 0,
        restored_count,
        len(placed),
        total_ms,
    )
    return ChatResult(
        status=STATUS_ANSWER,
        answer_demasked=answer_demasked,
        answer_masked=answer_masked,
        masked_prompt=masked_prompt,
        disposition=guarded.disposition,
        verdict=guarded.verdict,
        masking=guarded.masking,
        router=router_decision,
        chunks=outcome.chunks if outcome else [],
        citations=outcome.citations if outcome else [],
        llm_model=response.model,
        llm_usage=response.usage,
        latency_ms=total_ms,
        audit_id=audit_id,
    )
