"""FastAPI service exposing the guard pipeline through a public router."""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from guard.auth import (
    create_access_token,
    get_current_user,
    get_expire_minutes,
    require_admin,
)
from guard.chat import STATUS_LLM_ERROR, STATUS_REJECTED, ChatResult, chat
from guard.db import AuditLog, Chunk, Document, User, get_session, init_db
from guard.logconf import setup_logging
from guard.rag import AskResult, ask
from guard.steps.embedding import get_engine_mode as embedding_engine_mode
from guard.steps.embedding import embed
from guard.steps.masking import get_engine_mode as masking_engine
from guard.steps.masking import MaskingResult
from guard.pipeline import GuardResult, screen
from guard.steps.prompt_guard import get_engine_mode as prompt_guard_engine
from guard.steps.prompt_guard import PromptGuardVerdict

logger = logging.getLogger("guard.api")

REJECT_MESSAGE = "Your request was blocked: jailbreak or prompt-injection detected."
MASKED_MESSAGE = "Your request was blocked: PII detected."
CLEAN_MESSAGE = "Screening passed; prompt forwarded as-is."

public_router = APIRouter(prefix="/v1", tags=["public"])


class ScreenRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)


class VerdictOut(BaseModel):
    label: str
    benign_score: float
    suspicious_score: float
    engine: str


class MaskingOut(BaseModel):
    entities: dict[str, int]
    engine: str


class ScreenResponse(BaseModel):
    disposition: str
    flagged: bool
    rules: list[str]
    message: str
    masked_prompt: str | None
    verdict: VerdictOut | None
    masking: MaskingOut | None


class TokenRequestUser(str, Enum):
    admin = "admin"
    user1 = "user1"
    user2 = "user2"


class TokenRequest(BaseModel):
    username: TokenRequestUser


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    role: str
    email: str | None
    full_name: str | None
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class RetrievedChunkOut(BaseModel):
    chunk_id: int
    document_id: int
    ordinal: int
    title: str
    text: str
    score: float


class CitationOut(BaseModel):
    document_id: int
    chunk_id: int
    title: str
    score: float


class AskResponse(BaseModel):
    disposition: str
    message: str
    chunks: list[RetrievedChunkOut]
    assembled_context: str | None
    citations: list[CitationOut]
    policy_version: str
    embedding_engine: str | None
    engine_mismatch: bool


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class ChatResponse(BaseModel):
    status: str
    disposition: str
    answer: str | None
    message: str
    citations: list[CitationOut]
    verdict: VerdictOut | None
    masking: MaskingOut | None
    used_rag: bool
    audit_id: int | None


class AuditOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ts: datetime
    username: str
    role: str
    status: str
    masked_prompt: str | None
    guard: dict | None
    masking: dict | None
    router: dict | None
    rag: dict | None
    llm: dict | None
    demasking: dict | None


class DocumentOut(BaseModel):
    id: int
    title: str
    source_path: str
    doc_type: str
    attributes: dict
    chunk_count: int
    created_at: datetime


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        seeded = init_db()
        logger.info(
            "api | db ready (seeded: %s)",
            ", ".join(seeded) if seeded else "none; users already present",
        )
    except Exception:
        logger.exception(
            "api | database init failed; check GUARD_DATABASE_URL and that Postgres "
            "is reachable (init_db creates tables, not the database itself)"
        )
        raise
    if os.environ.get("GUARD_WARMUP", "1") != "0":
        logger.info("api | warming up engines (loads Prompt-Guard + Presidio + embedder)...")
        screen("warmup")
        embed("warmup")
        logger.info("api | engines warm; ready to serve")
    yield


@public_router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "prompt_guard_engine": prompt_guard_engine(),
        "masking_engine": masking_engine(),
        "embedding_engine": embedding_engine_mode(),
    }


@public_router.post("/token", response_model=TokenOut)
def issue_token(request: TokenRequest, session: Session = Depends(get_session)) -> TokenOut:
    user = session.scalar(select(User).where(User.username == request.username.value))
    if user is None:
        raise HTTPException(
            status_code=404,
            detail=f"User '{request.username.value}' not found; database not seeded",
        )
    return TokenOut(
        access_token=create_access_token(user.username, user.role),
        expires_in=get_expire_minutes() * 60,
        user=UserOut.model_validate(user),
    )


@public_router.get("/users/me", response_model=UserOut)
def read_current_user(current_user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.model_validate(current_user)


@public_router.get("/users", response_model=list[UserOut])
def list_users(
    session: Session = Depends(get_session),
    _admin: User = Depends(require_admin),
) -> list[UserOut]:
    users = session.scalars(select(User).order_by(User.id)).all()
    return [UserOut.model_validate(user) for user in users]


@public_router.post("/screen", response_model=ScreenResponse)
def screen_prompt(
    request: ScreenRequest,
    current_user: User = Depends(get_current_user),
) -> ScreenResponse:
    result = screen(request.prompt)
    message = {
        "REJECT": REJECT_MESSAGE,
        "MASKED": MASKED_MESSAGE,
        "CLEAN": CLEAN_MESSAGE,
    }[result.disposition]
    return ScreenResponse(
        disposition=result.disposition,
        flagged=result.flagged,
        rules=result.rules,
        message=message,
        masked_prompt=result.masked_prompt,
        verdict=_verdict_out(result.verdict),
        masking=_masking_out(result.masking),
    )


@public_router.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> AskResponse:
    result = ask(session, current_user, request.question, request.top_k)
    return _ask_response(result)


@public_router.post("/chat", response_model=ChatResponse)
def chat_prompt(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> ChatResponse:
    result = chat(session, current_user, request.prompt, request.top_k)
    if result.status == STATUS_LLM_ERROR:
        raise HTTPException(
            status_code=502,
            detail=result.error or "LLM answer call failed",
        )
    return _chat_response(result)


@public_router.get("/audit", response_model=list[AuditOut])
def list_audit(
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_session),
    _admin: User = Depends(require_admin),
) -> list[AuditOut]:
    rows = session.scalars(
        select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
    ).all()
    return [AuditOut.model_validate(row) for row in rows]


@public_router.get("/documents", response_model=list[DocumentOut])
def list_documents(
    session: Session = Depends(get_session),
    _admin: User = Depends(require_admin),
) -> list[DocumentOut]:
    rows = session.execute(
        select(Document, func.count(Chunk.id))
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .group_by(Document.id)
        .order_by(Document.id)
    ).all()
    return [
        DocumentOut(
            id=document.id,
            title=document.title,
            source_path=document.source_path,
            doc_type=document.doc_type,
            attributes=document.attributes or {},
            chunk_count=count,
            created_at=document.created_at,
        )
        for document, count in rows
    ]


def _ask_response(result: AskResult) -> AskResponse:
    return AskResponse(
        disposition=result.disposition,
        message=result.message,
        chunks=[
            RetrievedChunkOut(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                ordinal=chunk.ordinal,
                title=chunk.title,
                text=chunk.text,
                score=chunk.score,
            )
            for chunk in result.chunks
        ],
        assembled_context=result.assembled_context,
        citations=[
            CitationOut(
                document_id=citation.document_id,
                chunk_id=citation.chunk_id,
                title=citation.title,
                score=citation.score,
            )
            for citation in result.citations
        ],
        policy_version=result.policy_version,
        embedding_engine=result.embedding_engine,
        engine_mismatch=result.engine_mismatch,
    )


def _chat_response(result: ChatResult) -> ChatResponse:
    if result.status == STATUS_REJECTED:
        if result.disposition == "REJECT":
            message = REJECT_MESSAGE
        else:
            message = MASKED_MESSAGE
    elif result.chunks:
        message = f"Answer generated with {len(result.chunks)} retrieved chunk(s)."
    else:
        message = "Answer generated."
    return ChatResponse(
        status=result.status,
        disposition=result.disposition,
        answer=result.answer_demasked,
        message=message,
        citations=[
            CitationOut(
                document_id=citation.document_id,
                chunk_id=citation.chunk_id,
                title=citation.title,
                score=citation.score,
            )
            for citation in result.citations
        ],
        verdict=_verdict_out(result.verdict),
        masking=_masking_out(result.masking),
        used_rag=bool(result.router and result.router.needs_rag),
        audit_id=result.audit_id,
    )


def _verdict_out(verdict: PromptGuardVerdict | None) -> VerdictOut | None:
    if verdict is None:
        return None
    return VerdictOut(
        label=verdict.label,
        benign_score=verdict.benign_score,
        suspicious_score=verdict.suspicious_score,
        engine=verdict.engine,
    )


def _masking_out(masking: MaskingResult | None) -> MaskingOut | None:
    if masking is None:
        return None
    return MaskingOut(entities=masking.entities, engine=masking.engine)


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(
        title="Custom Guardrail Service",
        description=(
            "Prompt-Guard jailbreak check + Presidio PII masking + ABAC-filtered "
            "RAG retrieval + unified chat (mask -> route -> retrieve -> LLM -> demask)"
        ),
        version="1.1.0",
        lifespan=lifespan,
    )
    app.include_router(public_router)
    return app


app = create_app()


def main() -> None:
    import uvicorn

    setup_logging()
    uvicorn.run(
        "guard.api:app",
        host=os.environ.get("GUARD_HOST", "127.0.0.1"),
        port=int(os.environ.get("GUARD_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
