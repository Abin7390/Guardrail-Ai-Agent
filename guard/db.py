"""Postgres-backed storage for the Guardrail API (SQLAlchemy 2.0, sync).

The engine and session factory are created lazily from ``GUARD_DATABASE_URL``
so tests and env overrides can swap the database after import. JSON columns
use ``JSONB`` on Postgres and plain ``JSON`` on SQLite (offline tests).
"""

import os
from collections.abc import Iterator
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text, create_engine, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy import JSON
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5433/guardrail_poc"

JSONColumn = JSONB().with_variant(JSON(), "sqlite")

SEED_USERS = [
    {"username": "admin", "role": "admin", "email": "admin@guardrail.local", "full_name": "Admin User"},
    {"username": "user1", "role": "user", "email": "user1@guardrail.local", "full_name": "User One"},
    {"username": "user2", "role": "user", "email": "user2@guardrail.local", "full_name": "User Two"},
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255))
    full_name: Mapped[str | None] = mapped_column(String(255))
    attributes: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"User(id={self.id!r}, username={self.username!r}, role={self.role!r})"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    doc_type: Mapped[str] = mapped_column(String(50), nullable=False)
    attributes: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Chunk.ordinal"
    )

    def __repr__(self) -> str:
        return f"Document(id={self.id!r}, title={self.title!r}, doc_type={self.doc_type!r})"


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    sensitivity: Mapped[str] = mapped_column(String(20), nullable=False)
    doc_type: Mapped[str] = mapped_column(String(50), nullable=False)
    attributes: Mapped[dict] = mapped_column(JSONColumn, default=dict, nullable=False)
    embedding: Mapped[list] = mapped_column(JSONColumn, nullable=False)
    embedding_engine: Mapped[str] = mapped_column(String(20), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")

    def __repr__(self) -> str:
        return (
            f"Chunk(id={self.id!r}, document_id={self.document_id!r}, "
            f"ordinal={self.ordinal!r}, doc_type={self.doc_type!r}, "
            f"sensitivity={self.sensitivity!r}, engine={self.embedding_engine!r})"
        )


class AuditLog(Base):
    """One row per unified chat request; masked content only.

    Invariants: never the raw prompt (``masked_prompt`` stores the masked
    text), never the demasked answer (``llm["answer_masked"]`` is the LLM
    output before demasking), never mapping values - only ids/counts/labels.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False
    )
    username: Mapped[str] = mapped_column(String(50), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    masked_prompt: Mapped[str | None] = mapped_column(Text)
    guard: Mapped[dict | None] = mapped_column(JSONColumn)
    masking: Mapped[dict | None] = mapped_column(JSONColumn)
    router: Mapped[dict | None] = mapped_column(JSONColumn)
    rag: Mapped[dict | None] = mapped_column(JSONColumn)
    llm: Mapped[dict | None] = mapped_column(JSONColumn)
    demasking: Mapped[dict | None] = mapped_column(JSONColumn)

    def __repr__(self) -> str:
        return (
            f"AuditLog(id={self.id!r}, username={self.username!r}, "
            f"role={self.role!r}, status={self.status!r})"
        )


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _engine, _session_factory
    if _engine is None:
        url = os.environ.get("GUARD_DATABASE_URL", DEFAULT_DATABASE_URL)
        _engine = create_engine(url, pool_pre_ping=True)
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_session() -> Iterator[Session]:
    get_engine()
    session = _session_factory()
    try:
        yield session
    finally:
        session.close()


def init_db(engine: Engine | None = None) -> list[str]:
    """Create tables and seed the mock users; returns newly seeded usernames."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    seeded: list[str] = []
    with factory() as session:
        for seed in SEED_USERS:
            if session.scalar(select(User).where(User.username == seed["username"])) is None:
                session.add(User(**seed))
                seeded.append(seed["username"])
        session.commit()
    return seeded
