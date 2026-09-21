"""Ingest structured JSON sources into the RAG store: one record = one chunk.

Usage: ``python -m guard.ingest data\\medicines.json data\\patients.json``.
The source class (doc_type, sensitivity, text template) is selected by the
filename stem; re-running for the same source path replaces that document's
chunks and re-embeds them (idempotent at this scale).
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from guard.db import Chunk, Document, get_engine, init_db
from guard.logconf import setup_logging
from guard.steps.embedding import embed

logger = logging.getLogger("guard.ingest")


@dataclass(frozen=True)
class IngestClass:
    doc_type: str
    sensitivity: str
    title: str
    render: Callable[[dict], str]
    extra_attributes: Callable[[dict], dict]


def _render_medicine(record: dict) -> str:
    lines = [f"Medicine: {record['name']}", f"Usage: {record['usage']}"]
    if record.get("category"):
        lines.append(f"Category: {record['category']}")
    if record.get("dosage_form"):
        lines.append(f"Dosage form: {record['dosage_form']}")
    return "\n".join(lines)


def _medicine_attributes(record: dict) -> dict:
    attributes: dict[str, Any] = {}
    if record.get("category"):
        attributes["category"] = record["category"]
    if record.get("dosage_form"):
        attributes["dosage_form"] = record["dosage_form"]
    return attributes


def _render_patient(record: dict) -> str:
    lines = [
        f"Patient: {record['patient_name']}",
        f"Medicines used: {', '.join(record['medicines_used'])}",
    ]
    if record.get("prescribed_by"):
        lines.append(f"Prescribed by: {record['prescribed_by']}")
    if record.get("notes"):
        lines.append(f"Notes: {record['notes']}")
    return "\n".join(lines)


def _patient_attributes(record: dict) -> dict:
    attributes: dict[str, Any] = {}
    if record.get("prescribed_by"):
        attributes["prescribed_by"] = record["prescribed_by"]
    return attributes


INGEST_CLASSES: dict[str, IngestClass] = {
    "medicines": IngestClass(
        doc_type="medicine",
        sensitivity="public",
        title="Medicine catalog",
        render=_render_medicine,
        extra_attributes=_medicine_attributes,
    ),
    "patients": IngestClass(
        doc_type="patient_record",
        sensitivity="restricted",
        title="Patient usage records",
        render=_render_patient,
        extra_attributes=_patient_attributes,
    ),
}


@dataclass(frozen=True)
class IngestSummary:
    source_path: str
    document_id: int
    chunks: int
    embedding_engine: str


def _load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty JSON array of records")
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"{path}[{index}] is not a JSON object")
    return data


def ingest_document(session: Session, path: str | Path) -> IngestSummary:
    """Replace the document (and chunks) for one source path, then re-embed."""
    path = Path(path)
    klass = INGEST_CLASSES.get(path.stem)
    if klass is None:
        known = ", ".join(sorted(INGEST_CLASSES))
        raise ValueError(f"unknown source class for '{path.name}' (known stems: {known})")
    records = _load_records(path)
    source_path = path.as_posix()

    for existing in session.scalars(
        select(Document).where(Document.source_path == source_path)
    ):
        session.delete(existing)
    session.flush()

    document = Document(
        title=klass.title,
        source_path=source_path,
        doc_type=klass.doc_type,
        attributes={"sensitivity": klass.sensitivity},
    )
    engine = ""
    for ordinal, record in enumerate(records, start=1):
        text = klass.render(record)
        vector, engine, model = embed(text)
        document.chunks.append(
            Chunk(
                ordinal=ordinal,
                text=text,
                sensitivity=klass.sensitivity,
                doc_type=klass.doc_type,
                attributes=klass.extra_attributes(record),
                embedding=vector,
                embedding_engine=engine,
                embedding_model=model,
            )
        )
    session.add(document)
    session.commit()
    logger.info(
        "ingest | %s -> %d chunks (doc_type=%s sensitivity=%s engine=%s)",
        source_path,
        len(records),
        klass.doc_type,
        klass.sensitivity,
        engine,
    )
    return IngestSummary(source_path, document.id, len(records), engine)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] in {"-h", "--help"}:
        known = " ".join(sorted(INGEST_CLASSES))
        print("usage: python -m guard.ingest <source.json> [...]")
        print(f"known source stems: {known}")
        return 0 if args else 1
    engine = get_engine()
    init_db(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    failures = 0
    for arg in args:
        try:
            with factory() as session:
                summary = ingest_document(session, arg)
            print(
                f"ingested {summary.source_path}: {summary.chunks} chunks "
                f"(document_id={summary.document_id}, engine={summary.embedding_engine})"
            )
        except Exception as exc:
            logger.error("ingest | failed for %s: %s", arg, exc)
            print(f"failed: {arg}: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
