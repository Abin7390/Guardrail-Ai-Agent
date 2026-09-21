import json
import sys

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import guard.steps.embedding as embedding
from guard.db import Chunk, Document, init_db
from guard.ingest import IngestClass, ingest_document, main

MEDICINES = [
    {"name": "Amoxicillin", "usage": "Antibiotic for bacterial infections", "category": "antibiotic"},
    {"name": "Paracetamol", "usage": "Pain reliever and fever reducer", "category": "analgesic", "dosage_form": "tablet"},
]
PATIENTS = [
    {"patient_name": "John Mercer", "medicines_used": ["Amoxicillin"], "prescribed_by": "Dr. Alice Feldman"},
    {"patient_name": "Priya Nair", "medicines_used": ["Metformin", "Atorvastatin"]},
    {"patient_name": "Luis Ortega", "medicines_used": ["Omeprazole"], "notes": "morning dose"},
]


@pytest.fixture
def forced_fallback(monkeypatch):
    monkeypatch.setattr(embedding, "_model", None)
    monkeypatch.setattr(embedding, "_engine_mode", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    return None


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        yield db
    engine.dispose()


@pytest.fixture
def sources(tmp_path, forced_fallback):
    medicines = tmp_path / "medicines.json"
    medicines.write_text(json.dumps(MEDICINES), encoding="utf-8")
    patients = tmp_path / "patients.json"
    patients.write_text(json.dumps(PATIENTS), encoding="utf-8")
    return medicines, patients


def test_entity_per_chunk_and_attribute_stamping(session, sources):
    medicines, patients = sources
    ingest_document(session, medicines)
    ingest_document(session, patients)

    docs = session.scalars(select(Document).order_by(Document.id)).all()
    assert [d.doc_type for d in docs] == ["medicine", "patient_record"]
    assert [d.attributes["sensitivity"] for d in docs] == ["public", "restricted"]

    chunks = session.scalars(select(Chunk).order_by(Chunk.id)).all()
    assert len(chunks) == len(MEDICINES) + len(PATIENTS)
    assert [c.ordinal for c in chunks[: len(MEDICINES)]] == [1, 2]

    med_chunk = chunks[0]
    assert med_chunk.sensitivity == "public"
    assert med_chunk.doc_type == "medicine"
    assert "Amoxicillin" in med_chunk.text
    assert med_chunk.attributes.get("category") == "antibiotic"
    assert len(med_chunk.embedding) == 384
    assert med_chunk.embedding_engine == "fallback"

    patient_chunk = chunks[len(MEDICINES) :]
    assert all(c.sensitivity == "restricted" for c in patient_chunk)
    assert all(c.doc_type == "patient_record" for c in patient_chunk)
    assert "John Mercer" in patient_chunk[0].text
    assert patient_chunk[1].attributes == {}


def test_idempotent_reingest(session, sources):
    medicines, patients = sources
    ingest_document(session, medicines)
    ingest_document(session, patients)
    first_doc_ids = session.scalars(select(Document.id).order_by(Document.id)).all()
    ingest_document(session, medicines)
    ingest_document(session, patients)

    doc_count = session.scalar(select(func.count(Document.id)))
    chunk_count = session.scalar(select(func.count(Chunk.id)))
    assert doc_count == 2
    assert chunk_count == len(MEDICINES) + len(PATIENTS)
    new_doc_ids = session.scalars(select(Document.id).order_by(Document.id)).all()
    assert len(new_doc_ids) == 2
    assert new_doc_ids != first_doc_ids or len(first_doc_ids) == 2


def test_unknown_class_rejected(session, tmp_path):
    other = tmp_path / "invoices.json"
    other.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown source class"):
        ingest_document(session, other)


def test_rejects_non_array(tmp_path):
    bad = tmp_path / "medicines.json"
    bad.write_text('{"name": "oops"}', encoding="utf-8")
    engine = create_engine("sqlite+pysqlite:///:memory:")
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        with pytest.raises(ValueError, match="non-empty JSON array"):
            ingest_document(db, bad)
    engine.dispose()


def test_main_ingests_all(monkeypatch, sources, tmp_path):
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    monkeypatch.setattr("guard.ingest.get_engine", lambda: engine)
    medicines, patients = sources
    try:
        assert main([str(medicines), str(patients)]) == 0
        with sessionmaker(bind=engine)() as db:
            assert db.scalar(select(func.count(Chunk.id))) == len(MEDICINES) + len(PATIENTS)
    finally:
        engine.dispose()


def test_main_reports_failures(monkeypatch, tmp_path):
    engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool)
    monkeypatch.setattr("guard.ingest.get_engine", lambda: engine)
    missing = tmp_path / "medicines.json"
    try:
        assert main([str(missing)]) == 1
    finally:
        engine.dispose()


def test_render_medicine_template():
    from guard.ingest import INGEST_CLASSES

    render = INGEST_CLASSES["medicines"].render
    text = render({"name": "Aspirin", "usage": "blood thinner", "category": "analgesic", "dosage_form": "tablet"})
    assert text == "Medicine: Aspirin\nUsage: blood thinner\nCategory: analgesic\nDosage form: tablet"


def test_render_patient_template():
    from guard.ingest import INGEST_CLASSES

    render = INGEST_CLASSES["patients"].render
    text = render({"patient_name": "Ann Doe", "medicines_used": ["Aspirin", "Metformin"], "prescribed_by": "Dr. X", "notes": "stable"})
    assert text == (
        "Patient: Ann Doe\nMedicines used: Aspirin, Metformin\n"
        "Prescribed by: Dr. X\nNotes: stable"
    )
