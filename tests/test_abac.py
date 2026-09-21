import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from guard.abac import (
    RETRIEVAL_POLICY,
    And,
    Eq,
    In,
    Or,
    compile_sql,
    evaluate,
    retrieval_predicate,
    subject_attributes,
)
from guard.db import Chunk, Document, User, init_db


def test_evaluate_eq_and_in():
    subject = {"role": "admin"}
    resource = {"sensitivity": "restricted", "doc_type": "patient_record"}
    assert evaluate(Eq("subject.role", "admin"), subject, resource) is True
    assert evaluate(Eq("subject.role", "user"), subject, resource) is False
    assert evaluate(In("resource.doc_type", ["medicine", "patient_record"]), subject, resource) is True
    assert evaluate(In("resource.doc_type", ["medicine"]), subject, resource) is False


def test_evaluate_and_or_nesting():
    cond = Or(
        (
            And((Eq("resource.sensitivity", "public"), Eq("resource.doc_type", "medicine"))),
            Eq("subject.role", "admin"),
        )
    )
    assert evaluate(cond, {"role": "user"}, {"sensitivity": "public", "doc_type": "medicine"})
    assert evaluate(cond, {"role": "admin"}, {"sensitivity": "restricted", "doc_type": "patient_record"})
    assert not evaluate(cond, {"role": "user"}, {"sensitivity": "restricted", "doc_type": "patient_record"})


def test_deny_by_default_when_nothing_matches():
    subject = {"role": "user"}
    resource = {"sensitivity": "confidential", "doc_type": "internal"}
    assert evaluate(RETRIEVAL_POLICY, subject, resource) is False
    subject_none: dict = {}
    assert evaluate(RETRIEVAL_POLICY, subject_none, {"sensitivity": "restricted"}) is False
    assert evaluate(RETRIEVAL_POLICY, {"role": "admin"}, {"sensitivity": "restricted"}) is True


def test_unknown_scope_raises():
    with pytest.raises(ValueError):
        evaluate(Eq("foo.role", "admin"), {}, {})
    with pytest.raises(ValueError):
        compile_sql(Eq("bar.sensitivity", "public"), {})
    with pytest.raises(TypeError):
        evaluate("not-a-condition", {}, {})


def test_subject_attributes_merges_role_and_attributes():
    user = User(username="user1", role="user", attributes={"department": "cardiology"})
    assert subject_attributes(user) == {"role": "user", "department": "cardiology"}


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


def _seed_chunks(session):
    doc = Document(title="mixed", source_path="data/mixed.json", doc_type="mixed")
    specs = [
        ("public", "medicine"),
        ("public", "guide"),
        ("restricted", "patient_record"),
        ("confidential", "internal"),
    ]
    for ordinal, (sensitivity, doc_type) in enumerate(specs, start=1):
        doc.chunks.append(
            Chunk(
                ordinal=ordinal,
                text=f"{sensitivity} {doc_type}",
                sensitivity=sensitivity,
                doc_type=doc_type,
                embedding=[0.0],
                embedding_engine="fallback",
                embedding_model="test",
            )
        )
    session.add(doc)
    session.commit()
    return session.scalars(select(Chunk).order_by(Chunk.id)).all()


def test_differential_sql_matches_evaluator(session):
    chunks = _seed_chunks(session)
    for role in ("admin", "user", None):
        subject = {} if role is None else {"role": role}
        predicate = retrieval_predicate(subject)
        sql_ids = set(
            session.scalars(select(Chunk.id).where(predicate)).all()
        )
        eval_ids = {
            chunk.id
            for chunk in chunks
            if evaluate(RETRIEVAL_POLICY, subject, {"sensitivity": chunk.sensitivity, "doc_type": chunk.doc_type})
        }
        assert sql_ids == eval_ids, f"mismatch for role={role!r}"
        if role == "admin":
            assert len(sql_ids) == len(chunks)
        elif role == "user":
            assert {c.sensitivity for c in chunks if c.id in sql_ids} == {"public"}
        else:
            assert {c.sensitivity for c in chunks if c.id in sql_ids} == {"public"}


def test_retrieval_predicate_no_admin_leak_of_confidential(session):
    chunks = _seed_chunks(session)
    user_ids = set(
        session.scalars(
            select(Chunk.id).where(retrieval_predicate({"role": "user"}))
        ).all()
    )
    confidential = {c.id for c in chunks if c.sensitivity != "public"}
    assert user_ids.isdisjoint(confidential)


def test_compile_sql_folds_subject_to_literals(session):
    chunks = _seed_chunks(session)
    admin_ids = set(
        session.scalars(select(Chunk.id).where(compile_sql(RETRIEVAL_POLICY, {"role": "admin"}))).all()
    )
    all_ids = {c.id for c in chunks}
    assert admin_ids == all_ids
