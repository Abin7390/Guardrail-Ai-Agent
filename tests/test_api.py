import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from guard import api
from guard.db import AuditLog, get_session, init_db
from guard.pipeline import GuardResult
from guard.steps.masking import MaskingResult
from guard.steps.prompt_guard import PromptGuardVerdict


@pytest.fixture
def flag_log(tmp_path, monkeypatch):
    path = tmp_path / "flags.jsonl"
    monkeypatch.setenv("GUARD_FLAG_LOG", str(path))
    return path


@pytest.fixture
def db_session_factory():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    yield factory
    engine.dispose()


@pytest.fixture
def client(monkeypatch, flag_log, db_session_factory):
    monkeypatch.setenv("GUARD_WARMUP", "0")

    def override_get_session():
        session = db_session_factory()
        try:
            yield session
        finally:
            session.close()

    api.app.dependency_overrides[get_session] = override_get_session
    monkeypatch.setattr(api, "init_db", lambda: [])
    with TestClient(api.app) as test_client:
        yield test_client
    api.app.dependency_overrides.clear()


def get_token(client, username):
    response = client.post("/v1/token", json={"username": username})
    assert response.status_code == 200
    return response.json()["access_token"]


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _verdict(flagged: bool) -> PromptGuardVerdict:
    if flagged:
        return PromptGuardVerdict(True, "SUSPICIOUS", 0.05, 0.99, "hf")
    return PromptGuardVerdict(False, "BENIGN", 0.99, 0.01, "hf")


def test_health(client):
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "prompt_guard_engine" in body and "masking_engine" in body


def test_token_all_users(client):
    roles = {"admin": "admin", "user1": "user", "user2": "user"}
    for username, role in roles.items():
        response = client.post("/v1/token", json={"username": username})
        assert response.status_code == 200
        body = response.json()
        assert body["access_token"]
        assert body["token_type"] == "bearer"
        assert body["expires_in"] > 0
        assert body["user"]["username"] == username
        assert body["user"]["role"] == role
        assert body["user"]["email"] == f"{username}@guardrail.local"


def test_token_unknown_username(client):
    response = client.post("/v1/token", json={"username": "nobody"})
    assert response.status_code == 422


def test_users_me(client):
    token = get_token(client, "user2")
    response = client.get("/v1/users/me", headers=bearer(token))
    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "user2"
    assert body["role"] == "user"
    assert body["email"] == "user2@guardrail.local"
    assert body["full_name"] == "User Two"


def test_users_me_requires_auth(client):
    response = client.get("/v1/users/me")
    assert response.status_code == 401


def test_users_forbidden_for_regular_user(client):
    token = get_token(client, "user1")
    response = client.get("/v1/users", headers=bearer(token))
    assert response.status_code == 403
    assert response.json()["detail"] == "Admin privileges required"


def test_users_admin_lists_all(client):
    token = get_token(client, "admin")
    response = client.get("/v1/users", headers=bearer(token))
    assert response.status_code == 200
    users = response.json()
    assert [u["username"] for u in users] == ["admin", "user1", "user2"]
    assert all({"id", "role", "email", "full_name", "created_at"} <= set(u) for u in users)


def test_audit_forbidden_for_regular_user(client):
    token = get_token(client, "user1")
    response = client.get("/v1/audit", headers=bearer(token))
    assert response.status_code == 403
    assert response.json()["detail"] == "Admin privileges required"


def test_audit_admin_lists_rows_newest_first(client, db_session_factory):
    with db_session_factory() as session:
        session.add_all(
            [
                AuditLog(
                    username="user1",
                    role="user",
                    status="ANSWER",
                    masked_prompt="hi [REDACTED_1]",
                    guard={"disposition": "MASKED", "rules": ["PII_DETECTED"]},
                    masking={"engine": "presidio", "entities": {"PERSON": 1}, "placeholder_count": 1},
                    router={"needs_rag": False, "search_query": ""},
                    demasking={"restored_count": 1, "unmatched_count": 0},
                ),
                AuditLog(
                    username="admin",
                    role="admin",
                    status="REJECTED",
                    masked_prompt=None,
                    guard={"disposition": "REJECT", "rules": ["PROMPT_GUARD_SUSPICIOUS"]},
                ),
            ]
        )
        session.commit()
    token = get_token(client, "admin")
    response = client.get("/v1/audit", headers=bearer(token))
    assert response.status_code == 200
    rows = response.json()
    assert [row["status"] for row in rows] == ["REJECTED", "ANSWER"]
    assert rows[0]["username"] == "admin"
    assert rows[1]["masked_prompt"] == "hi [REDACTED_1]"
    assert rows[1]["rag"] is None and rows[1]["llm"] is None
    assert rows[1]["masking"]["placeholder_count"] == 1


def test_audit_limit_bounds_validated(client):
    token = get_token(client, "admin")
    response = client.get("/v1/audit?limit=0", headers=bearer(token))
    assert response.status_code == 422
    response = client.get("/v1/audit?limit=101", headers=bearer(token))
    assert response.status_code == 422


_UPLOAD_FILES_REPORT = [
    {
        "filename": "notes.txt",
        "extension": ".txt",
        "size_bytes": 22,
        "label": "BENIGN",
        "suspicious_score": 0.01,
        "engine": "regex-fallback",
        "entities": {},
    }
]


def _plain_guard_for_upload_tests(raw, reversible=False, user=None):
    return GuardResult(
        "CLEAN", False, [], _verdict(False), MaskingResult(raw, {}, "presidio"), raw
    )


class _FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, *, system=None, temperature=None, max_tokens=None, **kwargs):
        self.calls.append({"messages": messages, "system": system})
        return self.responses.pop(0)


def test_upload_chat_with_files_end_to_end(client, db_session_factory, monkeypatch):
    import guard.chat as chat_module
    import guard.steps.llm_flagging as llm_flagging_module
    from guard.llm import LLMResponse

    def _upload_guard(raw, files=None, reversible=False, user=None):
        assert files, "chat must forward the extracted files to screen_request"
        assert files[0].text == "attached file content"
        return GuardResult(
            "CLEAN",
            False,
            [],
            _verdict(False),
            MaskingResult(raw, {}, "presidio"),
            raw,
            _UPLOAD_FILES_REPORT,
            [("notes.txt", "attached file content")],
        )

    fake = _FakeLLM(
        [
            LLMResponse(
                '{"is_flagged": false, "flag_reason": "", "severity": "none",'
                ' "needs_rag": false, "search_query": ""}',
                "test-model",
                "stop",
                {},
            ),
            LLMResponse("summary of notes", "test-model", "stop", {}),
        ]
    )
    monkeypatch.setattr(chat_module, "screen_request", _upload_guard)
    monkeypatch.setattr(chat_module, "get_client", lambda: fake)
    monkeypatch.setattr(llm_flagging_module, "get_client", lambda: fake)
    token = get_token(client, "user1")
    response = client.post(
        "/v1/chat/upload",
        data={"prompt": "summarize my notes"},
        files=[("file", ("notes.txt", b"attached file content", "text/plain"))],
        headers=bearer(token),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ANSWER"
    assert body["answer"] == "summary of notes"
    assert body["files"][0]["filename"] == "notes.txt"
    assert body["files"][0]["verdict_label"] == "BENIGN"
    assert body["files"][0]["entities"] == {}
    assert "[ATTACHED FILE: notes.txt]" in fake.calls[0]["messages"][0]["content"]
    assert "[ATTACHED FILE: notes.txt]" in fake.calls[1]["messages"][0]["content"]
    with db_session_factory() as session:
        row = session.scalar(select(AuditLog).order_by(AuditLog.id.desc()).limit(1))
        assert row is not None and row.status == "ANSWER"
        assert row.guard["files"] == _UPLOAD_FILES_REPORT
        assert row.masked_prompt == "summarize my notes"


def test_upload_unsupported_file_rejected_422(client, db_session_factory):
    token = get_token(client, "user1")
    response = client.post(
        "/v1/chat/upload",
        data={"prompt": "hello"},
        files=[("file", ("payload.exe", b"MZ", "application/octet-stream"))],
        headers=bearer(token),
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["files"][0]["filename"] == "payload.exe"
    assert "unsupported file type" in detail["files"][0]["error"]
    with db_session_factory() as session:
        assert session.scalar(select(AuditLog).order_by(AuditLog.id.desc()).limit(1)) is None


def test_upload_corrupt_pdf_rejected_422(client):
    token = get_token(client, "user1")
    response = client.post(
        "/v1/chat/upload",
        data={"prompt": "hello"},
        files=[("file", ("broken.pdf", b"%PDF-1.4 junk", "application/pdf"))],
        headers=bearer(token),
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["files"][0]["filename"] == "broken.pdf"


def test_upload_without_file_uses_plain_chat_path(client, db_session_factory, monkeypatch):
    import guard.chat as chat_module
    import guard.steps.llm_flagging as llm_flagging_module
    from guard.llm import LLMResponse

    monkeypatch.setattr(chat_module, "screen", _plain_guard_for_upload_tests)
    fake = _FakeLLM(
        [
            LLMResponse(
                '{"is_flagged": false, "flag_reason": "", "severity": "none",'
                ' "needs_rag": false, "search_query": ""}',
                "test-model",
                "stop",
                {},
            ),
            LLMResponse("plain answer", "test-model", "stop", {}),
        ]
    )
    monkeypatch.setattr(chat_module, "get_client", lambda: fake)
    monkeypatch.setattr(llm_flagging_module, "get_client", lambda: fake)
    token = get_token(client, "user1")
    response = client.post("/v1/chat/upload", data={"prompt": "hello"}, headers=bearer(token))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ANSWER"
    assert body["files"] is None


def test_upload_requires_auth(client):
    response = client.post(
        "/v1/chat/upload",
        data={"prompt": "hello"},
        files=[("file", ("a.txt", b"x", "text/plain"))],
    )
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_upload_empty_prompt_validation_error(client):
    token = get_token(client, "user1")
    response = client.post(
        "/v1/chat/upload",
        data={"prompt": ""},
        files=[("file", ("a.txt", b"x", "text/plain"))],
        headers=bearer(token),
    )
    assert response.status_code == 422
