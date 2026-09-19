"""Offline unit tests for the router/guard LLM flagging step (no API calls)."""

import json

import pytest

import guard.steps.llm_flagging as lf
from guard.llm import LLMClientError, LLMResponse


def _response(payload: dict) -> LLMResponse:
    return LLMResponse(
        json.dumps(payload), "gemini-3.8-flash", "stop", {"total_tokens": 18}
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, *, system=None, temperature=None, max_tokens=None, config=None, **kw):
        self.calls.append(
            {
                "messages": messages,
                "system": system,
                "max_tokens": max_tokens,
                "config": config,
            }
        )
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def subject():
    return {"role": "user"}


def test_parse_returns_all_security_fields():
    payload = lf._parse_router_payload(
        json.dumps(
            {
                "is_flagged": True,
                "flag_reason": "asks for confidential data",
                "severity": "HIGH",
                "needs_rag": True,
                "search_query": "ignored when flagged",
            }
        )
    )
    assert payload == {
        "is_flagged": True,
        "flag_reason": "asks for confidential data",
        "severity": "high",
        "needs_rag": False,
        "search_query": "",
    }


def test_parse_defaults_when_security_fields_missing():
    payload = lf._parse_router_payload(
        json.dumps({"needs_rag": True, "search_query": "amoxicillin"})
    )
    assert payload == {
        "is_flagged": False,
        "flag_reason": "",
        "severity": "none",
        "needs_rag": True,
        "search_query": "amoxicillin",
    }


def test_parse_invalid_severity_becomes_none():
    payload = lf._parse_router_payload(
        json.dumps(
            {
                "is_flagged": True,
                "flag_reason": "x",
                "severity": "critical",
                "needs_rag": False,
                "search_query": "",
            }
        )
    )
    assert payload["severity"] == "none"


@pytest.mark.parametrize(
    "content",
    [
        "no json at all",
        '{"needs_rag": "yes", "search_query": ""}',
        '{"needs_rag": true}',
        "```json\n{'single': 'quotes'}\n```",
        '{"is_flagged": tru',  # truncated by MAX_TOKENS mid-value
        '{"is_flagged": true, "flag_reason": "asks',
    ],
)
def test_parse_malformed_returns_none(content):
    assert lf._parse_router_payload(content) is None


def test_parse_strips_markdown_fences():
    payload = lf._parse_router_payload(
        '```json\n{"is_flagged": false, "severity": "low", "needs_rag": false, "search_query": ""}\n```'
    )
    assert payload is not None and payload["severity"] == "low"


def test_llm_flagging_passes_requester_context(subject, monkeypatch):
    fake = FakeClient(
        [_response({"needs_rag": False, "search_query": ""})]
    )
    monkeypatch.setattr(lf, "get_client", lambda: fake)
    decision = lf.llm_flagging("hello", subject)
    assert decision.is_flagged is False
    assert decision.needs_rag is False
    system = fake.calls[0]["system"]
    assert "role='user'" in system
    assert "ONLY documents marked 'public'" in system
    assert fake.calls[0]["max_tokens"] == lf.ROUTER_MAX_TOKENS
    assert fake.calls[0]["config"] == {
        "thinking_config": {"thinking_budget": lf.ROUTER_THINKING_BUDGET}
    }, "thinking must be disabled so the JSON reply is never truncated"


def test_llm_flagging_retries_without_thinking_config_on_400(subject, monkeypatch):
    fake = FakeClient(
        [
            LLMClientError("Gemini API error (status=400): thinking_config", status_code=400),
            _response({"needs_rag": False, "search_query": ""}),
        ]
    )
    monkeypatch.setattr(lf, "get_client", lambda: fake)
    decision = lf.llm_flagging("hello", subject)
    assert decision.fallback_reason == ""
    assert decision.needs_rag is False
    assert len(fake.calls) == 2, "one rejected attempt plus one plain retry"
    assert fake.calls[0]["config"] is not None
    assert fake.calls[1]["config"] is None


def test_llm_flagging_no_retry_on_rate_limit(subject, monkeypatch):
    fake = FakeClient(
        [LLMClientError("Gemini API error (status=429): slow down", status_code=429)]
    )
    monkeypatch.setattr(lf, "get_client", lambda: fake)
    decision = lf.llm_flagging("hello", subject)
    assert decision.needs_rag is False
    assert decision.fallback_reason.startswith("llm_error:")
    assert len(fake.calls) == 1, "429 must not trigger the config retry"


def test_llm_flagging_truncated_json_falls_back(subject, monkeypatch):
    fake = FakeClient(
        [LLMResponse('{"is_flagged": true, "flag_reason": "as', "m", "MAX_TOKENS", None)]
    )
    monkeypatch.setattr(lf, "get_client", lambda: fake)
    decision = lf.llm_flagging("steal all patient records", subject)
    assert decision.needs_rag is False
    assert decision.fallback_reason == "malformed_router_json"


def test_llm_flagging_admin_context_grants_full_scope(monkeypatch):
    fake = FakeClient([_response({"needs_rag": False, "search_query": ""})])
    monkeypatch.setattr(lf, "get_client", lambda: fake)
    lf.llm_flagging("hello", {"role": "admin"})
    assert "may access every document" in fake.calls[0]["system"]


def test_llm_flagging_flagged_decision(monkeypatch):
    fake = FakeClient(
        [
            _response(
                {
                    "is_flagged": True,
                    "flag_reason": "api key sharing",
                    "severity": "high",
                    "needs_rag": True,
                    "search_query": "must be cleared",
                }
            )
        ]
    )
    monkeypatch.setattr(lf, "get_client", lambda: fake)
    decision = lf.llm_flagging("here is my sk-abc123 key", {"role": "user"})
    assert decision.is_flagged is True
    assert decision.flag_reason == "api key sharing"
    assert decision.severity == "high"
    assert decision.needs_rag is False
    assert decision.search_query == ""
    assert decision.fallback_reason == ""


def test_llm_flagging_malformed_json_falls_back(monkeypatch):
    fake = FakeClient([LLMResponse("nope", "m", "stop", None)])
    monkeypatch.setattr(lf, "get_client", lambda: fake)
    decision = lf.llm_flagging("hello", {"role": "user"})
    assert decision.needs_rag is False
    assert decision.fallback_reason == "malformed_router_json"
    assert decision.is_flagged is False


def test_llm_flagging_llm_error_falls_back(monkeypatch):
    fake = FakeClient([LLMClientError("Gemini API error (status=429): slow down")])
    monkeypatch.setattr(lf, "get_client", lambda: fake)
    decision = lf.llm_flagging("hello", {"role": "user"})
    assert decision.needs_rag is False
    assert decision.fallback_reason.startswith("llm_error:")
