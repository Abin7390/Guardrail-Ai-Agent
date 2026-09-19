import sys
import types
from types import SimpleNamespace

import pytest

from guard.llm import GeminiClient, LLMClientError, get_client


class _FakeGenAI:
    instances = []
    create_kwargs = None
    response = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.models = SimpleNamespace(generate_content=self._generate_content)
        _FakeGenAI.instances.append(self)

    @staticmethod
    def _generate_content(**kwargs):
        _FakeGenAI.create_kwargs = kwargs
        response = _FakeGenAI.response
        if isinstance(response, Exception):
            raise response
        return response


class _FakeAPIError(Exception):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


def _fake_response(content="hello there", model="gemini-3.8-flash"):
    return SimpleNamespace(
        text=content,
        model_version=model,
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(parts=[SimpleNamespace(text=content)]),
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=3, candidates_token_count=5, total_token_count=8
        ),
    )


@pytest.fixture
def fake_genai(monkeypatch):
    _FakeGenAI.instances = []
    _FakeGenAI.create_kwargs = None
    _FakeGenAI.response = _fake_response()
    genai_module = types.SimpleNamespace(Client=_FakeGenAI)
    google_package = types.ModuleType("google")
    google_package.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_package)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setenv("GUARD_LLM_API_KEY", "test-key")
    return _FakeGenAI


def test_chat_requires_api_key(monkeypatch):
    for var in ("GUARD_LLM_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    client = GeminiClient()
    with pytest.raises(LLMClientError, match="GUARD_LLM_API_KEY"):
        client.chat([{"role": "user", "content": "hi"}])


def test_sdk_env_key_accepted(fake_genai, monkeypatch):
    monkeypatch.delenv("GUARD_LLM_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    GeminiClient().chat([{"role": "user", "content": "hi"}])
    ctor = fake_genai.instances[-1].kwargs
    assert ctor["api_key"] == "google-key"


def test_chat_rejects_empty_messages():
    with pytest.raises(LLMClientError, match="at least one"):
        GeminiClient(api_key="k").chat([])


def test_chat_returns_response(fake_genai):
    result = GeminiClient(api_key="k").chat([{"role": "user", "content": "hi"}])
    assert result.content == "hello there"
    assert result.model == "gemini-3.8-flash"
    assert result.finish_reason == "STOP"
    assert result.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }


def test_chat_maps_system_and_options_into_config(fake_genai, monkeypatch):
    monkeypatch.delenv("GUARD_LLM_MODEL", raising=False)
    GeminiClient(api_key="k").chat(
        [{"role": "user", "content": "hi"}],
        system="be brief",
        temperature=0.2,
        max_tokens=64,
    )
    kwargs = fake_genai.create_kwargs
    assert kwargs["model"] == "gemini-3.8-flash"
    assert kwargs["contents"] == "hi"
    assert kwargs["config"] == {
        "system_instruction": "be brief",
        "temperature": 0.2,
        "max_output_tokens": 64,
    }


def test_multi_message_contents_mapped_to_gemini_roles(fake_genai):
    GeminiClient(api_key="k").chat(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
        ]
    )
    assert fake_genai.create_kwargs["contents"] == [
        {"role": "user", "parts": [{"text": "hello"}]},
        {"role": "model", "parts": [{"text": "hi"}]},
        {"role": "user", "parts": [{"text": "bye"}]},
    ]


def test_env_overrides_honored(fake_genai, monkeypatch):
    monkeypatch.setenv("GUARD_LLM_MODEL", "gemini-3.8-pro")
    monkeypatch.setenv("GUARD_LLM_TIMEOUT", "12.5")
    client = GeminiClient()
    assert client.model == "gemini-3.8-pro"
    client.chat([{"role": "user", "content": "hi"}])
    ctor = fake_genai.instances[-1].kwargs
    assert ctor["http_options"] == {"timeout": 12500}
    assert fake_genai.create_kwargs["model"] == "gemini-3.8-pro"


def test_api_error_surfaces_as_llm_error(fake_genai):
    fake_genai.response = _FakeAPIError("rate limited", 429)
    with pytest.raises(LLMClientError) as excinfo:
        GeminiClient(api_key="k").chat([{"role": "user", "content": "hi"}])
    assert excinfo.value.status_code == 429
    assert "Gemini API error" in str(excinfo.value)


def test_empty_output_raises(fake_genai):
    fake_genai.response = SimpleNamespace(
        text=None, candidates=[], model_version="gemini-3.8-flash", usage_metadata=None
    )
    with pytest.raises(LLMClientError, match="no output"):
        GeminiClient(api_key="k").chat([{"role": "user", "content": "hi"}])


def test_get_client_returns_same_instance():
    assert get_client() is get_client()
