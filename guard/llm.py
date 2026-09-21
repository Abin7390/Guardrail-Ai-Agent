"""Shared Gemini client for any module that needs an LLM call.

Thin wrapper around the ``google-genai`` SDK (``from google import genai``).
Credentials and settings come from the ``GUARD_LLM_*`` environment variables
plus the SDK-native ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` variables (see the
``.env`` file at the project root, loaded here via python-dotenv with
``override=False`` so real environment variables always win). The underlying
SDK client is constructed lazily and thread-safely on the first ``chat()``
call; a missing API key only fails at call time, so importing this module (or
``guard``) stays safe offline. Message content is never logged, matching the
project's guardrail conventions.
"""

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("guard.llm")

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_TIMEOUT = 60.0

API_KEY_VAR = "GUARD_LLM_API_KEY"
MODEL_VAR = "GUARD_LLM_MODEL"
TIMEOUT_VAR = "GUARD_LLM_TIMEOUT"
SDK_KEY_VARS = ("GOOGLE_API_KEY", "GEMINI_API_KEY")


def _load_env_file() -> None:
    """Load ``custom/.env`` without overriding variables already in the shell."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ENV_FILE, override=False)


_load_env_file()


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    finish_reason: str | None
    usage: dict | None


class LLMClientError(RuntimeError):
    """Raised when a Gemini call cannot be made or fails (missing key, API error)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _extract_usage(response) -> dict | None:
    """Pull token usage off a genai response, tolerating shape differences."""
    meta = getattr(response, "usage_metadata", None)
    if meta is not None:
        return {
            "prompt_tokens": getattr(meta, "prompt_token_count", None),
            "completion_tokens": getattr(meta, "candidates_token_count", None),
            "total_tokens": getattr(meta, "total_token_count", None),
        }
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


class GeminiClient:
    """Reusable chat client for the Gemini API (google-genai SDK)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get(API_KEY_VAR, "")
        self._model = model or os.environ.get(MODEL_VAR, DEFAULT_MODEL)
        self._timeout = (
            timeout
            if timeout is not None
            else float(os.environ.get(TIMEOUT_VAR, str(DEFAULT_TIMEOUT)))
        )
        self._lock = threading.Lock()
        self._client = None

    @property
    def model(self) -> str:
        return self._model

    def _resolve_api_key(self) -> str | None:
        if self._api_key:
            return self._api_key
        for var in SDK_KEY_VARS:
            value = os.environ.get(var, "")
            if value:
                return value
        return None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        with self._lock:
            if self._client is not None:
                return
            api_key = self._resolve_api_key()
            if not api_key:
                raise LLMClientError(
                    f"no API key found; set {API_KEY_VAR} (or GOOGLE_API_KEY) in "
                    f"{ENV_FILE.name} at the project root or export it as an "
                    "environment variable"
                )
            try:
                from google import genai
            except ImportError as exc:
                raise LLMClientError(
                    "the 'google-genai' package is not installed; "
                    "run: pip install -r requirements.txt"
                ) from exc
            logger.info(
                "llm | creating Gemini client (model=%s timeout=%ss)",
                self._model,
                self._timeout,
            )
            self._client = genai.Client(
                api_key=api_key,
                http_options={"timeout": int(self._timeout * 1000)},
            )

    @staticmethod
    def _contents_from_messages(messages: list[dict]):
        if len(messages) == 1:
            return messages[0].get("content", "")
        contents = []
        for message in messages:
            role = message.get("role", "user")
            contents.append(
                {
                    "role": "model" if role == "assistant" else role,
                    "parts": [{"text": message.get("content", "")}],
                }
            )
        return contents

    def chat(
        self,
        messages: list[dict],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """One blocking generation; returns content plus usage metadata."""
        if not messages:
            raise LLMClientError("messages must contain at least one message")
        self._ensure_client()

        config = dict(kwargs.pop("config", None) or {})
        if system is not None:
            config["system_instruction"] = system
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=self._contents_from_messages(messages),
                **({"config": config} if config else {}),
            )
        except Exception as exc:
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            logger.warning(
                "llm | API error: status=%s model=%s (%s)",
                status,
                self._model,
                type(exc).__name__,
            )
            raise LLMClientError(
                f"Gemini API error (status={status}): {exc}", status_code=status
            ) from exc

        candidates = getattr(response, "candidates", None) or []
        content = None
        for attr in ("text", "output_text"):
            try:
                value = getattr(response, attr, None)
            except Exception:
                value = None
            if isinstance(value, str) and value:
                content = value
                break
        if content is None:
            if not candidates:
                raise LLMClientError("Gemini response contained no output")
            parts = getattr(getattr(candidates[0], "content", None), "parts", None) or []
            content = "".join(getattr(part, "text", None) or "" for part in parts)
        finish_reason = None
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
        if finish_reason is not None and not isinstance(finish_reason, str):
            finish_reason = getattr(finish_reason, "value", None) or str(finish_reason)
        usage = _extract_usage(response)
        response_model = (
            getattr(response, "model", None)
            or getattr(response, "model_version", None)
            or self._model
        )
        if not isinstance(response_model, str):
            response_model = self._model
        logger.info(
            "llm | chat complete: model=%s finish=%s total_tokens=%s",
            response_model,
            finish_reason,
            usage.get("total_tokens") if usage else None,
        )
        return LLMResponse(
            content=content or "",
            model=response_model,
            finish_reason=finish_reason,
            usage=usage,
        )


_default_lock = threading.Lock()
_default_client: GeminiClient | None = None


def get_client() -> GeminiClient:
    """Shared env-configured GeminiClient (lazy singleton)."""
    global _default_client
    if _default_client is None:
        with _default_lock:
            if _default_client is None:
                _default_client = GeminiClient()
    return _default_client
