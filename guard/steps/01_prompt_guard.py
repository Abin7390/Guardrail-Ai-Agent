"""Stage 1: jailbreak/prompt-injection detection via Prompt-Guard-2-86M.

Falls back to regex heuristics (adapted from the NeMo POC) when the HF
model cannot be loaded, so flagging keeps working offline.
"""

import logging
import os
import re
import threading
from dataclasses import dataclass

logger = logging.getLogger("guard.prompt_guard")

DEFAULT_MODEL_ID = "project-free-llama/Llama-Prompt-Guard-2-86M"
BENIGN_LABEL = "BENIGN"
SUSPICIOUS_LABEL = "SUSPICIOUS"
DEFAULT_THRESHOLD = 0.5
MAX_TOKENS = 512

_lock = threading.Lock()
_tokenizer = None
_model = None
_benign_idx = 0
_engine_mode: str | None = None

_SUSPICIOUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+|the\s+|your\s+|these\s+|those\s+){0,3}"
        r"(?:previous|prior|above|earlier|preceding|past)\s+(?:\w+\s+){0,3}?"
        r"(?:instructions?|prompts?|rules?|directions?|guidance)\b",
        r"\b(?:reveal|show|print|repeat|output|expose|leak)\b[^.!?]{0,60}"
        r"\b(?:system\s+prompt|initial\s+instructions?|hidden\s+instructions?|"
        r"secret\s+(?:prompt|instructions?)|developer\s+message)\b",
        r"\bjailbreak\b",
        r"\bDAN\s+mode\b",
        r"\bdo\s+anything\s+now\b",
        r"\byou\s+are\s+now\s+(?:an?\s+)?(?:unrestricted|unfiltered|uncensored|free\s+of\s+restrictions)\b",
        r"\bact\s+as\s+(?:an?\s+)?(?:unrestricted|unfiltered|uncensored|unaligned)\b(?:\s+AI|\s+model|\s+assistant)?",
        r"\b(?:disable|turn\s+off|switch\s+off|shut\s+off|bypass|deactivate|circumvent|override|remove|negate)\b"
        r"[^.!?]{0,60}\b(?:presidio|guardrails?|guard\s+rails?|security\s+(?:screening|rail|check)s?|"
        r"screening\s+(?:pipeline|layer)|safety\s+(?:filters?|checks?|mechanisms?)|"
        r"content\s+(?:filters?|moderation)|pii\s+(?:detection|filter\w*|redaction)|anonymization)\b",
        r"\b(?:presidio|guardrails?|security\s+screening)\b[^.!?]{0,30}"
        r"\b(?:is|are|was|were)\s+(?:now\s+)?(?:off|disabled|deactivated|inactive|bypassed)\b",
    ]
]

_SUPPLEMENTAL_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\b(?:don'?t|do\s+not)\s+care\s+about\s+(?:the\s+|any\s+|all\s+|your\s+|this\s+|that\s+|those\s+)?"
        r"[a-z\s]{0,40}?(?:instruction|instructions|prompt|prompts|rule|rules|system|context|message)\b",
        r"\bpay\s+no\s+attention\s+to\s+(?:the\s+|any\s+|all\s+)?[a-z\s]{0,30}?"
        r"(?:instruction|instructions|prompt|rule|rules|system)\b",
        r"\b(?:pretend|act)\s+as\s+if\s+(?:the\s+|any\s+|all\s+|your\s+)?[a-z\s]{0,30}?"
        r"(?:instruction|instructions|rule|rules|restriction|restrictions)\s+(?:do(?:n'?t|\\s+not)?\s+)?exist\b",
    ]
]


@dataclass(frozen=True)
class PromptGuardVerdict:
    flagged: bool
    label: str
    benign_score: float
    suspicious_score: float
    engine: str
    matched: str | None = None


def _ensure_engine() -> None:
    global _tokenizer, _model, _benign_idx, _engine_mode
    if _engine_mode is not None:
        return
    with _lock:
        if _engine_mode is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_id = os.environ.get("PROMPTGUARD_MODEL_ID", DEFAULT_MODEL_ID)
            logger.info("prompt-guard | loading model: %s (first run downloads it)", model_id)
            _tokenizer = AutoTokenizer.from_pretrained(model_id)
            _model = AutoModelForSequenceClassification.from_pretrained(model_id)
            _model.eval()
            id2label = {int(k): str(v).upper() for k, v in _model.config.id2label.items()}
            _benign_idx = next((i for i, name in id2label.items() if "BENIGN" in name), 0)
            _engine_mode = "hf"
            logger.info("prompt-guard | engine ready: hf (%d labels)", len(id2label))
        except Exception as exc:
            logger.warning(
                "prompt-guard | model unavailable (%s): regex fallback active", exc
            )
            _engine_mode = "regex-fallback"


def _score(token_ids: list[int]) -> tuple[float, float]:
    import torch

    with torch.no_grad():
        logits = _model(input_ids=torch.tensor([token_ids])).logits
        probs = torch.softmax(logits, dim=-1)[0]
    benign = float(probs[_benign_idx])
    return benign, 1.0 - benign


def _classify_hf(text: str) -> tuple[float, float]:
    token_ids = _tokenizer(text)["input_ids"]
    if len(token_ids) <= MAX_TOKENS:
        return _score(token_ids)
    stride = MAX_TOKENS - 32
    windows = [
        token_ids[start : start + MAX_TOKENS]
        for start in range(0, len(token_ids), stride)
    ]
    benign_best, suspicious_best = 1.0, 0.0
    for window in windows:
        benign, suspicious = _score(window)
        benign_best = min(benign_best, benign)
        suspicious_best = max(suspicious_best, suspicious)
    return benign_best, suspicious_best


def _classify_regex(text: str) -> PromptGuardVerdict:
    flagged = any(pattern.search(text or "") for pattern in _SUSPICIOUS_PATTERNS)
    label = SUSPICIOUS_LABEL if flagged else BENIGN_LABEL
    suspicious = 1.0 if flagged else 0.0
    return PromptGuardVerdict(flagged, label, round(1.0 - suspicious, 4), suspicious, "regex-fallback")


def get_engine_mode() -> str:
    """Current classifier engine: 'hf', 'regex-fallback', or 'unloaded'."""
    return _engine_mode or "unloaded"


def get_threshold() -> float:
    return float(os.environ.get("PROMPTGUARD_THRESHOLD", DEFAULT_THRESHOLD))


def _match_supplemental(text: str) -> str | None:
    """Return the first supplemental injection match, or None if the text is clean."""
    for pattern in _SUPPLEMENTAL_INJECTION_PATTERNS:
        match = pattern.search(text or "")
        if match:
            snippet = match.group(0)
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            return snippet
    return None


def classify(text: str) -> PromptGuardVerdict:
    """Classify raw text; flagged=True means jailbreak/injection detected."""
    threshold = get_threshold()
    _ensure_engine()
    matched = _match_supplemental(text)
    if _engine_mode != "hf":
        if matched is None:
            return _classify_regex(text)
        return PromptGuardVerdict(
            True, SUSPICIOUS_LABEL, 0.0, 1.0, "regex-fallback", matched
        )
    benign, suspicious = _classify_hf(text)
    model_flagged = suspicious >= threshold
    if matched is not None and not model_flagged:
        benign, suspicious = 0.0, 1.0
    flagged = model_flagged or matched is not None
    label = SUSPICIOUS_LABEL if flagged else BENIGN_LABEL
    return PromptGuardVerdict(
        flagged, label, round(benign, 4), round(suspicious, 4), "hf", matched
    )
