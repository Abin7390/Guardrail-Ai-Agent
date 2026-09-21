"""Stage 2: PII detection and masking via Presidio, with regex fallback."""

import logging
import os
import re
import threading
from dataclasses import dataclass

logger = logging.getLogger("guard.masking")

REDACTED = "[REDACTED]"

# Microsoft Presidio 
PII_ENTITIES = [
    "PERSON",
    "ORGANIZATION",
    "EMAIL_ADDRESS",
    "US_SSN",
    "CREDIT_CARD",
    "PHONE_NUMBER",
    "IBAN_CODE",
    "MEDICAL_LICENSE",
]
PII_SCORE_THRESHOLD = 0.4

_engine_lock = threading.Lock()
_analyzer = None
_anonymizer = None
_engine_mode: str | None = None

_FALLBACK_PATTERNS: dict[str, re.Pattern[str]] = {
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "EMAIL_ADDRESS": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "PHONE_NUMBER": re.compile(
        r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)"
    ),
    "IBAN_CODE": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,26}\b"),
}

_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "oh": "0",
    "niner": "9",
}
_NUMBER_WORD_ALT = "|".join(_NUMBER_WORDS)
_SPELLED_NUMBER_SEQ = re.compile(
    rf"\b(?:{_NUMBER_WORD_ALT})(?:[\s,.\-]+(?:{_NUMBER_WORD_ALT})){{6,}}\b",
    re.IGNORECASE,
)

_NAME_STOPWORDS = {
    "later", "back", "now", "please", "asap", "today", "tomorrow", "again",
    "soon", "if", "when", "at", "on", "in", "the", "a", "an", "your", "you",
    "him", "her", "them", "it", "there", "here",
}
_DISCLOSED_NAME = re.compile(
    r"\b(?:my name is|my name's|call me)\s+([a-z][a-z'.\-]{1,24})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MaskedEntity:
    entity_type: str
    value: str
    placeholder: str


@dataclass(frozen=True)
class MaskingResult:
    masked_text: str
    entities: dict[str, int]
    engine: str
    mapping: list[MaskedEntity] | None = None


class _MaskState:
    """Per-call state for reversible masking: placeholder counter + mapping.

    Created inside ``mask()`` (never module-level) so concurrent requests
    each number their own placeholders.
    """

    def __init__(self) -> None:
        self.mapping: list[MaskedEntity] = []
        self._counter = 0

    def record(self, entity_type: str, value: str) -> str:
        self._counter += 1
        placeholder = f"[REDACTED_{self._counter}]"
        self.mapping.append(MaskedEntity(entity_type, value, placeholder))
        return placeholder

    def operator_for(self, entity_type: str):
        def _replace(text: str) -> str:
            return self.record(entity_type, text)

        return _replace


_PLACEHOLDER_TOKEN = re.compile(r"\[REDACTED_(\d+)\]")


def demask(text: str, mapping: list[MaskedEntity] | None = None) -> str:
    """Restore ``[REDACTED_n]`` tokens using an exact-token lookup.

    Tokens without a mapping entry (e.g. hallucinated or mangled by the LLM)
    are left visible as-is; chained substring replace is never used, so
    ``[REDACTED_1]`` can never collide with ``[REDACTED_12]``.
    """
    if not mapping:
        return text
    lookup = {entity.placeholder: entity.value for entity in mapping}
    return _PLACEHOLDER_TOKEN.sub(
        lambda match: lookup.get(match.group(0), match.group(0)), text
    )


def _ensure_engines() -> None:
    global _analyzer, _anonymizer, _engine_mode
    if _engine_mode is not None:
        return
    with _engine_lock:
        if _engine_mode is not None:
            return
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            kwargs: dict = {}
            nlp = "default NLP engine"
            try:
                import spacy

                model_name = os.environ.get("GUARD_SPACY_MODEL", "en_core_web_lg ")
                if spacy.util.is_package(model_name):
                    from presidio_analyzer.nlp_engine import NlpEngineProvider

                    kwargs["nlp_engine"] = NlpEngineProvider(
                        nlp_configuration={
                            "nlp_engine_name": "spacy",
                            "models": [
                                {"lang_code": "en", "model_name": model_name}
                            ],
                            "ner_model_configuration": {
                                "labels_to_ignore": ["O"],
                                "model_to_presidio_entity_mapping": {
                                    "PERSON": "PERSON",
                                    "ORG": "ORGANIZATION",
                                    "GPE": "LOCATION",
                                    "LOC": "LOCATION",
                                    "FAC": "LOCATION",
                                    "NORP": "NRP",
                                },
                            },
                        }
                    ).create_engine()
                    nlp = f"spacy {model_name}"
            except Exception as exc:
                logger.warning(
                    "masking | spaCy NLP engine unavailable (%s); using default", exc
                )
            _analyzer = AnalyzerEngine(**kwargs)
            _anonymizer = AnonymizerEngine()
            _engine_mode = "presidio"
            logger.info("masking | engine ready: presidio (%s)", nlp)
        except Exception:
            _analyzer = None
            _anonymizer = None
            _engine_mode = "fallback"
            logger.warning("masking | Presidio unavailable: regex fallback active")


def _analyze(text: str):
    try:
        logger.info("masking | analyzing text with Presidio")
        return _analyzer.analyze(
            text=text,
            language="en",
            entities=PII_ENTITIES,
            score_threshold=PII_SCORE_THRESHOLD,
        )
    except Exception:
        return _analyzer.analyze(
            text=text, language="en", score_threshold=PII_SCORE_THRESHOLD
        )


def get_engine_mode() -> str:
    """Current masking engine: 'presidio', 'fallback', or 'unloaded'."""
    return _engine_mode or "unloaded"


def supplemental_mask(
    text: str, state: _MaskState | None = None
) -> tuple[str, dict[str, int]]:
    """Catch PII Presidio misses: spelled-out digit sequences and self-disclosed names.

    With ``state``, replacements are indexed placeholders recorded on it;
    without, the literal ``[REDACTED]`` is emitted (original behavior).
    """
    entities: dict[str, int] = {}

    def _phone(match: re.Match) -> str:
        entities["PHONE_NUMBER"] = entities.get("PHONE_NUMBER", 0) + 1
        if state is not None:
            return state.record("PHONE_NUMBER", match.group(0))
        return REDACTED

    masked = _SPELLED_NUMBER_SEQ.sub(_phone, text)

    def _name(match: re.Match) -> str:
        name = match.group(1)
        if name.lower() in _NAME_STOPWORDS:
            return match.group(0)
        entities["PERSON"] = entities.get("PERSON", 0) + 1
        replacement = (
            state.record("PERSON", name) if state is not None else REDACTED
        )
        return match.group(0).replace(name, replacement, 1)

    masked = _DISCLOSED_NAME.sub(_name, masked)
    return masked, entities


def mask(text: str, reversible: bool = False) -> MaskingResult:
    """Replace every detected PII entity in text with ``[REDACTED]``.

    ``reversible=True`` emits indexed placeholders ``[REDACTED_1]``,
    ``[REDACTED_2]``, ... instead and returns the value mapping so the
    original text can be restored with :func:`demask`.
    """
    _ensure_engines()
    entities: dict[str, int] = {}
    state = _MaskState() if reversible else None
    if _engine_mode == "presidio":
        analyzer_results = _analyze(text)
        for result in analyzer_results:
            entities[result.entity_type] = entities.get(result.entity_type, 0) + 1
        if analyzer_results:
            from presidio_anonymizer.entities import OperatorConfig

            if state is not None:
                operator_entities = set(PII_ENTITIES) | {
                    result.entity_type for result in analyzer_results
                }
                operators = {
                    entity: OperatorConfig(
                        "custom", {"lambda": state.operator_for(entity)}
                    )
                    for entity in operator_entities
                }
            else:
                operators = {
                    entity: OperatorConfig("replace", {"new_value": REDACTED})
                    for entity in PII_ENTITIES
                }
            masked = _anonymizer.anonymize(
                text=text, analyzer_results=analyzer_results, operators=operators
            ).text
        else:
            masked = text
    else:
        masked = text
        for entity, pattern in _FALLBACK_PATTERNS.items():
            if state is not None:
                matches = pattern.findall(masked)
                if matches:
                    entities[entity] = entities.get(entity, 0) + len(matches)
                    masked = pattern.sub(
                        lambda match, etype=entity: state.record(
                            etype, match.group(0)
                        ),
                        masked,
                    )
            else:
                count = len(pattern.findall(text))
                if count:
                    entities[entity] = count
                    masked = pattern.sub(REDACTED, masked)
    if state is not None:
        masked, extra = supplemental_mask(masked, state)
    else:
        masked, extra = supplemental_mask(masked)
    for entity, count in extra.items():
        entities[entity] = entities.get(entity, 0) + count
    return MaskingResult(
        masked, entities, _engine_mode, state.mapping if state is not None else None
    )
