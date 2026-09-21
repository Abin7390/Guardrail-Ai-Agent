import pytest

import guard.steps.masking as masking_module
from guard.steps.masking import MaskedEntity, demask, mask, supplemental_mask


@pytest.fixture
def fallback_engine(monkeypatch):
    """Force the regex fallback engine so reversible tests are deterministic."""
    monkeypatch.setattr(masking_module, "_analyzer", None)
    monkeypatch.setattr(masking_module, "_anonymizer", None)
    monkeypatch.setattr(masking_module, "_engine_mode", "fallback")


def test_spelled_out_phone_masked():
    text = "my phone number is nine five nine five nine five nine five nine five okay"
    masked, entities = supplemental_mask(text)
    assert entities == {"PHONE_NUMBER": 1}
    assert "nine five" not in masked
    assert "[REDACTED]" in masked


def test_disclosed_name_masked():
    text = "What is the protocol for study 101? my name is alen"
    masked, entities = supplemental_mask(text)
    assert entities == {"PERSON": 1}
    assert "alen" not in masked
    assert "my name is [REDACTED]" in masked


def test_call_me_stopword_not_masked():
    masked, entities = supplemental_mask("call me later please")
    assert entities == {}
    assert masked == "call me later please"


def test_short_number_word_runs_not_masked():
    masked, entities = supplemental_mask("i have two three four options")
    assert entities == {}
    assert masked == "i have two three four options"


def test_supplemental_mask_with_state_records_mapping():
    state = masking_module._MaskState()
    masked, entities = supplemental_mask("my name is alen", state)
    assert masked == "my name is [REDACTED_1]"
    assert entities == {"PERSON": 1}
    assert state.mapping == [MaskedEntity("PERSON", "alen", "[REDACTED_1]")]


def test_default_mode_uses_literal_redacted(fallback_engine):
    result = mask("email john.doe@example.com now")
    assert result.masked_text == "email [REDACTED] now"
    assert result.entities == {"EMAIL_ADDRESS": 1}
    assert result.mapping is None


def test_reversible_fallback_round_trip(fallback_engine):
    text = "email john.doe@example.com about ssn 123-45-6789"
    result = mask(text, reversible=True)
    assert "[REDACTED_1]" in result.masked_text
    assert "[REDACTED_2]" in result.masked_text
    assert "john.doe@example.com" not in result.masked_text
    assert "123-45-6789" not in result.masked_text
    assert result.mapping is not None
    assert {entity.entity_type for entity in result.mapping} == {
        "EMAIL_ADDRESS",
        "US_SSN",
    }
    assert demask(result.masked_text, result.mapping) == text


def test_reversible_supplemental_placeholders(fallback_engine):
    text = "my name is alen and my number is nine five nine five nine five nine five"
    result = mask(text, reversible=True)
    assert result.entities == {"PHONE_NUMBER": 1, "PERSON": 1}
    assert "alen" not in result.masked_text
    assert "nine" not in result.masked_text
    assert "[REDACTED_1]" in result.masked_text and "[REDACTED_2]" in result.masked_text
    assert result.mapping is not None
    assert demask(result.masked_text, result.mapping) == text


def test_reversible_numbering_beyond_ten_round_trips(fallback_engine):
    text = " ".join(f"user{i}@example.com" for i in range(1, 13))
    result = mask(text, reversible=True)
    assert result.masked_text.count("[REDACTED_") == 12
    assert "[REDACTED_12]" in result.masked_text
    assert result.entities == {"EMAIL_ADDRESS": 12}
    assert demask(result.masked_text, result.mapping) == text


def test_demask_leaves_unknown_tokens_visible():
    mapping = [MaskedEntity("PERSON", "alen", "[REDACTED_1]")]
    assert demask("hi [REDACTED_1] and [REDACTED_9]", mapping) == "hi alen and [REDACTED_9]"
    assert demask("no tokens here", mapping) == "no tokens here"
    assert demask("anything", None) == "anything"
    assert demask("anything", []) == "anything"
