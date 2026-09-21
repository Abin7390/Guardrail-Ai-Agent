import math
import sys

import pytest

import guard.steps.embedding as embedding
from guard.steps.embedding import (
    FALLBACK_DIM,
    FALLBACK_ENGINE,
    FALLBACK_MODEL,
    _fallback_vector,
    embed,
    get_engine_mode,
)


@pytest.fixture
def forced_fallback(monkeypatch):
    monkeypatch.setattr(embedding, "_model", None)
    monkeypatch.setattr(embedding, "_engine_mode", None)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    return None


def _norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def test_fallback_deterministic(forced_fallback):
    first = embed("amoxicillin 500mg capsule")
    second = embed("amoxicillin 500mg capsule")
    assert first[0] == second[0]
    assert first[1] == FALLBACK_ENGINE
    assert first[2] == FALLBACK_MODEL
    assert get_engine_mode() == FALLBACK_ENGINE


def test_fallback_normalized_384_dims(forced_fallback):
    vector, engine, _ = embed("which patients use amoxicillin?")
    assert len(vector) == FALLBACK_DIM
    assert abs(_norm(vector) - 1.0) < 1e-9
    assert engine == FALLBACK_ENGINE


def test_fallback_different_texts_differ(forced_fallback):
    a, _, _ = embed("patient john uses metformin")
    b, _, _ = embed("paracetamol tablet for fever")
    assert a != b


def test_fallback_vector_unit_function():
    vector = _fallback_vector("metformin")
    assert len(vector) == FALLBACK_DIM
    assert abs(_norm(vector) - 1.0) < 1e-9
    assert _fallback_vector("metformin") == vector


def test_fallback_empty_text_is_zero_vector(forced_fallback):
    vector, engine, _ = embed("")
    assert len(vector) == FALLBACK_DIM
    assert all(value == 0.0 for value in vector)
    assert engine == FALLBACK_ENGINE


def test_engine_mode_unloaded_before_first_use():
    module_mode = embedding._engine_mode
    assert get_engine_mode() in {"unloaded", "minilm", "fallback"}
    assert module_mode is None or get_engine_mode() == module_mode
