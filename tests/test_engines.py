"""Tests for engine adapter behavior."""

import pytest

from src.engines.translation_engine import MarianTranslationEngine


def test_translation_engine_rejects_unsupported_pair():
    engine = MarianTranslationEngine()
    with pytest.raises(ValueError):
        engine.translate(["hello"], "tr", "zh")


def test_translation_engine_uses_mapped_model(monkeypatch):
    engine = MarianTranslationEngine()

    class _Tokenizer:
        def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
            return {"input_ids": texts}

        def batch_decode(self, generated, skip_special_tokens=True):
            return generated

    class _Model:
        def generate(self, **encoded):
            return [f"translated:{text}" for text in encoded["input_ids"]]

    monkeypatch.setattr(engine, "_load", lambda _name: (_Tokenizer(), _Model()))

    translated = engine.translate(["hello"], "en", "es")
    assert translated == ["translated:hello"]
