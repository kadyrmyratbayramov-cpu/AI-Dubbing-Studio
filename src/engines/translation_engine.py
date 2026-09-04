"""MarianMT translation engine."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


MARIAN_MODELS: Dict[Tuple[str, str], str] = {
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("en", "it"): "Helsinki-NLP/opus-mt-en-it",
    ("it", "en"): "Helsinki-NLP/opus-mt-it-en",
    ("en", "pt"): "Helsinki-NLP/opus-mt-en-ROMANCE",
    ("en", "ru"): "Helsinki-NLP/opus-mt-en-ru",
    ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
    ("ja", "en"): "Helsinki-NLP/opus-mt-jap-en",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
}


class MarianTranslationEngine:
    def __init__(self):
        self._cache: Dict[str, object] = {}

    def _load(self, model_name: str):
        if model_name in self._cache:
            return self._cache[model_name]
        from transformers import MarianMTModel, MarianTokenizer

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        self._cache[model_name] = (tokenizer, model)
        return tokenizer, model

    def translate(self, texts: Iterable[str], source_lang: str, target_lang: str) -> List[str]:
        if source_lang == target_lang:
            return list(texts)

        model_name = MARIAN_MODELS.get((source_lang, target_lang))
        if not model_name:
            raise ValueError(f"Unsupported language pair: {source_lang}->{target_lang}")

        tokenizer, model = self._load(model_name)
        encoded = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)
        generated = model.generate(**encoded)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)
