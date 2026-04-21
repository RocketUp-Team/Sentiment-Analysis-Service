from __future__ import annotations

import random
import time

from contracts.errors import UnsupportedLanguageError
from contracts.model_interface import AspectSentiment, ModelInference, PredictionResult, SHAPResult

SENTIMENTS = ("positive", "negative", "neutral")
ASPECTS = ("food", "service", "ambiance", "price", "location", "general")
SUPPORTED_LANGUAGES = ["en", "vi"]


class MockModelInference(ModelInference):
    @property
    def supported_languages(self) -> list[str]:
        return list(SUPPORTED_LANGUAGES)

    @property
    def is_loaded(self) -> bool:
        return True

    def _check_language(self, lang: str) -> None:
        if lang not in SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(lang)

    def _random_prediction(self) -> PredictionResult:
        time.sleep(random.uniform(0.03, 0.08))
        sentiment = random.choice(SENTIMENTS)
        confidence = random.uniform(0.55, 0.99)
        aspect_count = random.randint(1, 3)
        chosen_aspects = random.sample(ASPECTS, k=aspect_count)
        aspects = [
            AspectSentiment(
                aspect=aspect,
                sentiment=random.choice(SENTIMENTS),
                confidence=random.uniform(0.55, 0.99),
            )
            for aspect in chosen_aspects
        ]
        return PredictionResult(
            sentiment=sentiment,
            confidence=confidence,
            aspects=aspects,
            sarcasm_flag=random.random() < 0.1,
        )

    def _random_shap(self, text: str) -> SHAPResult:
        tokens = text.split()
        if not tokens:
            tokens = ["<empty>"]
        shap_values = [random.uniform(-1.0, 1.0) for _ in tokens]
        base_value = float(random.uniform(-0.5, 0.5))
        return SHAPResult(tokens=tokens, shap_values=shap_values, base_value=base_value)

    def predict_single(self, text: str, lang: str = "en") -> PredictionResult:
        self._check_language(lang)
        return self._random_prediction()

    def predict_batch(
        self,
        texts: list[str],
        lang: str = "en",
        *,
        batch_size: int | None = None,
        skip_absa: bool = False,
        skip_sarcasm: bool = False,
    ) -> list[PredictionResult]:
        self._check_language(lang)
        return [self._random_prediction() for _ in texts]

    def get_shap_explanation(self, text: str, lang: str = "en") -> SHAPResult:
        self._check_language(lang)
        return self._random_shap(text)
