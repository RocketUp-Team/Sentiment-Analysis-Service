from __future__ import annotations

import abc
from dataclasses import dataclass, field

from contracts.errors import ModelError  # Re-exported for convenience.


@dataclass
class AspectSentiment:
    aspect: str
    sentiment: str
    confidence: float


@dataclass
class PredictionResult:
    sentiment: str
    confidence: float
    aspects: list[AspectSentiment] = field(default_factory=list)
    sarcasm_flag: bool = False


@dataclass
class SHAPResult:
    tokens: list[str]
    shap_values: list[float]
    base_value: float


class ModelInference(abc.ABC):
    @abc.abstractmethod
    def predict_single(self, text: str, lang: str = "en") -> PredictionResult:
        raise NotImplementedError

    @abc.abstractmethod
    def predict_batch(
        self,
        texts: list[str],
        lang: str = "en",
        *,
        batch_size: int | None = None,
        skip_absa: bool = False,
    ) -> list[PredictionResult]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_shap_explanation(self, text: str, lang: str = "en") -> SHAPResult:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def supported_languages(self) -> list[str]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool:
        raise NotImplementedError
