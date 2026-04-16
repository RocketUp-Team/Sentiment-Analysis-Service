"""Model configuration dataclass."""

from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for BaselineModelInference.

    Attributes:
        model_name: HuggingFace model identifier.
        max_length: Maximum token length for tokenizer.
        default_lang: Default language code.
        supported_languages: Tuple of supported language codes.
        label_map: Mapping from model output index to sentiment label.
    """

    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    max_length: int = 512
    default_lang: str = "en"
    supported_languages: tuple[str, ...] = ("en",)
    label_map: Mapping[int, str] = field(
        default_factory=lambda: MappingProxyType(
            {
                0: "negative",
                1: "neutral",
                2: "positive",
            }
        )
    )
