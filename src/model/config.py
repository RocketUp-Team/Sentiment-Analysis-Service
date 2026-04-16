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
        absa_model_name: HuggingFace zero-shot ABSA model identifier.
        absa_threshold: Minimum confidence threshold for ABSA predictions.
        absa_categories: Tuple of supported ABSA aspect categories.
        absa_aspect_template: Template for aspect extraction.
        absa_sentiment_template: Template for per-aspect sentiment.
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
    absa_model_name: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    absa_threshold: float = 0.45
    absa_categories: tuple[str, ...] = (
        "food",
        "service",
        "ambiance",
        "price",
        "location",
        "general",
    )
    absa_aspect_template: str = "The text contains a discussion about {}."
    absa_sentiment_template: str = "The sentiment expressed towards {aspect} is {{}}."
