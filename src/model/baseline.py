"""BaselineModelInference — RoBERTa sentiment + zero-shot ABSA support."""

from __future__ import annotations

import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline as hf_pipeline

from contracts.errors import ModelError, UnsupportedLanguageError
from contracts.model_interface import (
    AspectSentiment,
    ModelInference,
    PredictionResult,
    SHAPResult,
)
from src.model.config import ModelConfig
from src.model.device import get_device

logger = logging.getLogger(__name__)


class BaselineModelInference(ModelInference):
    """Implement ModelInference with RoBERTa sentiment and zero-shot ABSA.

    - Overall sentiment: predicted from model
    - ABSA aspects: extracted with a zero-shot classifier
    - SHAP: uses shap.Explainer with HuggingFace pipeline
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        device: torch.device | None = None,
    ):
        self._config = config or ModelConfig()
        self._device = device or get_device()
        self._model = None
        self._tokenizer = None
        self._hf_pipeline = None
        self._absa_pipeline = None
        self._load_model()

    def _load_model(self) -> None:
        """Load tokenizer and model, then move the model to the target device."""
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._config.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._config.model_name
            ).to(self._device)
            self._model.eval()
        except Exception as exc:
            raise ModelError(f"Failed to load model: {exc}") from exc

    def _check_language(self, lang: str) -> None:
        if lang not in self._config.supported_languages:
            raise UnsupportedLanguageError(lang)

    def _move_inputs_to_device(self, inputs):
        """Handle both HuggingFace BatchEncoding and mocked dict payloads."""
        if hasattr(inputs, "to"):
            return inputs.to(self._device)
        if isinstance(inputs, dict):
            return {
                key: value.to(self._device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        return inputs

    def _predict_probabilities(
        self, texts: str | list[str], *, padding: bool = False
    ) -> torch.Tensor:
        """Run the tokenizer/model stack and return class probabilities."""
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "truncation": True,
            "max_length": self._config.max_length,
        }
        if padding:
            tokenizer_kwargs["padding"] = True

        inputs = self._move_inputs_to_device(
            self._tokenizer(
                texts,
                **tokenizer_kwargs,
            )
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        return torch.softmax(outputs.logits, dim=-1)

    def predict_single(self, text: str, lang: str = "en") -> PredictionResult:
        """Predict overall sentiment and extract aspect sentiment when available."""
        self._check_language(lang)
        probs = self._predict_probabilities(text)[0]
        pred_idx = probs.argmax().item()

        return PredictionResult(
            sentiment=self._config.label_map[pred_idx],
            confidence=round(probs[pred_idx].item(), 4),
            aspects=self._extract_aspects(text),
            sarcasm_flag=False,
        )

    def predict_batch(
        self, texts: list[str], lang: str = "en"
    ) -> list[PredictionResult]:
        """Predict sentiment for a batch of texts."""
        if not texts:
            return []

        self._check_language(lang)
        probs = self._predict_probabilities(texts, padding=True)
        results: list[PredictionResult] = []
        for index in range(len(texts)):
            pred_idx = probs[index].argmax().item()
            results.append(
                PredictionResult(
                    sentiment=self._config.label_map[pred_idx],
                    confidence=round(probs[index][pred_idx].item(), 4),
                    aspects=self._extract_aspects(texts[index]),
                    sarcasm_flag=False,
                )
            )
        return results

    @property
    def _pipeline_device(self):
        """Map the torch device to the format expected by HuggingFace pipelines."""
        if self._device.type == "cuda" and self._device.index is not None:
            return self._device.index
        if self._device.type == "cpu":
            return -1
        return self._device

    def _get_classification_pipeline(self):
        """Lazy-init a callable for SHAP explainability."""
        if self._hf_pipeline is None:
            self._hf_pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self._model,
                tokenizer=self._tokenizer,
                device=self._pipeline_device,
                top_k=None,
            )
        return self._hf_pipeline

    def _get_absa_pipeline(self):
        """Lazy-init the zero-shot classifier used for ABSA extraction."""
        if self._absa_pipeline is None:
            try:
                self._absa_pipeline = hf_pipeline(
                    "zero-shot-classification",
                    model=self._config.absa_model_name,
                    device=self._pipeline_device,
                )
            except Exception as exc:
                raise ModelError(f"Failed to load ABSA model: {exc}") from exc
        return self._absa_pipeline

    def _extract_aspects(self, text: str) -> list[AspectSentiment]:
        """Extract aspect-level sentiment, falling back to no aspects on failure."""
        try:
            pipeline = self._get_absa_pipeline()
            aspect_result = pipeline(
                text,
                candidate_labels=list(self._config.absa_categories),
                hypothesis_template="This review is about {}.",
                multi_label=True,
            )
            detected_aspects = [
                label
                for label, score in zip(
                    aspect_result["labels"],
                    aspect_result["scores"],
                )
                if score > self._config.absa_threshold
            ]

            aspects: list[AspectSentiment] = []
            for aspect_name in detected_aspects:
                sent_result = pipeline(
                    text,
                    candidate_labels=["positive", "negative", "neutral"],
                    hypothesis_template=f"The sentiment about {aspect_name} is {{}}.",
                    multi_label=False,
                )
                aspects.append(
                    AspectSentiment(
                        aspect=aspect_name,
                        sentiment=sent_result["labels"][0],
                        confidence=round(sent_result["scores"][0], 4),
                    )
                )
            return aspects
        except Exception:
            logger.warning("Failed to extract ABSA aspects for input text", exc_info=True)
            return []

    def get_shap_explanation(self, text: str, lang: str = "en") -> SHAPResult:
        """Return SHAP values per token for explainability."""
        import shap

        self._check_language(lang)
        predicted_probs = self._predict_probabilities(text)[0]
        predicted_class_idx = int(predicted_probs.argmax().item())
        pipe = self._get_classification_pipeline()
        explainer = shap.Explainer(pipe)
        shap_values = explainer([text])

        raw_tokens = shap_values.data[0]
        tokens = raw_tokens.tolist() if hasattr(raw_tokens, "tolist") else list(raw_tokens)
        values = shap_values.values[0][:, predicted_class_idx].tolist()
        base = float(shap_values.base_values[0][predicted_class_idx])

        return SHAPResult(tokens=tokens, shap_values=values, base_value=base)

    @property
    def supported_languages(self) -> list[str]:
        return list(self._config.supported_languages)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> torch.device:
        return self._device

    def __repr__(self) -> str:
        return (
            f"BaselineModelInference("
            f"model={self._config.model_name}, device={self._device})"
        )
