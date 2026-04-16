"""BaselineModelInference — pre-trained RoBERTa sentiment classification."""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline as hf_pipeline

from contracts.errors import ModelError, UnsupportedLanguageError
from contracts.model_interface import ModelInference, PredictionResult, SHAPResult
from src.model.config import ModelConfig
from src.model.device import get_device


class BaselineModelInference(ModelInference):
    """Implement ModelInference interface using pre-trained RoBERTa.

    - Overall sentiment: predicted from model
    - ABSA aspects: returns [] (baseline has no ABSA)
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
        """Predict overall sentiment. The baseline model does not produce aspects."""
        self._check_language(lang)
        probs = self._predict_probabilities(text)[0]
        pred_idx = probs.argmax().item()

        return PredictionResult(
            sentiment=self._config.label_map[pred_idx],
            confidence=round(probs[pred_idx].item(), 4),
            aspects=[],
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
                    aspects=[],
                    sarcasm_flag=False,
                )
            )
        return results

    def _get_classification_pipeline(self):
        """Lazy-init a callable for SHAP explainability."""
        if self._hf_pipeline is None:
            pipeline_device = (
                self._device.index
                if self._device.type == "cuda" and self._device.index is not None
                else (-1 if self._device.type == "cpu" else self._device)
            )
            self._hf_pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self._model,
                tokenizer=self._tokenizer,
                device=pipeline_device,
                top_k=None,
            )
        return self._hf_pipeline

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
