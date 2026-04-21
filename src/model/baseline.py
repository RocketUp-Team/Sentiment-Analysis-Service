"""BaselineModelInference — RoBERTa sentiment + zero-shot ABSA support."""

from __future__ import annotations

import logging
import time
import torch
from peft import PeftModel
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
from src.model.onnx_inference import OnnxInferenceSession

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
        self._onnx_session = None
        self._sarcasm_onnx_session = None
        self._load_model()

    def preload(self) -> None:
        """Eagerly load heavy sub-models (like ABSA) to avoid runtime delay."""
        self._get_absa_pipeline()

    def _load_model(self) -> None:
        """Load tokenizer and model, then move the model to the target device."""
        try:
            if self._config.mode.startswith("onnx"):
                path = (
                    self._config.onnx_int8_model_path
                    if self._config.mode == "onnx_int8"
                    else self._config.onnx_model_path
                )
                from pathlib import Path
                self._onnx_session = OnnxInferenceSession(
                    path,
                    str(Path(path).parent),
                    self._config.max_length,
                )
                
                # Load sarcasm ONNX model if available
                sarcasm_path = path.replace("sentiment_", "sarcasm_")
                try:
                    self._sarcasm_onnx_session = OnnxInferenceSession(
                        sarcasm_path,
                        str(Path(sarcasm_path).parent),
                        self._config.max_length,
                    )
                except Exception as exc:
                    logger.warning(f"Failed to load ONNX sarcasm model: {exc}")
                    self._sarcasm_onnx_session = None
            elif self._config.mode == "finetuned":
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._config.finetuned_model_name
                )
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    self._config.finetuned_model_name,
                    num_labels=len(self._config.label_map),
                    use_safetensors=True,
                ).to(self._device)
                self._model = PeftModel.from_pretrained(
                    base_model,
                    self._config.sentiment_adapter_path,
                    adapter_name="sentiment",
                )
                self._model.load_adapter(
                    self._config.sarcasm_adapter_path,
                    adapter_name="sarcasm",
                )
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(self._config.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self._config.model_name,
                    use_safetensors=True,
                ).to(self._device)
                
            if self._model is not None:
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
        self,
        texts: str | list[str],
        *,
        padding: bool = False,
        adapter_name: str | None = None,
    ) -> torch.Tensor:
        """Run the tokenizer/model stack and return class probabilities."""
        if self._onnx_session is not None:
            texts_list = [texts] if isinstance(texts, str) else texts
            probs = self._onnx_session.predict_probs(texts_list)
            return torch.from_numpy(probs)

        if adapter_name is not None and hasattr(self._model, "set_adapter"):
            self._model.set_adapter(adapter_name)

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

    def _predict_sarcasm_flag(self, text: str) -> bool:
        """Predict sarcasm from the dedicated finetuned adapter when available.

        Note: The sarcasm adapter is trained with a 3-logit head (to share the same
        backbone as the sentiment adapter), but only the first two logits correspond
        to the binary irony task (0 = non_irony, 1 = irony).  The third logit is
        intentionally discarded here.  See run_finetuning._model_num_labels for the
        full rationale.
        """
        if self._config.mode.startswith("onnx"):
            if getattr(self, "_sarcasm_onnx_session", None) is not None:
                texts_list = [text]
                probs = self._sarcasm_onnx_session.predict_probs(texts_list)[0][:2]
                return bool(probs.argmax() == 1)
            return False

        if self._config.mode != "finetuned":
            return False

        # Slice [:2] discards the unused third logit — see docstring above.
        probs = self._predict_probabilities(text, adapter_name="sarcasm")[0][:2]
        return bool(probs.argmax().item() == 1)

    def predict_single(self, text: str, lang: str = "en") -> PredictionResult:
        """Predict overall sentiment and extract aspect sentiment when available."""
        self._check_language(lang)
        adapter_name = "sentiment" if self._config.mode == "finetuned" else None
        probs = self._predict_probabilities(text, adapter_name=adapter_name)[0]
        pred_idx = probs.argmax().item()

        return PredictionResult(
            sentiment=self._config.label_map[pred_idx],
            confidence=round(probs[pred_idx].item(), 4),
            aspects=self._extract_aspects(text),
            sarcasm_flag=self._predict_sarcasm_flag(text),
        )

    def predict_batch(
        self,
        texts: list[str],
        lang: str = "en",
        *,
        batch_size: int | None = None,
        skip_absa: bool = False,
        skip_sarcasm: bool = False,
    ) -> list[PredictionResult]:
        """Predict sentiment for a batch of texts using chunked processing.

        Args:
            texts: Input texts to classify.
            lang: Language code (must be supported).
            batch_size: Number of texts per forward pass. None uses config default.
            skip_absa: When True, skip aspect extraction (aspects=[]).
            skip_sarcasm: When True, skip per-sample sarcasm prediction
                (sarcasm_flag=False). Useful for benchmarking sentiment-only
                throughput without the sequential sarcasm overhead.
        """
        if not texts:
            return []

        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self._check_language(lang)

        resolved_batch_size = batch_size if batch_size is not None else self._config.batch_size
        if resolved_batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {resolved_batch_size}")

        total_chunks = -(-len(texts) // resolved_batch_size)  # ceiling division

        logger.info(
            "predict_batch: %d texts, batch_size=%d, chunks=%d, skip_absa=%s, skip_sarcasm=%s",
            len(texts),
            resolved_batch_size,
            total_chunks,
            skip_absa,
            skip_sarcasm,
        )

        all_probs: list[torch.Tensor] = []
        for i in range(0, len(texts), resolved_batch_size):
            chunk = texts[i : i + resolved_batch_size]
            probs = self._predict_probabilities(
                chunk,
                padding=True,
                adapter_name="sentiment" if self._config.mode == "finetuned" else None,
            )
            all_probs.append(probs)

            chunk_number = i // resolved_batch_size + 1
            if chunk_number % 10 == 0:
                logger.debug(
                    "predict_batch: processed chunk %d/%d",
                    chunk_number,
                    total_chunks,
                )

        combined_probs = torch.cat(all_probs, dim=0)

        results: list[PredictionResult] = []
        n_texts = len(texts)
        for idx in range(n_texts):
            pred_idx = combined_probs[idx].argmax().item()
            aspects = [] if skip_absa else self._extract_aspects(texts[idx])
            sarcasm_flag = (
                False if skip_sarcasm else self._predict_sarcasm_flag(texts[idx])
            )
            results.append(
                PredictionResult(
                    sentiment=self._config.label_map[pred_idx],
                    confidence=round(combined_probs[idx][pred_idx].item(), 4),
                    aspects=aspects,
                    sarcasm_flag=sarcasm_flag,
                )
            )

            # Progress logging for long-running post-processing
            processed = idx + 1
            if processed % 200 == 0 or processed == n_texts:
                logger.info(
                    "predict_batch: post-processing %d/%d samples",
                    processed,
                    n_texts,
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
                hypothesis_template=self._config.absa_aspect_template,
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
                    hypothesis_template=self._config.absa_sentiment_template.format(
                        aspect=aspect_name
                    ),
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
        import numpy as np

        self._check_language(lang)
        predicted_probs = self._predict_probabilities(text)[0]
        predicted_class_idx = int(predicted_probs.argmax().item())
        
        def predict_func(texts):
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            if isinstance(texts, str):
                texts = [texts]
            probs = self._predict_probabilities(texts, padding=True)
            if hasattr(probs, "cpu"):
                return probs.cpu().numpy()
            return probs

        tokenizer = self._tokenizer if self._tokenizer is not None else getattr(self._onnx_session, "tokenizer", None)
        if tokenizer is None:
            raise ModelError("No tokenizer available for SHAP explanation")

        # Suppress parallelism warning during SHAP tokenization
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        explainer = shap.Explainer(predict_func, tokenizer)
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
        return self._model is not None or self._onnx_session is not None

    @property
    def device(self) -> torch.device:
        return self._device

    def __repr__(self) -> str:
        return (
            f"BaselineModelInference("
            f"model={self._config.model_name}, device={self._device})"
        )
