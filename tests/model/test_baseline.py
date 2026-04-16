"""Tests for BaselineModelInference — all HuggingFace calls are mocked."""
import torch
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from contracts.model_interface import PredictionResult, SHAPResult, ModelInference
from contracts.errors import UnsupportedLanguageError, ModelError
from src.model.config import ModelConfig


# ── Helpers ────────────────────────────────────────────────────


class _MockBatchEncoding(dict):
    """Minimal dict-like object with HuggingFace-style `.to()` support."""

    def to(self, device):
        return _MockBatchEncoding(
            {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in self.items()
            }
        )


def _make_mock_logits(batch_size: int = 1) -> torch.Tensor:
    """Create fake logits: shape (batch_size, 3). Class 2 (positive) wins."""
    logits = torch.tensor([[0.1, 0.2, 0.9]] * batch_size)
    return logits


def _build_model_with_mocks(config=None, device=None):
    """Patch HuggingFace and build a BaselineModelInference."""
    config = config or ModelConfig()
    device = device or torch.device("cpu")

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = _MockBatchEncoding(
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
    )

    mock_hf_model = MagicMock()
    mock_hf_model.to.return_value = mock_hf_model
    mock_hf_model.hf_device_map = None

    @dataclass
    class FakeOutput:
        logits: torch.Tensor

    mock_hf_model.__call__ = MagicMock(
        return_value=FakeOutput(logits=_make_mock_logits(1))
    )
    # Make model(**inputs) work
    mock_hf_model.return_value = FakeOutput(logits=_make_mock_logits(1))

    with patch("src.model.baseline.AutoTokenizer") as MockTokenizer, \
         patch("src.model.baseline.AutoModelForSequenceClassification") as MockModel:
        MockTokenizer.from_pretrained.return_value = mock_tokenizer
        MockModel.from_pretrained.return_value = mock_hf_model

        from src.model.baseline import BaselineModelInference
        model = BaselineModelInference(config=config, device=device)

    return model, mock_tokenizer, mock_hf_model


# ── Interface Compliance ───────────────────────────────────────

class TestInterfaceCompliance:
    def test_is_subclass_of_model_inference(self):
        from src.model.baseline import BaselineModelInference
        assert issubclass(BaselineModelInference, ModelInference)


# ── Properties ─────────────────────────────────────────────────

class TestProperties:
    def test_is_loaded_true_after_init(self):
        model, _, _ = _build_model_with_mocks()
        assert model.is_loaded is True

    def test_supported_languages_contains_en(self):
        model, _, _ = _build_model_with_mocks()
        assert "en" in model.supported_languages

    def test_supported_languages_returns_list(self):
        model, _, _ = _build_model_with_mocks()
        assert isinstance(model.supported_languages, list)

    def test_repr_contains_model_name(self):
        model, _, _ = _build_model_with_mocks()
        assert "cardiffnlp" in repr(model)


# ── predict_single ─────────────────────────────────────────────

class TestPredictSingle:
    def test_returns_prediction_result(self):
        model, _, _ = _build_model_with_mocks()
        result = model.predict_single("The food was great")
        assert isinstance(result, PredictionResult)

    def test_sentiment_is_valid_label(self):
        model, _, _ = _build_model_with_mocks()
        result = model.predict_single("The food was great")
        assert result.sentiment in ("positive", "negative", "neutral")

    def test_confidence_in_valid_range(self):
        model, _, _ = _build_model_with_mocks()
        result = model.predict_single("The food was great")
        assert 0.0 <= result.confidence <= 1.0

    def test_aspects_empty_for_baseline(self):
        model, _, _ = _build_model_with_mocks()
        result = model.predict_single("The food was great")
        assert result.aspects == []

    def test_sarcasm_flag_false_for_baseline(self):
        model, _, _ = _build_model_with_mocks()
        result = model.predict_single("The food was great")
        assert result.sarcasm_flag is False

    def test_unsupported_language_raises(self):
        model, _, _ = _build_model_with_mocks()
        with pytest.raises(UnsupportedLanguageError):
            model.predict_single("text", lang="fr")

    def test_tokenizer_called_with_expected_arguments(self):
        model, mock_tokenizer, _ = _build_model_with_mocks()

        model.predict_single("The food was great")

        mock_tokenizer.assert_called_once_with(
            "The food was great",
            return_tensors="pt",
            truncation=True,
            max_length=ModelConfig().max_length,
        )


# ── predict_batch ──────────────────────────────────────────────

class TestPredictBatch:
    def test_empty_list_returns_empty(self):
        model, _, _ = _build_model_with_mocks()
        assert model.predict_batch([]) == []

    def test_returns_correct_length(self):
        model, mock_tok, mock_hf = _build_model_with_mocks()
        # Re-configure mock for batch of 3
        @dataclass
        class FakeOutput:
            logits: torch.Tensor
        mock_hf.return_value = FakeOutput(logits=_make_mock_logits(3))

        results = model.predict_batch(["a", "b", "c"])
        assert len(results) == 3

    def test_all_results_are_prediction_result(self):
        model, mock_tok, mock_hf = _build_model_with_mocks()
        @dataclass
        class FakeOutput:
            logits: torch.Tensor
        mock_hf.return_value = FakeOutput(logits=_make_mock_logits(2))

        results = model.predict_batch(["a", "b"])
        for r in results:
            assert isinstance(r, PredictionResult)
            assert r.aspects == []
            assert r.sarcasm_flag is False

    def test_unsupported_language_raises(self):
        model, _, _ = _build_model_with_mocks()
        with pytest.raises(UnsupportedLanguageError):
            model.predict_batch(["text"], lang="de")

    def test_tokenizer_called_with_expected_arguments(self):
        model, mock_tokenizer, mock_hf = _build_model_with_mocks()

        @dataclass
        class FakeOutput:
            logits: torch.Tensor

        mock_hf.return_value = FakeOutput(logits=_make_mock_logits(2))

        model.predict_batch(["a", "b"])

        mock_tokenizer.assert_called_with(
            ["a", "b"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=ModelConfig().max_length,
        )


# ── SHAP Explainability ────────────────────────────────────────

class TestSHAPExplanation:
    @patch("src.model.baseline.hf_pipeline")
    @patch("src.model.baseline.shap")
    def test_returns_shap_result(self, mock_shap, mock_pipeline):
        model, _, _ = _build_model_with_mocks()

        # Configure mock shap explainer
        mock_pipeline.return_value = MagicMock()
        mock_explainer = MagicMock()
        mock_shap.Explainer.return_value = mock_explainer
        mock_shap_values = MagicMock()
        mock_shap_values.data = [["token1", "token2"]]
        import numpy as np
        mock_shap_values.values = [np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])]
        mock_shap_values.base_values = [np.array([0.1, 0.2, 0.7])]
        mock_explainer.return_value = mock_shap_values

        result = model.get_shap_explanation("test text")
        assert isinstance(result, SHAPResult)
        assert len(result.tokens) == 2
        assert len(result.shap_values) == 2

    @patch("src.model.baseline.hf_pipeline")
    @patch("src.model.baseline.shap")
    def test_returns_values_for_predicted_class(self, mock_shap, mock_pipeline):
        model, _, mock_hf_model = _build_model_with_mocks()
        mock_pipeline.return_value = MagicMock()

        @dataclass
        class FakeOutput:
            logits: torch.Tensor

        # Predicted class is index 0, even though SHAP totals favor index 2.
        mock_hf_model.return_value = FakeOutput(
            logits=torch.tensor([[0.9, 0.2, 0.1]])
        )

        mock_explainer = MagicMock()
        mock_shap.Explainer.return_value = mock_explainer
        mock_shap_values = MagicMock()
        mock_shap_values.data = [["token1", "token2"]]
        import numpy as np
        mock_shap_values.values = [
            np.array([[0.1, 0.2, 5.0], [0.3, 0.4, 6.0]])
        ]
        mock_shap_values.base_values = [np.array([0.6, 0.2, 0.1])]
        mock_explainer.return_value = mock_shap_values

        result = model.get_shap_explanation("test text")

        assert result.shap_values == [0.1, 0.3]
        assert result.base_value == 0.6


# ── Error Handling ─────────────────────────────────────────────

class TestErrorHandling:
    def test_model_load_failure_raises_model_error(self):
        """If HuggingFace model fails to load, raise ModelError."""
        with patch("src.model.baseline.AutoTokenizer") as MockTokenizer, \
             patch("src.model.baseline.AutoModelForSequenceClassification") as MockModel:
            MockTokenizer.from_pretrained.side_effect = OSError("network error")

            from src.model.baseline import BaselineModelInference
            with pytest.raises(ModelError, match="Failed to load model"):
                BaselineModelInference(
                    config=ModelConfig(), device=torch.device("cpu")
                )

    def test_pipeline_initialization_failure_propagates(self):
        model, _, _ = _build_model_with_mocks()

        with patch(
            "src.model.baseline.hf_pipeline",
            side_effect=RuntimeError("pipeline setup failed"),
        ):
            with pytest.raises(RuntimeError, match="pipeline setup failed"):
                model._get_classification_pipeline()
