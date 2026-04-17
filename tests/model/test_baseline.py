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


def _expected_pipeline_device(device: torch.device):
    """Mirror the pipeline device mapping used by the model."""
    if device.type == "cuda" and device.index is not None:
        return device.index
    if device.type == "cpu":
        return -1
    return device


def _assert_absa_pipeline_factory_call(mock_pipeline, model):
    """Verify prediction-time lazy loading uses the ABSA zero-shot pipeline."""
    mock_pipeline.assert_called_once()
    args, kwargs = mock_pipeline.call_args
    assert args == ("zero-shot-classification",)
    assert kwargs["model"] == model._config.absa_model_name
    assert kwargs["device"] == _expected_pipeline_device(model.device)


def _assert_aspect_detection_call(call, text):
    """Verify the first zero-shot pass extracts ABSA aspects."""
    assert call["text"] == text
    assert call["candidate_labels"] == list(ModelConfig().absa_categories)
    assert call["hypothesis_template"] == ModelConfig().absa_aspect_template
    assert call["multi_label"] is True


def _assert_aspect_sentiment_call(call, text, aspect_name):
    """Verify the per-aspect zero-shot pass scores sentiment labels."""
    assert call["text"] == text
    assert call["candidate_labels"] == ["positive", "negative", "neutral"]
    expected_template = ModelConfig().absa_sentiment_template.format(aspect=aspect_name)
    assert call["hypothesis_template"] == expected_template
    assert call["multi_label"] is False


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


class FakeZeroShotPipeline:
    """Simple fake for zero-shot aspect and sentiment pipeline calls."""

    def __init__(self, aspect_result=None, sentiment_result=None):
        self.aspect_result = aspect_result or {
            "labels": ["food", "service", "ambiance", "price", "location", "general"],
            "scores": [0.85, 0.78, 0.12, 0.08, 0.05, 0.10],
        }
        self.sentiment_result = sentiment_result or {
            "labels": ["positive", "negative", "neutral"],
            "scores": [0.82, 0.12, 0.06],
        }
        self._call_count = 0
        self.calls = []
        self.pipeline_factory_calls = []
        self.zero_shot_factory_kwargs = None
        self.sentiment_pipeline = MagicMock()

    def __call__(
        self, text, candidate_labels, hypothesis_template="", multi_label=False
    ):
        self._call_count += 1
        self.calls.append(
            {
                "text": text,
                "candidate_labels": list(candidate_labels),
                "hypothesis_template": hypothesis_template,
                "multi_label": multi_label,
            }
        )
        if multi_label:
            return self.aspect_result
        return self.sentiment_result


def _build_model_with_mocks_absa(
    config=None,
    device=None,
    fake_zero_shot: FakeZeroShotPipeline | None = None,
):
    """Patch HuggingFace and build a BaselineModelInference with ABSA fakes."""
    config = config or ModelConfig()
    device = device or torch.device("cpu")
    fake_zero_shot = fake_zero_shot or FakeZeroShotPipeline()

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
    mock_hf_model.return_value = FakeOutput(logits=_make_mock_logits(1))

    def fake_pipeline_factory(task, **kwargs):
        fake_zero_shot.pipeline_factory_calls.append(
            {"task": task, "kwargs": dict(kwargs)}
        )
        if task == "sentiment-analysis":
            return fake_zero_shot.sentiment_pipeline
        if task == "zero-shot-classification":
            missing_kwargs = {"model", "device"} - set(kwargs)
            if missing_kwargs:
                raise AssertionError(
                    f"Missing zero-shot pipeline kwargs: {sorted(missing_kwargs)}"
                )
            fake_zero_shot.zero_shot_factory_kwargs = dict(kwargs)
            return fake_zero_shot
        raise ValueError(f"Unexpected pipeline task: {task}")

    with patch("src.model.baseline.AutoTokenizer") as MockTokenizer, \
         patch("src.model.baseline.AutoModelForSequenceClassification") as MockModel, \
         patch("src.model.baseline.hf_pipeline", side_effect=fake_pipeline_factory):
        MockTokenizer.from_pretrained.return_value = mock_tokenizer
        MockModel.from_pretrained.return_value = mock_hf_model

        from src.model.baseline import BaselineModelInference
        model = BaselineModelInference(config=config, device=device)

    return model, fake_zero_shot


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

    def test_device_property_returns_device(self):
        model, _, _ = _build_model_with_mocks()
        assert model.device == torch.device("cpu")

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

    def test_aspects_returned_from_absa(self):
        fake_zero_shot = FakeZeroShotPipeline()
        model, _ = _build_model_with_mocks_absa(fake_zero_shot=fake_zero_shot)

        with patch("src.model.baseline.hf_pipeline", return_value=fake_zero_shot) as mock_pipeline:
            result = model.predict_single("The food was amazing but service was terrible")

        _assert_absa_pipeline_factory_call(mock_pipeline, model)

        assert len(result.aspects) > 0
        assert len(fake_zero_shot.calls) == 3
        _assert_aspect_detection_call(
            fake_zero_shot.calls[0],
            "The food was amazing but service was terrible",
        )
        _assert_aspect_sentiment_call(
            fake_zero_shot.calls[1],
            "The food was amazing but service was terrible",
            "food",
        )
        _assert_aspect_sentiment_call(
            fake_zero_shot.calls[2],
            "The food was amazing but service was terrible",
            "service",
        )

        allowed_aspects = set(ModelConfig().absa_categories)
        allowed_sentiments = {"positive", "negative", "neutral"}
        for aspect in result.aspects:
            assert aspect.aspect in allowed_aspects
            assert aspect.sentiment in allowed_sentiments
            assert 0.0 <= aspect.confidence <= 1.0

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
            assert isinstance(r.aspects, list)
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

    def test_chunking_processes_all_texts(self):
        """100 texts → 100 results regardless of chunk size."""
        from unittest.mock import MagicMock
        model, mock_tok, mock_hf = _build_model_with_mocks()

        def side_effect(*args, **kwargs):
            batch = mock_tok.call_args[0][0]
            n = len(batch) if isinstance(batch, list) else 1
            return MagicMock(logits=_make_mock_logits(n))

        mock_hf.side_effect = side_effect

        texts = [f"text {i}" for i in range(100)]
        results = model.predict_batch(texts, batch_size=32, skip_absa=True)
        assert len(results) == 100

    def test_chunking_calls_model_per_chunk(self):
        """batch_size=32, 100 texts → exactly 4 forward passes."""
        from unittest.mock import MagicMock
        model, mock_tok, mock_hf = _build_model_with_mocks()

        def side_effect(*args, **kwargs):
            batch = mock_tok.call_args[0][0]
            n = len(batch) if isinstance(batch, list) else 1
            return MagicMock(logits=_make_mock_logits(n))

        mock_hf.side_effect = side_effect

        texts = [f"text {i}" for i in range(100)]
        model.predict_batch(texts, batch_size=32, skip_absa=True)
        assert mock_hf.call_count == 4  # ceil(100/32) = 4

    def test_custom_batch_size_override(self):
        """batch_size=16, 100 texts → exactly 7 forward passes."""
        from unittest.mock import MagicMock
        model, mock_tok, mock_hf = _build_model_with_mocks()

        def side_effect(*args, **kwargs):
            batch = mock_tok.call_args[0][0]
            n = len(batch) if isinstance(batch, list) else 1
            return MagicMock(logits=_make_mock_logits(n))

        mock_hf.side_effect = side_effect

        texts = [f"text {i}" for i in range(100)]
        model.predict_batch(texts, batch_size=16, skip_absa=True)
        assert mock_hf.call_count == 7  # ceil(100/16) = 7

    def test_default_batch_size_from_config(self):
        """batch_size=None → uses config.batch_size."""
        from unittest.mock import MagicMock
        config = ModelConfig(batch_size=8)
        model, mock_tok, mock_hf = _build_model_with_mocks(config=config)

        def side_effect(*args, **kwargs):
            batch = mock_tok.call_args[0][0]
            n = len(batch) if isinstance(batch, list) else 1
            return MagicMock(logits=_make_mock_logits(n))

        mock_hf.side_effect = side_effect

        texts = [f"text {i}" for i in range(16)]
        model.predict_batch(texts, batch_size=None, skip_absa=True)
        assert mock_hf.call_count == 2  # ceil(16/8) = 2

    def test_skip_absa_returns_empty_aspects(self):
        """skip_absa=True → every result has aspects=[]."""
        from unittest.mock import MagicMock
        model, mock_tok, mock_hf = _build_model_with_mocks()
        mock_hf.return_value = MagicMock(logits=_make_mock_logits(3))

        results = model.predict_batch(["a", "b", "c"], skip_absa=True)
        for r in results:
            assert r.aspects == []

    def test_skip_absa_does_not_call_extract(self):
        """skip_absa=True → _extract_aspects is never called."""
        from unittest.mock import MagicMock
        model, mock_tok, mock_hf = _build_model_with_mocks()
        mock_hf.return_value = MagicMock(logits=_make_mock_logits(2))

        with patch.object(model, "_extract_aspects") as mock_extract:
            model.predict_batch(["a", "b"], skip_absa=True)
            mock_extract.assert_not_called()

    def test_invalid_batch_size_raises(self):
        """batch_size=0 → ValueError."""
        model, _, _ = _build_model_with_mocks()
        with pytest.raises(ValueError, match="batch_size must be positive"):
            model.predict_batch(["text"], batch_size=0)

    def test_negative_batch_size_raises(self):
        """batch_size=-1 → ValueError."""
        model, _, _ = _build_model_with_mocks()
        with pytest.raises(ValueError, match="batch_size must be positive"):
            model.predict_batch(["text"], batch_size=-1)

    def test_batch_size_larger_than_input(self):
        """5 texts with batch_size=32 → single chunk, 5 results."""
        from unittest.mock import MagicMock
        model, mock_tok, mock_hf = _build_model_with_mocks()
        mock_hf.return_value = MagicMock(logits=_make_mock_logits(5))

        results = model.predict_batch(
            ["a", "b", "c", "d", "e"], batch_size=32, skip_absa=True
        )
        assert len(results) == 5
        assert mock_hf.call_count == 1

    def test_result_order_matches_input(self):
        """result[i].sentiment corresponds to texts[i]."""
        from unittest.mock import MagicMock
        model, mock_tok, mock_hf = _build_model_with_mocks()

        # Class 0 (negative) wins for first text, class 2 (positive) for second
        mock_hf.return_value = MagicMock(
            logits=torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.2, 0.9]])
        )

        results = model.predict_batch(["bad text", "good text"], skip_absa=True)
        assert results[0].sentiment == "negative"
        assert results[1].sentiment == "positive"


# ── SHAP Explainability ────────────────────────────────────────

class TestSHAPExplanation:
    @patch("src.model.baseline.hf_pipeline")
    @patch("shap.Explainer")
    def test_returns_shap_result(self, mock_explainer_cls, mock_pipeline):
        model, _, _ = _build_model_with_mocks()

        mock_pipeline.return_value = MagicMock()
        mock_shap_values = MagicMock()
        mock_shap_values.data = [["token1", "token2"]]
        import numpy as np
        mock_shap_values.values = [np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])]
        mock_shap_values.base_values = [np.array([0.1, 0.2, 0.7])]
        mock_explainer_cls.return_value.return_value = mock_shap_values

        result = model.get_shap_explanation("test text")
        assert isinstance(result, SHAPResult)
        assert len(result.tokens) == 2
        assert len(result.shap_values) == 2

    @patch("src.model.baseline.hf_pipeline")
    @patch("shap.Explainer")
    def test_returns_values_for_predicted_class(self, mock_explainer_cls, mock_pipeline):
        model, _, mock_hf_model = _build_model_with_mocks()
        mock_pipeline.return_value = MagicMock()

        @dataclass
        class FakeOutput:
            logits: torch.Tensor

        mock_hf_model.return_value = FakeOutput(
            logits=torch.tensor([[0.9, 0.2, 0.1]])
        )

        mock_shap_values = MagicMock()
        mock_shap_values.data = [["token1", "token2"]]
        import numpy as np
        mock_shap_values.values = [
            np.array([[0.1, 0.2, 5.0], [0.3, 0.4, 6.0]])
        ]
        mock_shap_values.base_values = [np.array([0.6, 0.2, 0.1])]
        mock_explainer_cls.return_value.return_value = mock_shap_values

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


class TestABSA:
    def test_absa_pipeline_not_loaded_before_predict(self):
        model, fake_zero_shot = _build_model_with_mocks_absa()

        assert hasattr(model, "_absa_pipeline")
        assert model._absa_pipeline is None

        with patch(
            "src.model.baseline.hf_pipeline",
            return_value=fake_zero_shot,
        ) as mock_pipeline:
            first = model.predict_single("The food was amazing but service was terrible")
            second = model.predict_single("The food was amazing but service was terrible")

        assert model._absa_pipeline is fake_zero_shot
        _assert_absa_pipeline_factory_call(mock_pipeline, model)
        assert len(first.aspects) > 0
        assert len(second.aspects) > 0

    def test_absa_fallback_on_pipeline_error(self):
        model, _ = _build_model_with_mocks_absa()

        with patch(
            "src.model.baseline.hf_pipeline",
            side_effect=RuntimeError("absa pipeline setup failed"),
        ) as mock_pipeline:
            result = model.predict_single("The food was amazing but service was terrible")

        _assert_absa_pipeline_factory_call(mock_pipeline, model)
        assert result.aspects == []

    def test_absa_threshold_filters_low_scores(self):
        fake_zero_shot = FakeZeroShotPipeline(
            aspect_result={
                "labels": ["food", "service", "ambiance", "price", "location", "general"],
                "scores": [0.44, 0.30, 0.12, 0.08, 0.05, 0.10],
            }
        )
        model, _ = _build_model_with_mocks_absa(fake_zero_shot=fake_zero_shot)

        with patch(
            "src.model.baseline.hf_pipeline",
            return_value=fake_zero_shot,
        ) as mock_pipeline:
            result = model.predict_single("The food was okay but the service was average")

        _assert_absa_pipeline_factory_call(mock_pipeline, model)
        assert result.aspects == []
        assert len(fake_zero_shot.calls) == 1
        _assert_aspect_detection_call(
            fake_zero_shot.calls[0],
            "The food was okay but the service was average",
        )

    def test_absa_per_aspect_sentiment_assigned_correctly(self):
        class PerAspectFakeZeroShotPipeline(FakeZeroShotPipeline):
            def __init__(self):
                super().__init__(
                    aspect_result={
                        "labels": [
                            "food",
                            "service",
                            "ambiance",
                            "price",
                            "location",
                            "general",
                        ],
                        "scores": [0.91, 0.88, 0.12, 0.08, 0.05, 0.04],
                    }
                )

            def __call__(
                self, text, candidate_labels, hypothesis_template="", multi_label=False
            ):
                self._call_count += 1
                self.calls.append(
                    {
                        "text": text,
                        "candidate_labels": list(candidate_labels),
                        "hypothesis_template": hypothesis_template,
                        "multi_label": multi_label,
                    }
                )
                if multi_label:
                    return self.aspect_result
                if "food" in hypothesis_template:
                    return {
                        "labels": ["positive", "neutral", "negative"],
                        "scores": [0.93, 0.04, 0.03],
                    }
                if "service" in hypothesis_template:
                    return {
                        "labels": ["negative", "neutral", "positive"],
                        "scores": [0.89, 0.08, 0.03],
                    }
                raise AssertionError(f"Unexpected sentiment prompt: {hypothesis_template}")

        fake_zero_shot = PerAspectFakeZeroShotPipeline()
        model, _ = _build_model_with_mocks_absa(fake_zero_shot=fake_zero_shot)

        with patch("src.model.baseline.hf_pipeline", return_value=fake_zero_shot) as mock_pipeline:
            result = model.predict_single("The food was amazing but service was terrible")

        _assert_absa_pipeline_factory_call(mock_pipeline, model)
        assert [(aspect.aspect, aspect.sentiment) for aspect in result.aspects] == [
            ("food", "positive"),
            ("service", "negative"),
        ]
        assert len(fake_zero_shot.calls) == 3
        _assert_aspect_detection_call(
            fake_zero_shot.calls[0],
            "The food was amazing but service was terrible",
        )
        _assert_aspect_sentiment_call(
            fake_zero_shot.calls[1],
            "The food was amazing but service was terrible",
            "food",
        )
        _assert_aspect_sentiment_call(
            fake_zero_shot.calls[2],
            "The food was amazing but service was terrible",
            "service",
        )

    def test_predict_batch_includes_aspects(self):
        class PerTextFakeZeroShotPipeline(FakeZeroShotPipeline):
            def __call__(
                self, text, candidate_labels, hypothesis_template="", multi_label=False
            ):
                self._call_count += 1
                self.calls.append(
                    {
                        "text": text,
                        "candidate_labels": list(candidate_labels),
                        "hypothesis_template": hypothesis_template,
                        "multi_label": multi_label,
                    }
                )
                if multi_label:
                    if "slow service" in text:
                        return {
                            "labels": [
                                "service",
                                "ambiance",
                                "food",
                                "price",
                                "location",
                                "general",
                            ],
                            "scores": [0.94, 0.20, 0.12, 0.08, 0.05, 0.04],
                        }
                    return {
                        "labels": [
                            "food",
                            "service",
                            "ambiance",
                            "price",
                            "location",
                            "general",
                        ],
                        "scores": [0.92, 0.21, 0.12, 0.08, 0.05, 0.04],
                    }
                if "service" in hypothesis_template:
                    return {
                        "labels": ["negative", "neutral", "positive"],
                        "scores": [0.91, 0.06, 0.03],
                    }
                if "food" in hypothesis_template:
                    return {
                        "labels": ["positive", "neutral", "negative"],
                        "scores": [0.90, 0.07, 0.03],
                    }
                raise AssertionError(f"Unexpected sentiment prompt: {hypothesis_template}")

        fake_zero_shot = PerTextFakeZeroShotPipeline()
        model, _ = _build_model_with_mocks_absa(fake_zero_shot=fake_zero_shot)
        texts = [
            "The food was amazing but service was terrible",
            "Great ambiance, slow service, fair price",
        ]

        @dataclass
        class FakeOutput:
            logits: torch.Tensor

        model._model.return_value = FakeOutput(logits=_make_mock_logits(2))

        with patch("src.model.baseline.hf_pipeline", return_value=fake_zero_shot) as mock_pipeline:
            results = model.predict_batch(texts)

        _assert_absa_pipeline_factory_call(mock_pipeline, model)
        assert len(results) == 2
        assert [aspect.aspect for aspect in results[0].aspects] == ["food"]
        assert [aspect.aspect for aspect in results[1].aspects] == ["service"]
        _assert_aspect_detection_call(fake_zero_shot.calls[0], texts[0])
        _assert_aspect_sentiment_call(fake_zero_shot.calls[1], texts[0], "food")
        _assert_aspect_detection_call(fake_zero_shot.calls[2], texts[1])
        _assert_aspect_sentiment_call(fake_zero_shot.calls[3], texts[1], "service")

        for result in results:
            for aspect in result.aspects:
                assert aspect.aspect in ModelConfig().absa_categories
                assert aspect.sentiment in ("positive", "negative", "neutral")
                assert 0.0 <= aspect.confidence <= 1.0
