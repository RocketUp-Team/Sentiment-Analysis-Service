# Baseline Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `BaselineModelInference` — a pre-trained RoBERTa model that fulfills the `ModelInference` contract — with device auto-detection, SHAP explainability, evaluation CLI, and MLflow logging.

**Architecture:** Four focused modules in `src/model/`: device detection (`device.py`), configuration dataclass (`config.py`), core inference class (`baseline.py`), and evaluation CLI (`evaluate.py`). The `BaselineModelInference` class implements the abstract `ModelInference` interface from `contracts/model_interface.py`. All unit tests mock the HuggingFace model so they run fast without downloading weights.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace Transformers, SHAP, MLflow, scikit-learn, matplotlib, pytest

---

## File Structure

| File                                       | Responsibility                                                         |
| ------------------------------------------ | ---------------------------------------------------------------------- |
| **Create:** `src/model/__init__.py`        | Package exports: `BaselineModelInference`, `get_device`, `ModelConfig` |
| **Create:** `src/model/device.py`          | `get_device()` → auto-detect cuda/mps/cpu                              |
| **Create:** `src/model/config.py`          | `ModelConfig` frozen dataclass with model name, label map, etc.        |
| **Create:** `src/model/baseline.py`        | `BaselineModelInference(ModelInference)` — predict, batch, SHAP        |
| **Create:** `src/model/evaluate.py`        | CLI: evaluate on processed data, log to MLflow                         |
| **Create:** `tests/model/__init__.py`      | Test package init                                                      |
| **Create:** `tests/model/test_device.py`   | Unit tests for device detection                                        |
| **Create:** `tests/model/test_config.py`   | Unit tests for ModelConfig                                             |
| **Create:** `tests/model/test_baseline.py` | Unit tests for BaselineModelInference (mocked model)                   |
| **Create:** `tests/model/test_evaluate.py` | Unit tests for evaluation metrics computation                          |
| **Modify:** `requirements.txt`             | Add torch, transformers, shap, matplotlib                              |
| **Modify:** `params.yaml`                  | Add `mlflow.model_experiment_name` key                                 |
| **Modify:** `dvc.yaml`                     | Add `evaluate_baseline` stage                                          |

---

### Task 1: Add Dependencies

**Files:**

- Modify: `requirements.txt`

- [ ] **Step 1: Add model dependencies to requirements.txt**

Add these lines to the end of `requirements.txt` (before the trailing blank line):

```
torch>=2.0.0
transformers>=4.30.0
shap>=0.42.0
matplotlib>=3.7.0
```

> `mlflow` is already present (`mlflow>=2.0,<3.0`). `scikit-learn` is already present. `pandas` is already present.

- [ ] **Step 2: Install dependencies**

Run:

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/pip install -r requirements.txt
```

Expected: all packages install successfully. torch ~2GB download, transformers ~few hundred MB.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add torch, transformers, shap, matplotlib deps"
```

---

### Task 2: Device Detection Module

**Files:**

- Create: `src/model/__init__.py`
- Create: `src/model/device.py`
- Create: `tests/model/__init__.py`
- Create: `tests/model/test_device.py`

- [ ] **Step 1: Write failing tests for `get_device()`**

Create `tests/model/__init__.py`:

```python
"""Tests for src.model package."""
```

Create `tests/model/test_device.py`:

```python
import torch
from unittest.mock import patch

from src.model.device import get_device


class TestGetDevice:
    def test_returns_torch_device(self):
        """get_device() must return a torch.device instance."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_returns_cuda_when_available(self):
        with patch("src.model.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.device = torch.device
            device = get_device()
        assert device == torch.device("cuda")

    def test_returns_mps_when_no_cuda(self):
        with patch("src.model.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = True
            mock_torch.device = torch.device
            # hasattr on mock returns True by default
            device = get_device()
        assert device == torch.device("mps")

    def test_returns_cpu_as_fallback(self):
        with patch("src.model.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            mock_torch.device = torch.device
            device = get_device()
        assert device == torch.device("cpu")

    def test_returns_cpu_when_no_mps_attr(self):
        with patch("src.model.device.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            # Remove mps from backends
            del mock_torch.backends.mps
            mock_torch.device = torch.device
            device = get_device()
        assert device == torch.device("cpu")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_device.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.model'`

- [ ] **Step 3: Implement `device.py` and `__init__.py`**

Create `src/model/__init__.py`:

```python
"""Model inference package: baseline, device detection, and configuration."""

from src.model.device import get_device

__all__ = ["get_device"]
```

> Note: `BaselineModelInference` will be added to `__init__.py` in Task 4 after the class is created.

Create `src/model/device.py`:

```python
"""Auto-detect the best available compute device."""

import torch


def get_device() -> torch.device:
    """Auto-detect best available device.

    Priority: CUDA (Colab/NVIDIA) > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_device.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/model/__init__.py src/model/device.py tests/model/__init__.py tests/model/test_device.py
git commit -m "feat(model): add device auto-detection module"
```

---

### Task 3: Model Configuration Dataclass

**Files:**

- Create: `src/model/config.py`
- Create: `tests/model/test_config.py`

- [ ] **Step 1: Write failing tests for `ModelConfig`**

Create `tests/model/test_config.py`:

```python
from src.model.config import ModelConfig


class TestModelConfig:
    def test_default_model_name(self):
        config = ModelConfig()
        assert config.model_name == "cardiffnlp/twitter-roberta-base-sentiment-latest"

    def test_default_max_length(self):
        config = ModelConfig()
        assert config.max_length == 512

    def test_default_language(self):
        config = ModelConfig()
        assert config.default_lang == "en"
        assert "en" in config.supported_languages

    def test_label_map_has_three_classes(self):
        config = ModelConfig()
        assert len(config.label_map) == 3
        assert set(config.label_map.values()) == {"negative", "neutral", "positive"}

    def test_label_map_indices(self):
        """cardiffnlp model: 0=negative, 1=neutral, 2=positive."""
        config = ModelConfig()
        assert config.label_map[0] == "negative"
        assert config.label_map[1] == "neutral"
        assert config.label_map[2] == "positive"

    def test_frozen_immutability(self):
        """Config should be immutable (frozen dataclass)."""
        config = ModelConfig()
        try:
            config.model_name = "other"
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass  # Expected — frozen dataclass

    def test_custom_model_name(self):
        config = ModelConfig(model_name="custom/model")
        assert config.model_name == "custom/model"

    def test_label_map_matches_project_sentiment_labels(self):
        """Ensure config labels match the project's expected sentiment labels."""
        config = ModelConfig()
        project_labels = {"positive", "negative", "neutral"}
        assert set(config.label_map.values()) == project_labels
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.model.config'`

- [ ] **Step 3: Implement `config.py`**

Create `src/model/config.py`:

```python
"""Model configuration dataclass."""

from dataclasses import dataclass, field


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
    label_map: dict[int, str] = field(
        default_factory=lambda: {
            0: "negative",
            1: "neutral",
            2: "positive",
        }
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_config.py -v
```

Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/model/config.py tests/model/test_config.py
git commit -m "feat(model): add ModelConfig dataclass"
```

---

### Task 4: BaselineModelInference — Core Class

**Files:**

- Create: `src/model/baseline.py`
- Create: `tests/model/test_baseline.py`
- Modify: `src/model/__init__.py`

- [ ] **Step 1: Write failing tests for `BaselineModelInference`**

Create `tests/model/test_baseline.py`:

```python
"""Tests for BaselineModelInference — all HuggingFace calls are mocked."""
import torch
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from contracts.model_interface import PredictionResult, SHAPResult, ModelInference
from contracts.errors import UnsupportedLanguageError, ModelError
from src.model.config import ModelConfig


# ── Helpers ────────────────────────────────────────────────────

def _make_mock_logits(batch_size: int = 1) -> torch.Tensor:
    """Create fake logits: shape (batch_size, 3). Class 2 (positive) wins."""
    logits = torch.tensor([[0.1, 0.2, 0.9]] * batch_size)
    return logits


def _build_model_with_mocks(config=None, device=None):
    """Patch HuggingFace and build a BaselineModelInference."""
    config = config or ModelConfig()
    device = device or torch.device("cpu")

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    # Make tokenizer output moveable to device
    mock_tokenizer.return_value.to = MagicMock(
        return_value=mock_tokenizer.return_value
    )

    mock_hf_model = MagicMock()
    mock_hf_model.to.return_value = mock_hf_model

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
        mock_tok.return_value.to = MagicMock(return_value=mock_tok.return_value)

        results = model.predict_batch(["a", "b", "c"])
        assert len(results) == 3

    def test_all_results_are_prediction_result(self):
        model, mock_tok, mock_hf = _build_model_with_mocks()
        @dataclass
        class FakeOutput:
            logits: torch.Tensor
        mock_hf.return_value = FakeOutput(logits=_make_mock_logits(2))
        mock_tok.return_value.to = MagicMock(return_value=mock_tok.return_value)

        results = model.predict_batch(["a", "b"])
        for r in results:
            assert isinstance(r, PredictionResult)
            assert r.aspects == []
            assert r.sarcasm_flag is False

    def test_unsupported_language_raises(self):
        model, _, _ = _build_model_with_mocks()
        with pytest.raises(UnsupportedLanguageError):
            model.predict_batch(["text"], lang="de")


# ── SHAP Explainability ────────────────────────────────────────

class TestSHAPExplanation:
    @patch("src.model.baseline.shap")
    def test_returns_shap_result(self, mock_shap):
        model, _, _ = _build_model_with_mocks()

        # Configure mock shap explainer
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_baseline.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.model.baseline'`

- [ ] **Step 3: Implement `baseline.py`**

Create `src/model/baseline.py`:

```python
"""BaselineModelInference — pre-trained RoBERTa sentiment classification."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline as hf_pipeline

from contracts.model_interface import ModelInference, PredictionResult, SHAPResult
from contracts.errors import UnsupportedLanguageError, ModelError
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
        self._hf_pipeline = None  # Lazy-init for SHAP
        self._load_model()

    def _load_model(self) -> None:
        """Load tokenizer + model, move to device."""
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._config.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._config.model_name
            ).to(self._device)
            self._model.eval()
        except Exception as e:
            raise ModelError(f"Failed to load model: {e}") from e

    def _check_language(self, lang: str) -> None:
        if lang not in self._config.supported_languages:
            raise UnsupportedLanguageError(lang)

    # ── Inference ──────────────────────────────────────────────

    def predict_single(self, text: str, lang: str = "en") -> PredictionResult:
        """Predict overall sentiment. aspects=[] for baseline."""
        self._check_language(lang)
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._config.max_length,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0]
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
        """Batch predict — tokenize batch for efficiency."""
        if not texts:
            return []

        self._check_language(lang)
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._config.max_length,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        results = []
        for i in range(len(texts)):
            pred_idx = probs[i].argmax().item()
            results.append(
                PredictionResult(
                    sentiment=self._config.label_map[pred_idx],
                    confidence=round(probs[i][pred_idx].item(), 4),
                    aspects=[],
                    sarcasm_flag=False,
                )
            )
        return results

    # ── SHAP Explainability ────────────────────────────────────

    def _get_classification_pipeline(self):
        """Lazy-init HuggingFace pipeline for SHAP explainer."""
        if self._hf_pipeline is None:
            self._hf_pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self._model,
                tokenizer=self._tokenizer,
                device=self._device,
                top_k=None,
            )
        return self._hf_pipeline

    def get_shap_explanation(self, text: str, lang: str = "en") -> SHAPResult:
        """SHAP values per token — for explainability UI."""
        import shap

        self._check_language(lang)
        pipe = self._get_classification_pipeline()
        explainer = shap.Explainer(pipe)
        shap_values = explainer([text])

        import numpy as np

        class_idx = int(
            np.argmax(
                shap_values.base_values[0] + shap_values.values[0].sum(axis=0)
            )
        )

        tokens = shap_values.data[0].tolist()
        values = shap_values.values[0][:, class_idx].tolist()
        base = float(shap_values.base_values[0][class_idx])

        return SHAPResult(
            tokens=tokens,
            shap_values=values,
            base_value=base,
        )

    # ── Properties ─────────────────────────────────────────────

    @property
    def supported_languages(self) -> list[str]:
        return list(self._config.supported_languages)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def __repr__(self) -> str:
        return (
            f"BaselineModelInference("
            f"model={self._config.model_name}, device={self._device})"
        )
```

- [ ] **Step 4: Update `src/model/__init__.py` to export `BaselineModelInference`**

Replace `src/model/__init__.py` with:

```python
"""Model inference package: baseline, device detection, and configuration."""

from src.model.baseline import BaselineModelInference
from src.model.config import ModelConfig
from src.model.device import get_device

__all__ = ["BaselineModelInference", "ModelConfig", "get_device"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_baseline.py -v
```

Expected: 16 passed

- [ ] **Step 6: Run all model tests together**

Run:

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/ -v
```

Expected: all tests pass (device + config + baseline)

- [ ] **Step 7: Commit**

```bash
git add src/model/baseline.py src/model/__init__.py tests/model/test_baseline.py
git commit -m "feat(model): add BaselineModelInference with mocked tests"
```

---

### Task 5: Evaluation Functions & MLflow Logging

**Files:**

- Create: `src/model/evaluate.py`
- Create: `tests/model/test_evaluate.py`
- Modify: `params.yaml`

- [ ] **Step 1: Add `model_experiment_name` to `params.yaml`**

Add the `model_experiment_name` key under the existing `mlflow` section in `params.yaml`. The `mlflow` section (lines 29-31) should become:

```yaml
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "data_preprocessing"
  model_experiment_name: "sentiment_baseline"
```

- [ ] **Step 2: Write failing tests for `evaluate_on_dataset`**

Create `tests/model/test_evaluate.py`:

```python
"""Tests for evaluation metrics computation — model predictions are mocked."""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from contracts.model_interface import PredictionResult


# ── Test Data ──────────────────────────────────────────────────

def _make_test_df() -> pd.DataFrame:
    """Minimal sentences DataFrame with known labels."""
    return pd.DataFrame(
        {
            "text": [
                "The food was great",
                "Terrible service",
                "It was okay",
                "Amazing pasta",
                "Bad experience",
                "Normal place",
            ],
            "sentiment": [
                "positive",
                "negative",
                "neutral",
                "positive",
                "negative",
                "neutral",
            ],
            "split": ["test"] * 6,
        }
    )


def _make_mock_model(predictions: list[str]) -> MagicMock:
    """Create a mock model that returns fixed predictions."""
    mock = MagicMock()
    results = [
        PredictionResult(
            sentiment=s, confidence=0.9, aspects=[], sarcasm_flag=False
        )
        for s in predictions
    ]
    mock.predict_batch.return_value = results
    return mock


# ── evaluate_on_dataset ───────────────────────────────────────

class TestEvaluateOnDataset:
    def test_returns_early_on_empty_split(self):
        from src.model.evaluate import evaluate_on_dataset
        df = _make_test_df()
        mock_model = _make_mock_model([])
        # Try evaluating on a split that does not exist
        metrics = evaluate_on_dataset(mock_model, df, split="val", batch_size=32)
        assert metrics["n_samples"] == 0
        assert "error" in metrics

    def test_returns_required_metric_keys(self):
        from src.model.evaluate import evaluate_on_dataset

        df = _make_test_df()
        # Perfect predictions
        preds = df["sentiment"].tolist()
        mock_model = _make_mock_model(preds)

        metrics = evaluate_on_dataset(mock_model, df, split="test", batch_size=32)

        required_keys = {
            "split",
            "n_samples",
            "accuracy",
            "f1_macro",
            "f1_per_class",
            "precision_macro",
            "recall_macro",
            "mean_confidence",
            "classification_report",
            "confusion_matrix",
        }
        assert required_keys.issubset(metrics.keys())

    def test_perfect_predictions_accuracy_is_one(self):
        from src.model.evaluate import evaluate_on_dataset

        df = _make_test_df()
        preds = df["sentiment"].tolist()
        mock_model = _make_mock_model(preds)

        metrics = evaluate_on_dataset(mock_model, df, split="test", batch_size=32)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1_macro"] == pytest.approx(1.0)

    def test_n_samples_matches_split(self):
        from src.model.evaluate import evaluate_on_dataset

        df = _make_test_df()
        preds = df["sentiment"].tolist()
        mock_model = _make_mock_model(preds)

        metrics = evaluate_on_dataset(mock_model, df, split="test", batch_size=32)
        assert metrics["n_samples"] == 6

    def test_f1_per_class_has_three_values(self):
        from src.model.evaluate import evaluate_on_dataset

        df = _make_test_df()
        preds = df["sentiment"].tolist()
        mock_model = _make_mock_model(preds)

        metrics = evaluate_on_dataset(mock_model, df, split="test", batch_size=32)
        assert len(metrics["f1_per_class"]) == 3

    def test_confusion_matrix_is_3x3(self):
        from src.model.evaluate import evaluate_on_dataset

        df = _make_test_df()
        preds = df["sentiment"].tolist()
        mock_model = _make_mock_model(preds)

        metrics = evaluate_on_dataset(mock_model, df, split="test", batch_size=32)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_batching_works_with_small_batch_size(self):
        """Batch size smaller than dataset should still work."""
        from src.model.evaluate import evaluate_on_dataset

        df = _make_test_df()
        preds = df["sentiment"].tolist()
        mock_model = _make_mock_model(preds[:2])  # batch of 2
        # Need mock_model.predict_batch to handle multiple calls
        mock_model.predict_batch.side_effect = [
            [
                PredictionResult(
                    sentiment=s, confidence=0.9, aspects=[], sarcasm_flag=False
                )
                for s in preds[i : i + 2]
            ]
            for i in range(0, len(preds), 2)
        ]

        metrics = evaluate_on_dataset(mock_model, df, split="test", batch_size=2)
        assert metrics["n_samples"] == 6
        assert metrics["accuracy"] == pytest.approx(1.0)


# ── log_to_mlflow ─────────────────────────────────────────────

class TestLogToMlflow:
    @patch("src.model.evaluate.mlflow")
    def test_calls_mlflow_log_params_and_metrics(self, mock_mlflow):
        from src.model.evaluate import log_to_mlflow
        from src.model.config import ModelConfig

        config = ModelConfig()
        metrics = {
            "accuracy": 0.8,
            "f1_macro": 0.75,
            "f1_per_class": [0.7, 0.8, 0.75],
            "precision_macro": 0.76,
            "recall_macro": 0.74,
            "mean_confidence": 0.85,
            "n_samples": 100,
            "device": "cpu",
            "confusion_matrix": [[30, 5, 5], [3, 25, 2], [2, 3, 25]],
            "classification_report": "dummy report",
        }
        params_yaml = {
            "mlflow": {
                "tracking_uri": "http://localhost:5000",
                "model_experiment_name": "test_exp",
            }
        }

        log_to_mlflow(config, metrics, params_yaml)

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.set_experiment.assert_called_once_with("test_exp")
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.log_metrics.assert_called_once()

    @patch("src.model.evaluate.mlflow")
    def test_uses_default_experiment_name(self, mock_mlflow):
        from src.model.evaluate import log_to_mlflow
        from src.model.config import ModelConfig

        config = ModelConfig()
        metrics = {
            "accuracy": 0.8,
            "f1_macro": 0.75,
            "f1_per_class": [0.7, 0.8, 0.75],
            "precision_macro": 0.76,
            "recall_macro": 0.74,
            "mean_confidence": 0.85,
            "n_samples": 100,
            "device": "cpu",
            "confusion_matrix": [[30, 5, 5], [3, 25, 2], [2, 3, 25]],
            "classification_report": "dummy report",
        }
        params_yaml = {}  # No mlflow config

        log_to_mlflow(config, metrics, params_yaml)

        mock_mlflow.set_experiment.assert_called_once_with("sentiment_baseline")
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_evaluate.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.model.evaluate'`

- [ ] **Step 4: Implement `evaluate.py`**

Create `src/model/evaluate.py`:

```python
"""CLI script: evaluate baseline model on processed data, log results to MLflow.

Usage: python -m src.model.evaluate
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

from contracts.model_interface import ModelInference
from src.model.config import ModelConfig
from src.data.utils import load_params

logger = logging.getLogger("model_evaluate")
LABELS = ["positive", "negative", "neutral"]


def evaluate_on_dataset(
    model: ModelInference,
    sentences_df: pd.DataFrame,
    split: str = "test",
    batch_size: int = 32,
) -> dict:
    """Evaluate model on one split, return metrics dict."""
    df = sentences_df[sentences_df["split"] == split].copy()
    texts = df["text"].tolist()
    true_labels = df["sentiment"].tolist()

    if not texts:
        return {"split": split, "n_samples": 0, "error": f"No samples found for split '{split}'"}

    # Batch predict
    pred_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        pred_results.extend(model.predict_batch(batch))

    pred_labels = [r.sentiment for r in pred_results]
    pred_confidences = [r.confidence for r in pred_results]

    return {
        "split": split,
        "n_samples": len(texts),
        "accuracy": accuracy_score(true_labels, pred_labels),
        "f1_macro": f1_score(
            true_labels, pred_labels, labels=LABELS, average="macro"
        ),
        "f1_per_class": f1_score(
            true_labels, pred_labels, labels=LABELS, average=None
        ).tolist(),
        "precision_macro": precision_score(
            true_labels, pred_labels, labels=LABELS, average="macro"
        ),
        "recall_macro": recall_score(
            true_labels, pred_labels, labels=LABELS, average="macro"
        ),
        "mean_confidence": sum(pred_confidences) / len(pred_confidences),
        "classification_report": classification_report(
            true_labels, pred_labels, labels=LABELS
        ),
        "confusion_matrix": confusion_matrix(
            true_labels, pred_labels, labels=LABELS
        ).tolist(),
    }


def log_to_mlflow(config: ModelConfig, metrics: dict, params_yaml: dict) -> None:
    """Log evaluation results to MLflow."""
    tracking_uri = params_yaml.get("mlflow", {}).get(
        "tracking_uri", "http://localhost:5000"
    )
    experiment_name = params_yaml.get("mlflow", {}).get(
        "model_experiment_name", "sentiment_baseline"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="baseline_roberta"):
        # Params
        mlflow.log_params(
            {
                "model_name": config.model_name,
                "model_type": "baseline_pretrained",
                "max_length": config.max_length,
                "device": str(metrics.get("device", "cpu")),
                "fine_tuned": False,
                "absa_enabled": False,
            }
        )

        # Metrics
        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "mean_confidence": metrics["mean_confidence"],
                "n_samples": metrics["n_samples"],
            }
        )

        # Per-class F1
        for i, label in enumerate(LABELS):
            mlflow.log_metric(f"f1_{label}", metrics["f1_per_class"][i])

        # Artifact: confusion matrix plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(
            confusion_matrix=np.array(metrics["confusion_matrix"]),
            display_labels=LABELS,
        ).plot(ax=ax)
        plt.title("Baseline Model — Confusion Matrix")

        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ) as cm_file:
            cm_path = cm_file.name
            fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(cm_path)

        # Artifact: classification report
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w"
        ) as report_file:
            report_path = report_file.name
            report_file.write(metrics["classification_report"])
        mlflow.log_artifact(report_path)

        logger.info(
            f"MLflow run logged: F1={metrics['f1_macro']:.4f}, "
            f"Acc={metrics['accuracy']:.4f}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Validate processed data existence
    if not Path("data/processed/sentences.csv").exists():
        logger.error("Processed data not found. Run 'dvc repro' first.")
        sys.exit(1)

    params = load_params("params.yaml")
    config = ModelConfig()
    model = BaselineModelInference(config)

    sentences_df = pd.read_csv("data/processed/sentences.csv")
    metrics = evaluate_on_dataset(model, sentences_df, split="test")

    if metrics.get("n_samples", 0) > 0:
        metrics["device"] = str(model._device)
        log_to_mlflow(config, metrics, params)

        # Save metrics for DVC
        Path("data/reports").mkdir(parents=True, exist_ok=True)
        with open("data/reports/baseline_metrics.json", "w") as f:
            json.dump(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
                f, indent=2
            )

        print(f"\n{'='*50}")
        print("Baseline Evaluation Results (test split)")
        print(f"{'='*50}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"F1 Macro:  {metrics['f1_macro']:.4f}")
        print(f"Precision: {metrics['precision_macro']:.4f}")
        print(f"Recall:    {metrics['recall_macro']:.4f}")
        print(f"\n{metrics['classification_report']}")
    else:
        logger.error(f"Evaluation failed: {metrics.get('error')}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_evaluate.py -v
```

Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add src/model/evaluate.py tests/model/test_evaluate.py params.yaml
git commit -m "feat(model): add evaluation functions with MLflow logging"
```

---

### Task 6: DVC Pipeline Stage

**Files:**

- Modify: `dvc.yaml`

- [ ] **Step 1: Add `evaluate_baseline` stage to `dvc.yaml`**

Append this stage at the end of `dvc.yaml` (after the `validate` stage, which ends at line 33):

```yaml
evaluate_baseline:
  cmd: python -m src.model.evaluate
  deps:
    - src/model/
    - data/processed/sentences.csv
  params:
    - params.yaml:
        - mlflow
  metrics:
    - data/reports/baseline_metrics.json:
        cache: false
```

- [ ] **Step 2: Verify DVC recognizes the new stage**

Run:

```bash
dvc dag
```

Expected: Output shows `evaluate_baseline` stage depending on `data/processed/sentences.csv` (which depends on upstream `preprocess` stage).

- [ ] **Step 3: Commit**

```bash
git add dvc.yaml
git commit -m "chore: add evaluate_baseline DVC stage"
```

---

### Task 7: Full Test Suite & Coverage

**Files:**

- All test files in `tests/model/`

- [ ] **Step 1: Run full model test suite with coverage**

Run:

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/ --cov=src/model --cov-report=term-missing -v
```

Expected: All tests pass, ≥ 80% line coverage for `src/model/`.

> Note: `evaluate.py`'s `if __name__ == "__main__"` block won't be covered by unit tests — this is expected.

- [ ] **Step 2: Run entire project test suite**

Run:

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/ -v
```

Expected: All existing tests still pass — no regressions.

- [ ] **Step 3: Commit any remaining changes**

```bash
git add tests/model/ src/model/
git commit -m "test(model): verify full model test suite passes"
```

---

### Task 8: Integration Smoke Test (Manual — Requires Model Download)

> This task downloads the real model (~500MB). Run it once to verify end-to-end correctness.

**Files:**

- None (manual verification)

- [ ] **Step 1: Start MLflow tracking server**

Run (in a separate terminal):

```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

- [ ] **Step 2: Run the evaluation CLI**

Run:

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m src.model.evaluate
```

Expected output:

```
==================================================
Baseline Evaluation Results (test split)
==================================================
Accuracy:  0.XXXX
F1 Macro:  0.XXXX
Precision: 0.XXXX
Recall:    0.XXXX

              precision    recall  f1-score   support
...
```

F1 Macro should be in the **0.55–0.70** range (zero-shot on restaurant domain).

- [ ] **Step 3: Verify MLflow UI**

Open `http://localhost:5000` in browser. Check:

- Experiment: `sentiment_baseline`
- Run: `baseline_roberta`
- Params logged: model_name, model_type, device, max_length, fine_tuned, absa_enabled
- Metrics logged: accuracy, f1_macro, precision_macro, recall_macro, mean_confidence, n_samples, f1_positive, f1_negative, f1_neutral
- Artifacts: confusion_matrix PNG, classification_report TXT

- [ ] **Step 4: Commit DVC lock file if evaluation stage was run via DVC**

```bash
git add dvc.lock
git commit -m "chore: update dvc.lock after baseline evaluation"
```

---

## Verification Plan

### Automated Tests

Run all unit tests (no model download required):

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/ --cov=src/model --cov-report=term-missing -v
```

Expected:

- All tests pass
- ≥ 80% line coverage for `src/model/`
- No regressions in existing `tests/data/` or `tests/contracts/`

Full project regression:

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/ -v
```

### Manual Verification

1. Run `python -m src.model.evaluate` — verify it prints metrics and logs to MLflow
2. Check MLflow UI at `http://localhost:5000` — verify `sentiment_baseline` experiment exists with expected params, metrics, and artifacts
3. Verify DVC pipeline: `dvc dag` shows `evaluate_baseline` stage
