# Zero-Shot ABSA Draft Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `BaselineModelInference` with a DeBERTa zero-shot ABSA pipeline so `predict_single` and `predict_batch` return real `AspectSentiment` objects instead of empty lists.

**Architecture:** `ModelConfig` gains three ABSA fields (model name, threshold, categories). A lazy-loaded `_absa_pipeline` is added to `BaselineModelInference` with a `_get_absa_pipeline()` initializer (mirrors the existing SHAP pipeline pattern) and a `_extract_aspects()` method that runs two sequential zero-shot calls—first for aspect detection, then per-aspect sentiment. Fallback to `[]` on any error keeps the interface backward-compatible.

**Tech Stack:** `transformers` (already installed, `pipeline("zero-shot-classification")`), `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (downloaded via HuggingFace cache at first call), `pytest` + `unittest.mock`.

---

## File Map

| Action | File                           | Responsibility                                                                                                |
| ------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| Modify | `src/model/config.py`          | Add `absa_model_name`, `absa_threshold`, `absa_categories` fields                                             |
| Modify | `src/model/baseline.py`        | Add `_absa_pipeline`, `_get_absa_pipeline()`, `_extract_aspects()`, update `predict_single` + `predict_batch` |
| Modify | `tests/model/test_config.py`   | Add 3 tests for new ABSA config fields                                                                        |
| Modify | `tests/model/test_baseline.py` | Replace `test_aspects_empty_for_baseline`, add 5 ABSA tests, extend batch test                                |

---

## Task 1: Extend `ModelConfig` with ABSA Fields

**Files:**

- Modify: `src/model/config.py`
- Test: `tests/model/test_config.py`

- [ ] **Step 1: Write the failing tests for new ABSA config fields**

Add at the bottom of `tests/model/test_config.py`:

```python
class TestModelConfigABSA:
    def test_default_absa_model_name(self):
        config = ModelConfig()
        assert config.absa_model_name == "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

    def test_default_absa_threshold(self):
        config = ModelConfig()
        assert config.absa_threshold == 0.5

    def test_default_absa_categories(self):
        config = ModelConfig()
        assert set(config.absa_categories) == {
            "food", "service", "ambiance", "price", "location", "general"
        }
        assert len(config.absa_categories) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python \
  -m pytest tests/model/test_config.py::TestModelConfigABSA -v
```

Expected: FAIL — `AttributeError: 'ModelConfig' object has no attribute 'absa_model_name'`

- [ ] **Step 3: Add ABSA fields to `ModelConfig`**

Replace the entire contents of `src/model/config.py` with:

```python
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
        absa_model_name: HuggingFace model identifier for Zero-Shot Classification.
        absa_threshold: Confidence threshold for aspect extraction (aspects with
            score <= threshold are discarded).
        absa_categories: Tuple of aspect categories to detect (must match SemEval schema).
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

    # ABSA Zero-Shot config
    absa_model_name: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    absa_threshold: float = 0.5
    absa_categories: tuple[str, ...] = (
        "food", "service", "ambiance", "price", "location", "general"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python \
  -m pytest tests/model/test_config.py -v
```

Expected: All tests PASS (existing 7 + new 3 = 10 total).

- [ ] **Step 5: Commit**

```bash
git add src/model/config.py tests/model/test_config.py
git commit -m "feat(config): add ABSA zero-shot fields to ModelConfig"
```

---

## Task 2: Add Mocking Infrastructure for DeBERTa Pipeline

**Files:**

- Modify: `tests/model/test_baseline.py`

The existing `_build_model_with_mocks()` only patches `AutoTokenizer` and `AutoModelForSequenceClassification`. We now need a `fake_pipeline_factory` that routes `hf_pipeline` calls by task name.

- [ ] **Step 1: Add `FakeZeroShotPipeline` class and updated `_build_model_with_mocks_absa()` helper**

Add the following helpers **after** the existing `_build_model_with_mocks` function in `tests/model/test_baseline.py` (around line 68, before the `# ── Interface Compliance` comment):

```python
class FakeZeroShotPipeline:
    """Callable that simulates a zero-shot-classification HuggingFace pipeline.

    Configure `aspect_result` and `sentiment_result` before calling, or pass
    them to the constructor. Each call returns whichever result matches how the
    pipeline would be used (aspect extraction vs per-aspect sentiment).

    The real pipeline always returns a dict with 'labels' and 'scores'.
    """

    def __init__(self, aspect_result=None, sentiment_result=None):
        # aspect_result used when multi_label=True (aspect detection step)
        self.aspect_result = aspect_result or {
            "labels": ["food", "service", "ambiance", "price", "location", "general"],
            "scores": [0.85, 0.78, 0.12, 0.08, 0.05, 0.10],
        }
        # sentiment_result used when multi_label is absent (per-aspect step)
        self.sentiment_result = sentiment_result or {
            "labels": ["positive", "negative", "neutral"],
            "scores": [0.82, 0.12, 0.06],
        }
        self._call_count = 0

    def __call__(self, text, candidate_labels, hypothesis_template="", multi_label=False):
        self._call_count += 1
        if multi_label:
            return self.aspect_result
        return self.sentiment_result


def _build_model_with_mocks_absa(
    config=None,
    device=None,
    fake_zero_shot: FakeZeroShotPipeline | None = None,
):
    """Patch HuggingFace loaders AND hf_pipeline and build a BaselineModelInference.

    Args:
        fake_zero_shot: Optional pre-configured FakeZeroShotPipeline. If None,
            a default instance (food 0.85, service 0.78) is used.
    """
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

    mock_hf_model.return_value = FakeOutput(logits=_make_mock_logits(1))

    def fake_pipeline_factory(task, **kwargs):
        if task == "sentiment-analysis":
            return MagicMock()  # SHAP pipeline — not under test here
        if task == "zero-shot-classification":
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
```

- [ ] **Step 2: Verify the file is syntactically valid (no test run yet, just import check)**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python \
  -c "import tests.model.test_baseline"
```

Expected: No output (clean import).

- [ ] **Step 3: Commit**

```bash
git add tests/model/test_baseline.py
git commit -m "test(baseline): add FakeZeroShotPipeline and _build_model_with_mocks_absa helper"
```

---

## Task 3: Add Core ABSA Tests (Red Phase)

**Files:**

- Modify: `tests/model/test_baseline.py`

- [ ] **Step 1: Replace `test_aspects_empty_for_baseline` and add ABSA test class**

In `tests/model/test_baseline.py`, replace the existing test:

```python
    def test_aspects_empty_for_baseline(self):
        model, _, _ = _build_model_with_mocks()
        result = model.predict_single("The food was great")
        assert result.aspects == []
```

with:

```python
    def test_aspects_returned_from_absa(self):
        """After ABSA integration, predict_single returns at least one aspect."""
        model, _ = _build_model_with_mocks_absa()
        with patch("src.model.baseline.hf_pipeline") as mock_pipe:
            mock_pipe.side_effect = lambda task, **kw: FakeZeroShotPipeline()
            result = model.predict_single("The food was amazing but service was terrible")
        assert len(result.aspects) > 0
        for aspect in result.aspects:
            assert aspect.aspect in ("food", "service", "ambiance", "price", "location", "general")
            assert aspect.sentiment in ("positive", "negative", "neutral")
            assert 0.0 <= aspect.confidence <= 1.0
```

Then add a new test class at the end of `tests/model/test_baseline.py`:

```python
# ── ABSA Zero-Shot ─────────────────────────────────────────────

class TestABSA:
    def test_absa_pipeline_not_loaded_before_predict(self):
        """_absa_pipeline must be None until the first predict call."""
        with patch("src.model.baseline.AutoTokenizer") as MockTok, \
             patch("src.model.baseline.AutoModelForSequenceClassification") as MockModel:
            mock_tok = MagicMock()
            mock_tok.return_value = _MockBatchEncoding(
                {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
            )
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model.hf_device_map = None

            @dataclass
            class FakeOutput:
                logits: torch.Tensor

            mock_model.return_value = FakeOutput(logits=_make_mock_logits(1))
            MockTok.from_pretrained.return_value = mock_tok
            MockModel.from_pretrained.return_value = mock_model

            from src.model.baseline import BaselineModelInference
            model = BaselineModelInference(config=ModelConfig(), device=torch.device("cpu"))

        assert model._absa_pipeline is None

    def test_absa_fallback_on_pipeline_error(self):
        """If _absa_pipeline raises, _extract_aspects returns [] without crashing."""
        model, _ = _build_model_with_mocks_absa()

        with patch("src.model.baseline.hf_pipeline", side_effect=RuntimeError("network error")):
            # Force _absa_pipeline to be None so it re-initializes
            model._absa_pipeline = None
            result = model.predict_single("The food was great")

        assert result.aspects == []

    def test_absa_threshold_filters_low_scores(self):
        """Aspects with all scores <= threshold should return empty aspects list."""
        low_score_pipeline = FakeZeroShotPipeline(
            aspect_result={
                "labels": ["food", "service", "ambiance", "price", "location", "general"],
                "scores": [0.4, 0.3, 0.2, 0.1, 0.1, 0.1],  # all below default 0.5
            }
        )
        model, fake_pipe = _build_model_with_mocks_absa(fake_zero_shot=low_score_pipeline)

        with patch("src.model.baseline.hf_pipeline", side_effect=lambda task, **kw: low_score_pipeline):
            model._absa_pipeline = None
            result = model.predict_single("Nothing to detect here")

        assert result.aspects == []

    def test_absa_per_aspect_sentiment_assigned_correctly(self):
        """Each detected aspect gets its own sentiment from the second zero-shot call."""
        food_positive_service_negative = FakeZeroShotPipeline(
            aspect_result={
                "labels": ["food", "service", "ambiance", "price", "location", "general"],
                "scores": [0.85, 0.78, 0.12, 0.08, 0.05, 0.10],
            },
            sentiment_result={
                "labels": ["negative", "positive", "neutral"],
                "scores": [0.71, 0.22, 0.07],
            },
        )
        model, fake_pipe = _build_model_with_mocks_absa(
            fake_zero_shot=food_positive_service_negative
        )

        with patch("src.model.baseline.hf_pipeline",
                   side_effect=lambda task, **kw: food_positive_service_negative):
            model._absa_pipeline = None
            result = model.predict_single("The food was amazing but service was terrible")

        # 2 aspects detected (food 0.85, service 0.78 > 0.5)
        assert len(result.aspects) == 2
        detected_names = {a.aspect for a in result.aspects}
        assert "food" in detected_names
        assert "service" in detected_names
        # All sentiments come from the mock sentiment_result
        for aspect in result.aspects:
            assert aspect.sentiment == "negative"   # labels[0] is "negative" (0.71)
            assert aspect.confidence == 0.71

    def test_predict_batch_includes_aspects(self):
        """predict_batch must call _extract_aspects for each text."""
        model, fake_pipe = _build_model_with_mocks_absa()

        @dataclass
        class FakeOutput:
            logits: torch.Tensor

        with patch("src.model.baseline.hf_pipeline",
                   side_effect=lambda task, **kw: fake_pipe):
            model._absa_pipeline = None
            # Batch size 2 — need to configure mock_hf_model for 2 rows
            model._model.return_value = FakeOutput(logits=_make_mock_logits(2))
            results = model.predict_batch(["The food was great", "Service was slow"])

        assert len(results) == 2
        for result in results:
            assert isinstance(result.aspects, list)
            assert len(result.aspects) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python \
  -m pytest tests/model/test_baseline.py::TestABSA -v
```

Expected: All 5 ABSA tests FAIL — `AttributeError: 'BaselineModelInference' object has no attribute '_absa_pipeline'`.

Also verify the replaced test now fails:

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python \
  -m pytest tests/model/test_baseline.py::TestPredictSingle::test_aspects_returned_from_absa -v
```

Expected: FAIL — `assert [] == []` becomes false assertion or `AttributeError`.

- [ ] **Step 3: Commit (red tests)**

```bash
git add tests/model/test_baseline.py
git commit -m "test(baseline): add ABSA red-phase tests"
```

---

## Task 4: Implement ABSA in `baseline.py`

**Files:**

- Modify: `src/model/baseline.py`

- [ ] **Step 1: Add `AspectSentiment` to the import and add `logging`**

At the top of `src/model/baseline.py`, make these two changes:

Change line 3-12 from:

```python
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline as hf_pipeline

from contracts.errors import ModelError, UnsupportedLanguageError
from contracts.model_interface import ModelInference, PredictionResult, SHAPResult
from src.model.config import ModelConfig
from src.model.device import get_device
```

To:

```python
from __future__ import annotations

import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline as hf_pipeline

from contracts.errors import ModelError, UnsupportedLanguageError
from contracts.model_interface import AspectSentiment, ModelInference, PredictionResult, SHAPResult
from src.model.config import ModelConfig
from src.model.device import get_device

logger = logging.getLogger(__name__)
```

- [ ] **Step 2: Add `_absa_pipeline = None` to `__init__`**

In `src/model/baseline.py`, update `__init__` to initialise the lazy attribute. Change:

```python
        self._hf_pipeline = None
        self._load_model()
```

To:

```python
        self._hf_pipeline = None
        self._absa_pipeline = None  # lazy-loaded on first ABSA call
        self._load_model()
```

- [ ] **Step 3: Add `_get_absa_pipeline()` method**

After the existing `_get_classification_pipeline` method (currently ending at line 135), add:

```python
    def _get_absa_pipeline(self):
        """Lazy-init a zero-shot-classification pipeline for ABSA."""
        if self._absa_pipeline is None:
            pipeline_device = (
                self._device.index
                if self._device.type == "cuda" and self._device.index is not None
                else (-1 if self._device.type == "cpu" else self._device)
            )
            try:
                self._absa_pipeline = hf_pipeline(
                    "zero-shot-classification",
                    model=self._config.absa_model_name,
                    device=pipeline_device,
                )
            except Exception as exc:
                raise ModelError(f"Failed to load ABSA model: {exc}") from exc
        return self._absa_pipeline
```

- [ ] **Step 4: Add `_extract_aspects()` method**

Directly after `_get_absa_pipeline`, add:

```python
    def _extract_aspects(self, text: str) -> list[AspectSentiment]:
        """Extract aspects and their sentiments using two Zero-Shot calls.

        Step 1 — aspect detection: classify `text` against the 6 configured
        categories with multi_label=True. Keep only those above threshold.
        Step 2 — per-aspect sentiment: for each detected aspect, run a second
        zero-shot call with labels ["positive", "negative", "neutral"].

        Returns an empty list on any exception (safe fallback).
        """
        try:
            absa_pipe = self._get_absa_pipeline()

            # Step 1: Detect which aspects are mentioned
            aspect_result = absa_pipe(
                text,
                candidate_labels=list(self._config.absa_categories),
                hypothesis_template="This review is about {}.",
                multi_label=True,
            )

            detected_aspects = [
                (label, score)
                for label, score in zip(aspect_result["labels"], aspect_result["scores"])
                if score > self._config.absa_threshold
            ]

            if not detected_aspects:
                return []

            # Step 2: Classify sentiment for each detected aspect
            aspects: list[AspectSentiment] = []
            for aspect_name, _ in detected_aspects:
                sent_result = absa_pipe(
                    text,
                    candidate_labels=["positive", "negative", "neutral"],
                    hypothesis_template=f"The sentiment about {aspect_name} is {{}}.",
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
            logger.warning("ABSA extraction failed, returning empty aspects", exc_info=True)
            return []
```

- [ ] **Step 5: Update `predict_single` to use ABSA**

Change:

```python
        return PredictionResult(
            sentiment=self._config.label_map[pred_idx],
            confidence=round(probs[pred_idx].item(), 4),
            aspects=[],
            sarcasm_flag=False,
        )
```

To:

```python
        return PredictionResult(
            sentiment=self._config.label_map[pred_idx],
            confidence=round(probs[pred_idx].item(), 4),
            aspects=self._extract_aspects(text),
            sarcasm_flag=False,
        )
```

- [ ] **Step 6: Update `predict_batch` to use ABSA**

Change (in the for-loop inside `predict_batch`):

```python
            results.append(
                PredictionResult(
                    sentiment=self._config.label_map[pred_idx],
                    confidence=round(probs[index][pred_idx].item(), 4),
                    aspects=[],
                    sarcasm_flag=False,
                )
            )
```

To:

```python
            results.append(
                PredictionResult(
                    sentiment=self._config.label_map[pred_idx],
                    confidence=round(probs[index][pred_idx].item(), 4),
                    aspects=self._extract_aspects(texts[index]),
                    sarcasm_flag=False,
                )
            )
```

- [ ] **Step 7: Update the class docstring**

Change the class docstring from:

```python
    """Implement ModelInference interface using pre-trained RoBERTa.

    - Overall sentiment: predicted from model
    - ABSA aspects: returns [] (baseline has no ABSA)
    - SHAP: uses shap.Explainer with HuggingFace pipeline
    """
```

To:

```python
    """Implement ModelInference interface using pre-trained RoBERTa + Zero-Shot ABSA.

    - Overall sentiment: predicted by RoBERTa (twitter-roberta-base-sentiment-latest)
    - ABSA aspects: extracted via DeBERTa zero-shot-classification (two-step pipeline)
    - SHAP: uses shap.Explainer with HuggingFace sentiment-analysis pipeline
    """
```

- [ ] **Step 8: Run the full model test suite**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python \
  -m pytest tests/model/ -v --tb=short
```

Expected: All tests PASS. Pay attention to:

- `TestABSA` — all 5 new tests green
- `TestPredictSingle::test_aspects_returned_from_absa` — green
- `TestPredictBatch::test_all_results_are_prediction_result` — this test currently asserts `r.aspects == []`. We need to update it (see note below).

**Fix the existing batch test:** In `test_all_results_are_prediction_result`, change:

```python
        for r in results:
            assert isinstance(r, PredictionResult)
            assert r.aspects == []          # <-- remove this line
            assert r.sarcasm_flag is False
```

To:

```python
        for r in results:
            assert isinstance(r, PredictionResult)
            assert isinstance(r.aspects, list)  # aspects is now populated by ABSA
            assert r.sarcasm_flag is False
```

Re-run after the fix:

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python \
  -m pytest tests/model/ -v --tb=short
```

Expected: All tests PASS (zero failures).

- [ ] **Step 9: Commit**

```bash
git add src/model/baseline.py tests/model/test_baseline.py
git commit -m "feat(baseline): integrate DeBERTa zero-shot ABSA pipeline"
```

---

## Task 5: Integration Smoke Test (Real Model Download)

**Files:** None — run only.

> **Note:** This test downloads `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (~700MB) from HuggingFace on first run. Subsequent runs use the local cache at `~/.cache/huggingface/hub/`. Ensure you have internet access and ~1.5GB free disk space.

- [ ] **Step 1: Run the smoke test with the real model**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -c "
from src.model.baseline import BaselineModelInference
model = BaselineModelInference()
result = model.predict_single('The food was amazing but service was slow')
print(f'Sentiment: {result.sentiment} ({result.confidence})')
print(f'Aspects ({len(result.aspects)}):')
for a in result.aspects:
    print(f'  {a.aspect}: {a.sentiment} ({a.confidence})')
assert len(result.aspects) > 0, 'ABSA should return at least 1 aspect'
for a in result.aspects:
    assert a.aspect in ('food','service','ambiance','price','location','general')
    assert a.sentiment in ('positive','negative','neutral')
    assert 0.0 <= a.confidence <= 1.0
print('PASS')
"
```

Expected output (exact values will vary):

```
Sentiment: positive (0.72xx)
Aspects (2):
  food: positive (0.8xxx)
  service: negative (0.7xxx)
PASS
```

- [ ] **Step 2: Commit (smoke test is manual — just commit the plan completion note)**

```bash
git commit --allow-empty -m "chore: smoke test passed for zero-shot ABSA integration"
```

---

## Self-Review Checklist

### Spec Coverage

| Spec Section                                                          | Task                          |
| --------------------------------------------------------------------- | ----------------------------- |
| `ModelConfig` ABSA fields                                             | Task 1                        |
| `_absa_pipeline = None` in `__init__`                                 | Task 4 Step 2                 |
| `_get_absa_pipeline()` lazy init                                      | Task 4 Step 3                 |
| `_extract_aspects()` — aspect detection (Step 2)                      | Task 4 Step 4                 |
| `_extract_aspects()` — per-aspect sentiment (Step 3)                  | Task 4 Step 4                 |
| `predict_single` → `aspects=self._extract_aspects(text)`              | Task 4 Step 5                 |
| `predict_batch` → aspects per text                                    | Task 4 Step 6                 |
| Fallback to `[]` on error                                             | `_extract_aspects` try/except |
| `test_aspects_empty_for_baseline` → `test_aspects_returned_from_absa` | Task 3 Step 1                 |
| `test_absa_pipeline_lazy_loaded`                                      | Task 3 Step 1                 |
| `test_absa_fallback_on_error`                                         | Task 3 Step 1                 |
| `test_absa_threshold_filters_low_scores`                              | Task 3 Step 1                 |
| `test_absa_per_aspect_sentiment`                                      | Task 3 Step 1                 |
| `test_predict_batch_includes_aspects`                                 | Task 3 Step 1                 |
| `test_default_absa_model_name/threshold/categories`                   | Task 1 Step 1                 |
| Integration smoke test                                                | Task 5                        |

No gaps found.

### Type Consistency

- `AspectSentiment` imported from `contracts.model_interface` in both `baseline.py` and `test_baseline.py` — consistent.
- `_extract_aspects(self, text: str) -> list[AspectSentiment]` — type matches usage in `predict_single` and `predict_batch`.
- `FakeZeroShotPipeline` returns `{"labels": [...], "scores": [...]}` — matches what `_extract_aspects` indexes with `result["labels"]` and `result["scores"]`.
- `_get_absa_pipeline()` raises `ModelError` on load failure. `_extract_aspects` catches any `Exception` and falls back to `[]`, so `ModelError` (a subclass of `Exception`) is also caught — correct fallback behavior.
