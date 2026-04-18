# Batch Inference Logic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `BaselineModelInference.predict_batch()` to support chunked processing, a `skip_absa` flag, and a `batch_size` config default — achieving ≥100 samples/sec for sentiment-only inference on CPU.

**Architecture:** The method splits large input lists into fixed-size chunks, runs `_predict_probabilities()` per chunk under `torch.no_grad()`, concatenates chunk tensors, then assembles `PredictionResult` objects with optional ABSA extraction. Peak memory is bounded by `batch_size`, not total input.

**Tech Stack:** Python 3.11, PyTorch, HuggingFace Transformers, pytest, `@pytest.mark.benchmark`

---

## File Map

| File | Change |
|---|---|
| `src/model/config.py` | Add `batch_size: int = 32` field |
| `contracts/model_interface.py` | Update `predict_batch` signature with `batch_size` and `skip_absa` |
| `contracts/mock_model.py` | Update `predict_batch` signature to match interface |
| `src/model/baseline.py` | Rewrite `predict_batch` with chunking logic |
| `src/model/evaluate.py` | Remove manual chunking loop — delegate to `predict_batch` |
| `tests/model/test_baseline.py` | Fix `_assert_aspect_detection_call` bug; add chunking + skip_absa tests |
| `tests/model/test_batch_throughput.py` | **NEW** — throughput benchmark tests |
| `pyproject.toml` | **NEW** — add `[tool.pytest.ini_options]` with `benchmark` marker |

---

## Task 1: Add `batch_size` to `ModelConfig`

**Files:**
- Modify: `src/model/config.py`
- Test: `tests/model/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/model/test_config.py`:

```python
def test_default_batch_size_is_32():
    from src.model.config import ModelConfig
    config = ModelConfig()
    assert config.batch_size == 32

def test_batch_size_is_configurable():
    from src.model.config import ModelConfig
    config = ModelConfig(batch_size=16)
    assert config.batch_size == 16
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/trungshin/.gemini/antigravity/worktrees/Sentiment-Analysis-Service/implement-batch-inference-logic-20260417
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_config.py::test_default_batch_size_is_32 tests/model/test_config.py::test_batch_size_is_configurable -v
```

Expected: `FAILED` — `TypeError: ModelConfig.__init__() got an unexpected keyword argument 'batch_size'`

- [ ] **Step 2.5: Fix pre-existing `absa_threshold` test bug**

In `tests/model/test_config.py`, replace lines 61-62 to match the actual config default of `0.45`:

```python
    def test_default_absa_threshold(self):
        assert ModelConfig().absa_threshold == 0.45
```

- [ ] **Step 3: Add `batch_size` to `ModelConfig`**

In `src/model/config.py`, add after `absa_sentiment_template` (line 49):

```python
    batch_size: int = 32
```

Also add to the docstring:
```python
        batch_size: Default chunk size for predict_batch. Overridable at call time.
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_config.py::test_default_batch_size_is_32 tests/model/test_config.py::test_batch_size_is_configurable -v
```

Expected: `PASSED PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/model/config.py tests/model/test_config.py
git commit -m "feat(config): add batch_size field to ModelConfig (default 32)"
```

---

## Task 2: Update `ModelInference` Interface

**Files:**
- Modify: `contracts/model_interface.py`

- [ ] **Step 1: Update `predict_batch` abstract method signature**

Replace lines 37–38 in `contracts/model_interface.py`:

```python
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
```

- [ ] **Step 2: Run existing tests to verify interface change doesn't break anything**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/ -v -m "not benchmark"
```

Expected: All previously passing tests still pass. Failures on new signature are expected only if `MockModelInference` hasn't been updated yet — that is Task 3.

- [ ] **Step 3: Commit**

```bash
git add contracts/model_interface.py
git commit -m "feat(interface): update predict_batch signature with batch_size and skip_absa params"
```

---

## Task 3: Update `MockModelInference`

**Files:**
- Modify: `contracts/mock_model.py`

- [ ] **Step 1: Update `predict_batch` signature to match interface**

Replace lines 60–62 in `contracts/mock_model.py`:

```python
    def predict_batch(
        self,
        texts: list[str],
        lang: str = "en",
        *,
        batch_size: int | None = None,
        skip_absa: bool = False,
    ) -> list[PredictionResult]:
        self._check_language(lang)
        return [self._random_prediction() for _ in texts]
```

- [ ] **Step 2: Run tests to verify mock is consistent**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/ -v -m "not benchmark"
```

Expected: All previously passing tests pass.

- [ ] **Step 3: Commit**

```bash
git add contracts/mock_model.py
git commit -m "feat(mock): update MockModelInference.predict_batch to match new interface signature"
```

---

## Task 4: Fix `_assert_aspect_detection_call` Bug in Test Helpers

**Files:**
- Modify: `tests/model/test_baseline.py`

**Context:** Line 55 has a hardcoded template `"This review is about {}."` that doesn't match `ModelConfig().absa_aspect_template` (`"The text contains a discussion about {}."`). This bug makes existing ABSA tests wrong.

- [ ] **Step 1: Fix the hardcoded template on line 55**

Replace lines 51–56 in `tests/model/test_baseline.py`:

```python
def _assert_aspect_detection_call(call, text):
    """Verify the first zero-shot pass extracts ABSA aspects."""
    assert call["text"] == text
    assert call["candidate_labels"] == list(ModelConfig().absa_categories)
    assert call["hypothesis_template"] == ModelConfig().absa_aspect_template
    assert call["multi_label"] is True
```

Also fix `_assert_aspect_sentiment_call` on lines 59–64 — the template `"The sentiment about {aspect_name} is {}."` doesn't match `absa_sentiment_template`. Replace lines 59–64:

```python
def _assert_aspect_sentiment_call(call, text, aspect_name):
    """Verify the per-aspect zero-shot pass scores sentiment labels."""
    assert call["text"] == text
    assert call["candidate_labels"] == ["positive", "negative", "neutral"]
    expected_template = ModelConfig().absa_sentiment_template.format(aspect=aspect_name)
    assert call["hypothesis_template"] == expected_template
    assert call["multi_label"] is False
```

- [ ] **Step 2: Run existing ABSA tests to see what passes/fails after the fix**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_baseline.py::TestABSA -v
```

Expected: Tests now use the correct template. Some may still fail if the `FakeZeroShotPipeline` in the test uses hardcoded strings — those will be fixed in Task 6.

- [ ] **Step 3: Commit the helper fix**

```bash
git add tests/model/test_baseline.py
git commit -m "fix(tests): correct hardcoded ABSA template strings in test helper assertions"
```

---

## Task 5: Rewrite `predict_batch` in `baseline.py`

**Files:**
- Modify: `src/model/baseline.py:107-127`

- [ ] **Step 1: Write failing tests first**

Add to `TestPredictBatch` class in `tests/model/test_baseline.py`:

```python
def test_chunking_processes_all_texts(self):
    """100 texts → 100 results regardless of chunk size."""
    from unittest.mock import MagicMock
    model, mock_tok, mock_hf = _build_model_with_mocks()

    # Each chunk call returns logits for its chunk size
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
    """batch_size=None → uses config.batch_size (32)."""
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
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_baseline.py::TestPredictBatch::test_chunking_processes_all_texts tests/model/test_baseline.py::TestPredictBatch::test_skip_absa_returns_empty_aspects tests/model/test_baseline.py::TestPredictBatch::test_invalid_batch_size_raises -v
```

Expected: `FAILED` — `TypeError: predict_batch() got an unexpected keyword argument 'batch_size'`

- [ ] **Step 3: Rewrite `predict_batch` in `baseline.py`**

Replace lines 107–127 in `src/model/baseline.py`:

```python
    def predict_batch(
        self,
        texts: list[str],
        lang: str = "en",
        *,
        batch_size: int | None = None,
        skip_absa: bool = False,
    ) -> list[PredictionResult]:
        """Predict sentiment for a batch of texts using chunked processing.

        Args:
            texts: Input texts to classify.
            lang: Language code (must be supported).
            batch_size: Number of texts per forward pass. None uses config default.
            skip_absa: When True, skip aspect extraction (aspects=[]).
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
            "predict_batch: %d texts, batch_size=%d, chunks=%d, skip_absa=%s",
            len(texts),
            resolved_batch_size,
            total_chunks,
            skip_absa,
        )

        all_probs: list[torch.Tensor] = []
        for i in range(0, len(texts), resolved_batch_size):
            chunk = texts[i : i + resolved_batch_size]
            probs = self._predict_probabilities(chunk, padding=True)
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
        for idx in range(len(texts)):
            pred_idx = combined_probs[idx].argmax().item()
            aspects = [] if skip_absa else self._extract_aspects(texts[idx])
            results.append(
                PredictionResult(
                    sentiment=self._config.label_map[pred_idx],
                    confidence=round(combined_probs[idx][pred_idx].item(), 4),
                    aspects=aspects,
                    sarcasm_flag=False,
                )
            )

        return results
```

- [ ] **Step 4: Run all `TestPredictBatch` tests**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_baseline.py::TestPredictBatch -v
```

Expected: All tests pass.

- [ ] **Step 5: Run the full test suite to catch regressions**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_baseline.py -v -m "not benchmark"
```

Expected: All previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/model/baseline.py tests/model/test_baseline.py
git commit -m "feat(baseline): rewrite predict_batch with chunking, batch_size, and skip_absa support"
```

---

## Task 6: Refactor `evaluate.py` to Delegate Chunking

**Files:**
- Modify: `src/model/evaluate.py:62-68`
- Modify: `tests/model/test_evaluate.py`

**Context:** `evaluate_on_dataset` already has its own manual chunking loop. Now that `predict_batch` handles chunking internally, this loop can be replaced with a single call, and we must remove `batch_size` passing from both the function and its test calls.

- [ ] **Step 1: Write a test to confirm delegation and remove old test**

In `tests/model/test_evaluate.py`, **delete** `test_batching_works_with_small_batch_size` (lines 129-148), because `evaluate_on_dataset` no longer handles batching.

Then, add this new test to confirm delegation:

```python
def test_evaluate_on_dataset_delegates_chunking_to_predict_batch():
    """evaluate_on_dataset should call predict_batch once with all texts."""
    from unittest.mock import patch, MagicMock
    import pandas as pd
    from src.model.evaluate import evaluate_on_dataset
    from contracts.model_interface import PredictionResult

    texts = [f"text {i}" for i in range(10)]
    df = pd.DataFrame({
        "text": texts,
        "sentiment": ["positive"] * 10,
        "split": ["test"] * 10,
    })

    mock_model = MagicMock()
    mock_model.predict_batch.return_value = [
        PredictionResult(sentiment="positive", confidence=0.9)
        for _ in range(10)
    ]

    evaluate_on_dataset(mock_model, df, split="test")

    mock_model.predict_batch.assert_called_once()
    call_args = mock_model.predict_batch.call_args
    assert call_args[0][0] == texts
```

- [ ] **Step 2: Remove `batch_size=` from all test calls**

In `tests/model/test_evaluate.py`, remove `, batch_size=32`, `, batch_size=2`, and `, batch_size=100` from all calls to `evaluate_on_dataset`. There are 7 remaining calls after deleting the batching test.

- [ ] **Step 3: Run test to confirm it fails**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_evaluate.py::test_evaluate_on_dataset_delegates_chunking_to_predict_batch -v
```

Expected: `FAILED` — `assert mock_model.predict_batch.call_count == 1` fails because the current loop calls it multiple times.

- [ ] **Step 4: Replace the chunking loop in `evaluate_on_dataset`**

Replace lines 62–68 in `src/model/evaluate.py`:

```python
    batch_results = model.predict_batch(texts, skip_absa=True)
    predictions = [result.sentiment for result in batch_results]
    confidences = [float(result.confidence) for result in batch_results]
```

Also remove the now-unused `batch_size: int = 32` parameter from the `evaluate_on_dataset` signature (line 48) and docstring.

- [ ] **Step 5: Run the evaluate test**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_evaluate.py -v
```

Expected: All tests pass including the new delegation test.

- [ ] **Step 6: Commit**

```bash
git add src/model/evaluate.py tests/model/test_evaluate.py
git commit -m "refactor(evaluate): delegate chunking to predict_batch, remove manual loop"
```

---

## Task 7: Add `pytest.ini` Benchmark Marker + New Throughput Test File

**Files:**
- Create: `pyproject.toml`
- Create: `tests/model/test_batch_throughput.py`

- [ ] **Step 1: Create `pyproject.toml` with benchmark marker**

```toml
[tool.pytest.ini_options]
markers = [
    "benchmark: Throughput benchmark tests (deselect with '-m \"not benchmark\"')",
]
```

- [ ] **Step 2: Create `tests/model/test_batch_throughput.py`**

```python
"""Throughput benchmark tests for predict_batch.

Run locally:   pytest tests/model/test_batch_throughput.py -m benchmark -v
Skip in CI:    pytest -m "not benchmark"

These tests load the real model — they are NOT mocked.
Requires: PYTHONPATH set to project root.
"""
import time
import pytest


@pytest.mark.benchmark
def test_sentiment_throughput_meets_target():
    """skip_absa=True on 1000 texts must achieve ≥ 100 samples/sec."""
    from src.model.baseline import BaselineModelInference
    from src.model.config import ModelConfig

    config = ModelConfig(batch_size=32)
    model = BaselineModelInference(config=config)

    texts = ["The food was absolutely amazing and I loved every bite!"] * 1000

    start = time.perf_counter()
    results = model.predict_batch(texts, batch_size=32, skip_absa=True)
    elapsed = time.perf_counter() - start

    throughput = len(texts) / elapsed
    print(f"\nThroughput: {throughput:.1f} samples/sec ({elapsed:.2f}s for {len(texts)} texts)")

    assert len(results) == 1000, "Result count must match input count"
    assert throughput >= 100.0, (
        f"Throughput {throughput:.1f} samples/sec is below the 100 samples/sec target"
    )


@pytest.mark.benchmark
def test_full_pipeline_throughput():
    """skip_absa=False on 50 texts — log throughput, no hard assert."""
    from src.model.baseline import BaselineModelInference
    from src.model.config import ModelConfig

    config = ModelConfig(batch_size=32)
    model = BaselineModelInference(config=config)

    texts = [
        "The food was great but the service was slow.",
        "Absolutely terrible experience. Never coming back.",
        "Best ambiance in the city, fair prices.",
        "Average meal, nothing special.",
        "Outstanding in every way!",
    ] * 10  # 50 texts

    start = time.perf_counter()
    results = model.predict_batch(texts, batch_size=32, skip_absa=False)
    elapsed = time.perf_counter() - start

    throughput = len(texts) / elapsed
    print(f"\nFull pipeline throughput: {throughput:.2f} samples/sec ({elapsed:.1f}s for {len(texts)} texts)")

    assert len(results) == 50, "Result count must match input count"
    # No hard throughput assert — ABSA is intentionally slow (~60-120s for 100 texts in CI)
```

- [ ] **Step 3: Verify benchmark tests are correctly marked (dry run)**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_batch_throughput.py --collect-only
```

Expected output lists both `test_sentiment_throughput_meets_target` and `test_full_pipeline_throughput` with `benchmark` marker.

- [ ] **Step 4: Verify benchmark tests are excluded by default CI filter**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/ -m "not benchmark" --collect-only 2>&1 | grep "test_batch_throughput"
```

Expected: No output — benchmark tests are excluded.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/model/test_batch_throughput.py
git commit -m "feat(tests): add throughput benchmark tests and register benchmark pytest marker"
```

---

## Task 8: Full CI Verification

- [ ] **Step 1: Run the complete non-benchmark test suite**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/ -m "not benchmark" -v
```

Expected: All tests pass. Zero failures, zero errors.

- [ ] **Step 2: Verify no regressions in evaluate or ABSA tests**

```bash
/Users/trungshin/miniconda3/envs/sentiment_analysis_service/bin/python -m pytest tests/model/test_evaluate.py tests/model/test_baseline.py::TestABSA -v
```

Expected: All tests pass.

- [ ] **Step 3: Final commit with summary**

```bash
git add -A
git commit -m "chore: verify all non-benchmark tests pass after batch inference refactor"
```

---

## Self-Review Checklist

**Spec Coverage:**
- [x] `batch_size: int = 32` added to `ModelConfig` → Task 1
- [x] `predict_batch` signature with `batch_size | None` and `skip_absa` → Tasks 2–3
- [x] Chunking loop with `torch.no_grad()` → Task 5
- [x] `skip_absa=True` → `aspects=[]`, no `_extract_aspects` call → Task 5
- [x] `batch_size <= 0` raises `ValueError` → Task 5
- [x] Empty list → `[]` → pre-existing, preserved in Task 5
- [x] Logging at batch start and every 10 chunks → Task 5
- [x] `evaluate.py` delegates chunking to `predict_batch` → Task 6
- [x] `_assert_aspect_detection_call` bug fixed → Task 4
- [x] 9 unit tests from spec → Task 5 (8 new + 1 pre-existing empty list test)
- [x] Throughput benchmark with `@pytest.mark.benchmark` → Task 7
- [x] CI marker filter (`-m "not benchmark"`) → Task 7

**Type Consistency:** All tasks use `list[str]`, `list[PredictionResult]`, `int | None`, `bool` — consistent throughout.
