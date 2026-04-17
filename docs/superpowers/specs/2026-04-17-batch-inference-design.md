# Batch Inference Logic — Design Specification

**Date**: 2026-04-17
**Task**: Task 2 — `predict_batch` chunked batch processing
**Status**: Draft

---

## Goal

Enhance `BaselineModelInference.predict_batch()` with chunked batch processing,
memory-safe padding/truncation, and configurable ABSA extraction. The method must
achieve ≥ 100 samples/sec throughput for sentiment-only inference on CPU and
provide a `skip_absa` flag for flexible usage in CI/CD and local testing.

## Scope

- **In scope**: `predict_batch()` method, `ModelConfig`, `ModelInference` interface,
  `MockModelInference`, unit tests, throughput benchmark test.
- **Out of scope**: `/batch_predict` API endpoint, background job (Celery),
  train pipeline, model weights.

## Context

### Current State

`predict_batch` exists in `baseline.py` (L107-127) but has two problems:

1. **No chunking** — feeds the entire `list[str]` into a single forward pass.
   For large inputs, this will exceed available RAM/VRAM.
2. **Sequential ABSA** — calls `_extract_aspects()` per text (~0.6-1.2s/text).
   Acceptable for ≤ 100 samples but not scalable.

### Target Environment

| Environment | Hardware | Dataset Size | Purpose |
|---|---|---|---|
| CI/CD (GitHub Actions) | 2-core CPU, 7GB RAM | ~100 samples | Regression check, F1 validation |
| Local developer | Mac MPS / CPU | 500-2000 samples | Throughput benchmark, stress test |

### Timing Estimates (100 samples, GitHub Actions)

| Phase | Time |
|---|---|
| Sentiment batch (chunked) | ~2-3s |
| ABSA extraction (sequential) | ~60-120s |
| Total inference | ~1.5-2.5 min |
| Total CI pipeline (cached) | ~2-3 min |

---

## Design

### 1. Method Signature & Config Changes

#### `ModelConfig` (`src/model/config.py`)

Add one field:

```python
batch_size: int = 32  # default chunk size for predict_batch
```

Default of 32 balances throughput and memory for 2-core CPU with 7GB RAM.
*Note: `ModelConfig` is `@dataclass(frozen=True)`, so this acts only as a default. Runtime overrides are handled via `predict_batch` arguments.*

#### `ModelInference` interface (`contracts/model_interface.py`)

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

- `batch_size`: Override chunk size. `None` → use `config.batch_size` (32).
- `skip_absa`: When `True`, return empty `aspects=[]` for all texts. Keyword-only
  (after `*` barrier) to prevent positional misuse.
- Both params have defaults → backward compatible with all existing callers.

#### `MockModelInference` (`contracts/mock_model.py`)

Update signature to match interface. Logic unchanged — mock ignores `batch_size`
and `skip_absa` (always returns random predictions).

### 2. Chunking Logic (`baseline.py`)

```
predict_batch(texts, lang, batch_size=32, skip_absa=False)
│
├── 1. Guard: texts is empty → return []
├── 2. Validate: batch_size > 0, else raise ValueError
├── 3. Check language
├── 4. Resolve batch_size: None → self._config.batch_size
│
├── 5. CHUNK LOOP (sentiment):
│   with torch.no_grad():
│       for i in range(0, len(texts), batch_size):
│           chunk = texts[i : i + batch_size]
│           probs = self._predict_probabilities(chunk, padding=True)
│           all_probs.append(probs)
│       all_probs = torch.cat(all_probs, dim=0)
│
├── 6. RESULT ASSEMBLY:
│   for idx in range(len(texts)):
│       pred_idx = all_probs[idx].argmax()
│       aspects = _extract_aspects(texts[idx]) if not skip_absa else []
│       results.append(PredictionResult(...))
│
└── 7. Return results
```

**Key design decisions:**

- `torch.cat(all_probs)` combines chunk results into a single tensor (N×3 floats,
  negligible memory) for uniform indexing.
- `_predict_probabilities()` is called unchanged — it already supports
  `list[str]` with `padding=True`.
- `_extract_aspects()` is called unchanged — its existing `try/except` handles
  per-text failures gracefully (returns `[]`).
- Peak memory is bounded by `batch_size`, not by total input size.

**Memory profile (batch_size=32, max_length=512):**

| Component | Size |
|---|---|
| Input token IDs (int64) | 32 × 512 × 8B = ~128KB |
| Attention mask | ~128KB |
| Model activations (RoBERTa) | ~50-100MB |
| Output probs (accumulated) | N × 3 × 4B (negligible) |
| **Peak per chunk** | **~100-200MB** |

Safe for 7GB RAM CI runner with comfortable margin.

### 3. Error Handling & Edge Cases

| Case | Behavior |
|---|---|
| `texts = []` | Return `[]` immediately |
| `texts = [""]` | Tokenizer creates `[CLS][SEP]`, model predicts normally |
| `batch_size <= 0` | Raise `ValueError("batch_size must be positive, got {batch_size}")` |
| `batch_size > len(texts)` | Single chunk — handled naturally by `range()` |
| Varied text lengths in chunk | `padding=True` pads to max length within chunk |
| ABSA fails for one text | `_extract_aspects` returns `[]`, other texts unaffected |
| Unsupported language | `UnsupportedLanguageError` raised before any processing |

### 4. Logging

Start of batch:
```python
logger.info(
    "predict_batch: %d texts, batch_size=%d, chunks=%d, skip_absa=%s",
    len(texts), batch_size, -(-len(texts) // batch_size), skip_absa,
)
```

During chunk loop (for large batches):
```python
if (i // batch_size + 1) % 10 == 0:
    logger.debug("predict_batch: processed chunk %d/%d", i // batch_size + 1, total_chunks)
```

### 5. Files Changed

| File | Change |
|---|---|
| `src/model/config.py` | Add `batch_size: int = 32` |
| `contracts/model_interface.py` | Update `predict_batch` signature |
| `contracts/mock_model.py` | Update `predict_batch` signature |
| `src/model/baseline.py` | Rewrite `predict_batch` with chunking |
| `src/model/evaluate.py` | Refactor `evaluate_on_dataset` to delegate chunking to `predict_batch` |
| `tests/model/test_baseline.py` | Add chunking + skip_absa unit tests; fix pre-existing template bug in `_assert_aspect_detection_call` |
| `tests/model/test_batch_throughput.py` | **NEW** — throughput benchmark |

### 6. Files NOT Changed

- `src/main.py` — endpoint logic untouched
- `src/data/` — data pipeline untouched
- `dvc.yaml` / `params.yaml` — pipeline config untouched
- Model weights — no retraining

---

## Testing Strategy

### Unit Tests (mocked, always run in CI)

**Pre-requisite fix**: Fix `_assert_aspect_detection_call` in `tests/model/test_baseline.py` to match `config.absa_aspect_template` (`"The text contains a discussion about {}."`) instead of the hardcoded `"This review is about {}."`

Added to `TestPredictBatch` in `tests/model/test_baseline.py`:

| Test | Validates |
|---|---|
| `test_chunking_processes_all_texts` | 100 texts → 100 results |
| `test_chunking_calls_model_per_chunk` | batch_size=32, 100 texts → 4 forward calls |
| `test_custom_batch_size_override` | batch_size=16 → 7 forward calls |
| `test_default_batch_size_from_config` | None → uses config.batch_size |
| `test_skip_absa_returns_empty_aspects` | skip_absa=True → all aspects=[] |
| `test_skip_absa_does_not_call_extract` | skip_absa=True → _extract_aspects never called |
| `test_invalid_batch_size_raises` | batch_size=0 → ValueError |
| `test_batch_size_larger_than_input` | 5 texts, batch_size=32 → 5 results |
| `test_result_order_matches_input` | Result[i] corresponds to texts[i] |

### Throughput Benchmark (real model, local only)

New file `tests/model/test_batch_throughput.py`:

| Test | Target |
|---|---|
| `test_sentiment_throughput_meets_target` | skip_absa=True, 1000 texts → ≥ 100 samples/sec (matches local dev environment scale) |
| `test_full_pipeline_throughput` | skip_absa=False, 50 texts → log throughput (no hard assert) |

Marked with `@pytest.mark.benchmark`, skipped in CI by default:

```ini
# pytest.ini or pyproject.toml
[pytest]
markers =
    benchmark: Throughput benchmark tests (deselect with '-m "not benchmark"')
```

CI runs: `pytest -m "not benchmark"`
Local runs: `pytest -m benchmark`

---

## Constraints

1. **No training pipeline changes** — model integrity preserved.
2. **Backward compatible** — all new parameters have defaults.
3. **Memory bounded** — peak RAM depends on batch_size, not input size.
4. **Result ordering** — output[i] always corresponds to input[i].
