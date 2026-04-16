# Handoff Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `contracts/` package — errors, data classes, abstract interface, mock model, Pydantic schemas, and sample data — so Quân (Backend) and Long (Frontend) can start coding in parallel today without waiting for the real model.

**Architecture:** All shared types live in `contracts/` at the repo root. The package is dependency-injected into FastAPI via `MockModelInference` (or the real `ModelInference` once trained). Pydantic schemas in `schemas.py` drive FastAPI's auto-generated Swagger UI, which is Long's primary source of truth for frontend types.

**Tech Stack:** Python 3.12, `dataclasses`, `pydantic` v2, `pytest` (unit tests), `pytest-cov` (coverage)

---

## File Map

| File | Responsibility |
|------|---------------|
| `contracts/__init__.py` | Package marker; re-exports public symbols |
| `contracts/errors.py` | `ModelError`, `UnsupportedLanguageError` |
| `contracts/model_interface.py` | `AspectSentiment`, `PredictionResult`, `SHAPResult` dataclasses + `ModelInference` abstract class |
| `contracts/mock_model.py` | `MockModelInference` — random valid outputs, fake latency |
| `contracts/schemas.py` | Pydantic v2 models for all 6 API endpoints |
| `contracts/sample_batch_input.csv` | 25 rows of sample restaurant reviews for batch tests |
| `contracts/sample_responses.json` | One canonical example response per endpoint |
| `contracts/README.md` | Integration guide for Quân and Long |
| `tests/__init__.py` | Test package marker |
| `tests/contracts/__init__.py` | Test sub-package marker |
| `tests/contracts/test_errors.py` | Error hierarchy and inheritance checks |
| `tests/contracts/test_model_interface.py` | Dataclass field presence and type checks |
| `tests/contracts/test_mock_model.py` | MockModelInference behavioral tests |
| `tests/contracts/test_schemas.py` | Pydantic schema validation against sample responses |

---

## Task 1: Project scaffolding + error classes

**Files:**
- Create: `contracts/__init__.py`
- Create: `contracts/errors.py`
- Create: `tests/__init__.py`
- Create: `tests/contracts/__init__.py`
- Create: `tests/contracts/test_errors.py`
- Create: `requirements.txt` (add `pytest`, `pytest-cov`, `pydantic`)

- [ ] **Step 1.1: Write the failing tests**

```python
# tests/contracts/test_errors.py
import pytest
from contracts.errors import ModelError, UnsupportedLanguageError


def test_model_error_is_exception():
    err = ModelError("something broke")
    assert isinstance(err, Exception)
    assert str(err) == "something broke"


def test_unsupported_language_error_is_model_error():
    err = UnsupportedLanguageError("zh")
    assert isinstance(err, ModelError)
    assert isinstance(err, Exception)
    assert "zh" in str(err)


def test_raise_model_error():
    with pytest.raises(ModelError):
        raise ModelError("model load failed")


def test_raise_unsupported_language_error_caught_as_model_error():
    with pytest.raises(ModelError):
        raise UnsupportedLanguageError("fr")
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```bash
cd /Users/trungshin/learning/Sentiment-Analysis-Service
pytest tests/contracts/test_errors.py -v
```

Expected: `ModuleNotFoundError: No module named 'contracts'`

- [ ] **Step 1.3: Create scaffolding files**

```bash
# Create directory structure
mkdir -p contracts tests/contracts
touch contracts/__init__.py tests/__init__.py tests/contracts/__init__.py
```

- [ ] **Step 1.4: Add dependencies to requirements.txt**

```
# requirements.txt
pydantic>=2.0,<3.0
pytest>=8.0
pytest-cov>=5.0
```

Install them:

```bash
pip install -r requirements.txt
```

- [ ] **Step 1.5: Implement `contracts/errors.py`**

```python
# contracts/errors.py
"""
Custom exceptions for the ModelInference layer.

HTTP mapping (enforced in FastAPI):
    ModelError              → HTTP 500 Internal Server Error
    UnsupportedLanguageError → HTTP 400 Bad Request
"""


class ModelError(Exception):
    """Base exception for all model-level failures (load, predict, etc.)."""


class UnsupportedLanguageError(ModelError):
    """Raised when the model is asked to handle a language it does not support."""

    def __init__(self, lang: str) -> None:
        super().__init__(f"Language not supported: '{lang}'")
        self.lang = lang
```

- [ ] **Step 1.6: Run tests to confirm they pass**

```bash
pytest tests/contracts/test_errors.py -v
```

Expected output:
```
tests/contracts/test_errors.py::test_model_error_is_exception PASSED
tests/contracts/test_errors.py::test_unsupported_language_error_is_model_error PASSED
tests/contracts/test_errors.py::test_raise_model_error PASSED
tests/contracts/test_errors.py::test_raise_unsupported_language_error_caught_as_model_error PASSED
4 passed
```

- [ ] **Step 1.7: Commit**

```bash
git add contracts/__init__.py contracts/errors.py \
        tests/__init__.py tests/contracts/__init__.py \
        tests/contracts/test_errors.py requirements.txt
git commit -m "feat(contracts): scaffold package and add error hierarchy"
```

---

## Task 2: Data classes + abstract ModelInference interface

**Files:**
- Create: `contracts/model_interface.py`
- Create: `tests/contracts/test_model_interface.py`

- [ ] **Step 2.1: Write the failing tests**

```python
# tests/contracts/test_model_interface.py
import dataclasses
from contracts.model_interface import (
    AspectSentiment,
    PredictionResult,
    SHAPResult,
    ModelInference,
)


# --- AspectSentiment ---

def test_aspect_sentiment_fields():
    asp = AspectSentiment(aspect="food", sentiment="positive", confidence=0.95)
    assert asp.aspect == "food"
    assert asp.sentiment == "positive"
    assert asp.confidence == 0.95


def test_aspect_sentiment_is_dataclass():
    assert dataclasses.is_dataclass(AspectSentiment)


# --- PredictionResult ---

def test_prediction_result_fields():
    result = PredictionResult(
        sentiment="negative",
        confidence=0.88,
        aspects=[AspectSentiment("service", "negative", 0.80)],
        sarcasm_flag=False,
    )
    assert result.sentiment == "negative"
    assert result.confidence == 0.88
    assert len(result.aspects) == 1
    assert result.sarcasm_flag is False


def test_prediction_result_empty_aspects():
    result = PredictionResult(
        sentiment="neutral",
        confidence=0.60,
        aspects=[],
        sarcasm_flag=False,
    )
    assert result.aspects == []


# --- SHAPResult ---

def test_shap_result_fields():
    shap = SHAPResult(
        tokens=["The", "food", "was", "great"],
        shap_values=[0.01, 0.45, 0.02, 0.52],
        base_value=0.15,
    )
    assert shap.tokens == ["The", "food", "was", "great"]
    assert len(shap.shap_values) == 4
    assert shap.base_value == 0.15


# --- ModelInference abstract interface ---

def test_model_inference_cannot_be_instantiated_directly():
    """Concrete subclasses must implement all abstract methods."""
    import pytest
    with pytest.raises(TypeError):
        ModelInference()
```

- [ ] **Step 2.2: Run tests to confirm they fail**

```bash
pytest tests/contracts/test_model_interface.py -v
```

Expected: `ImportError: cannot import name 'AspectSentiment' from 'contracts.model_interface'`

- [ ] **Step 2.3: Implement `contracts/model_interface.py`**

```python
# contracts/model_interface.py
"""
Shared data types and the abstract ModelInference contract.

Quân imports ModelInference + data classes for FastAPI dependency injection.
Trung implements the concrete model by subclassing ModelInference.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

from contracts.errors import ModelError  # noqa: F401 (re-exported for convenience)


@dataclass
class AspectSentiment:
    """Sentiment for a single detected aspect."""

    aspect: str          # One of: food, service, ambiance, price, location, general
    sentiment: str       # "positive" | "negative" | "neutral"
    confidence: float    # 0.0 – 1.0


@dataclass
class PredictionResult:
    """Output of a single-text sentiment prediction."""

    sentiment: str                          # "positive" | "negative" | "neutral"
    confidence: float                       # 0.0 – 1.0 (overall sentiment)
    aspects: list[AspectSentiment] = field(default_factory=list)
    sarcasm_flag: bool = False


@dataclass
class SHAPResult:
    """SHAP explainability output for a single text."""

    tokens: list[str]          # Tokenized words
    shap_values: list[float]   # Contribution of each token (same length as tokens)
    base_value: float          # Baseline (model output before any token influence)


class ModelInference(abc.ABC):
    """
    Abstract interface for the sentiment model.

    Swap strategy:
        - Phase 1 (now): use MockModelInference(ModelInference)
        - Phase 2 (after training): use RealModelInference(ModelInference, model_path=...)
    """

    @abc.abstractmethod
    def predict_single(self, text: str, lang: str = "en") -> PredictionResult:
        """
        Predict sentiment for one text.
        Raises:
            UnsupportedLanguageError: if lang is not in supported_languages.
            ModelError: on any other model failure.
        """

    @abc.abstractmethod
    def predict_batch(
        self, texts: list[str], lang: str = "en"
    ) -> list[PredictionResult]:
        """
        Predict sentiment for a list of texts.
        Returns results in the same order as input.
        """

    @abc.abstractmethod
    def get_shap_explanation(self, text: str, lang: str = "en") -> SHAPResult:
        """
        Compute SHAP token-attribution values for one text.
        Long uses the returned tokens + shap_values to render the bar/heatmap chart.
        """

    @property
    @abc.abstractmethod
    def supported_languages(self) -> list[str]:
        """Languages this model can handle. MVP: ['en']"""

    @property
    @abc.abstractmethod
    def is_loaded(self) -> bool:
        """True once the model weights are loaded and ready. Quân uses this in /health."""
```

- [ ] **Step 2.4: Run tests to confirm they pass**

```bash
pytest tests/contracts/test_model_interface.py -v
```

Expected output:
```
tests/contracts/test_model_interface.py::test_aspect_sentiment_fields PASSED
tests/contracts/test_model_interface.py::test_aspect_sentiment_is_dataclass PASSED
tests/contracts/test_model_interface.py::test_prediction_result_fields PASSED
tests/contracts/test_model_interface.py::test_prediction_result_empty_aspects PASSED
tests/contracts/test_model_interface.py::test_shap_result_fields PASSED
tests/contracts/test_model_interface.py::test_model_inference_cannot_be_instantiated_directly PASSED
6 passed
```

- [ ] **Step 2.5: Commit**

```bash
git add contracts/model_interface.py tests/contracts/test_model_interface.py
git commit -m "feat(contracts): add data classes and ModelInference abstract interface"
```

---

## Task 3: MockModelInference

**Files:**
- Create: `contracts/mock_model.py`
- Create: `tests/contracts/test_mock_model.py`

- [ ] **Step 3.1: Write the failing tests**

```python
# tests/contracts/test_mock_model.py
import pytest
from contracts.mock_model import MockModelInference
from contracts.model_interface import ModelInference, PredictionResult, SHAPResult
from contracts.errors import ModelError, UnsupportedLanguageError

VALID_SENTIMENTS = {"positive", "negative", "neutral"}
VALID_ASPECTS = {"food", "service", "ambiance", "price", "location", "general"}


@pytest.fixture
def mock() -> MockModelInference:
    return MockModelInference()


# --- Interface compliance ---

def test_is_model_inference_subclass(mock):
    assert isinstance(mock, ModelInference)


def test_is_loaded_true(mock):
    assert mock.is_loaded is True


def test_supported_languages_contains_en(mock):
    assert "en" in mock.supported_languages


# --- predict_single ---

def test_predict_single_returns_prediction_result(mock):
    result = mock.predict_single("Great food!")
    assert isinstance(result, PredictionResult)


def test_predict_single_valid_sentiment(mock):
    result = mock.predict_single("Great food!")
    assert result.sentiment in VALID_SENTIMENTS


def test_predict_single_confidence_in_range(mock):
    result = mock.predict_single("Great food!")
    assert 0.0 <= result.confidence <= 1.0


def test_predict_single_aspects_are_valid(mock):
    result = mock.predict_single("Great food!")
    assert 1 <= len(result.aspects) <= 3
    for asp in result.aspects:
        assert asp.aspect in VALID_ASPECTS
        assert asp.sentiment in VALID_SENTIMENTS
        assert 0.0 <= asp.confidence <= 1.0


def test_predict_single_sarcasm_flag_is_bool(mock):
    result = mock.predict_single("Great food!")
    assert isinstance(result.sarcasm_flag, bool)


def test_predict_single_unsupported_language_raises(mock):
    with pytest.raises(UnsupportedLanguageError):
        mock.predict_single("Bonjour", lang="fr")


# --- predict_batch ---

def test_predict_batch_returns_correct_length(mock):
    texts = ["Good food", "Bad service", "Meh ambiance"]
    results = mock.predict_batch(texts)
    assert len(results) == 3


def test_predict_batch_all_prediction_results(mock):
    results = mock.predict_batch(["a", "b"])
    for r in results:
        assert isinstance(r, PredictionResult)


def test_predict_batch_empty_list(mock):
    results = mock.predict_batch([])
    assert results == []


# --- get_shap_explanation ---

def test_shap_returns_shap_result(mock):
    result = mock.get_shap_explanation("Great food, bad service!")
    assert isinstance(result, SHAPResult)


def test_shap_tokens_and_values_same_length(mock):
    result = mock.get_shap_explanation("Great food, bad service!")
    assert len(result.tokens) == len(result.shap_values)


def test_shap_tokens_are_nonempty(mock):
    result = mock.get_shap_explanation("Great food!")
    assert len(result.tokens) >= 1


def test_shap_base_value_is_float(mock):
    result = mock.get_shap_explanation("Great food!")
    assert isinstance(result.base_value, float)
```

- [ ] **Step 3.2: Run tests to confirm they fail**

```bash
pytest tests/contracts/test_mock_model.py -v
```

Expected: `ImportError: cannot import name 'MockModelInference'`

- [ ] **Step 3.3: Implement `contracts/mock_model.py`**

```python
# contracts/mock_model.py
"""
MockModelInference — a drop-in fake for the real model.

Usage (Quân's FastAPI):
    from contracts.mock_model import MockModelInference

    model = MockModelInference()

    @app.post("/api/v1/predict")
    def predict(req: PredictRequest):
        result = model.predict_single(req.text, lang=req.lang)
        return { ... }

    # When Trung's real model is ready, swap to:
    # from ai_core.model import RealModelInference
    # model = RealModelInference(model_path="weights/")
"""
from __future__ import annotations

import random
import time

from contracts.errors import UnsupportedLanguageError
from contracts.model_interface import (
    AspectSentiment,
    ModelInference,
    PredictionResult,
    SHAPResult,
)

_SENTIMENTS = ["positive", "negative", "neutral"]
_ASPECTS = ["food", "service", "ambiance", "price", "location", "general"]
_SUPPORTED_LANGUAGES = ["en"]


class MockModelInference(ModelInference):
    """
    Returns random but schema-valid data.

    Latency: sleep(0.03–0.08 s) to simulate inference.
    Sarcasm: 10% probability of True.
    """

    # --- Interface properties ---

    @property
    def supported_languages(self) -> list[str]:
        return list(_SUPPORTED_LANGUAGES)

    @property
    def is_loaded(self) -> bool:
        return True

    # --- Helpers ---

    def _check_language(self, lang: str) -> None:
        if lang not in _SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(lang)

    def _random_prediction(self) -> PredictionResult:
        time.sleep(random.uniform(0.03, 0.08))
        n_aspects = random.randint(1, 3)
        aspects = [
            AspectSentiment(
                aspect=asp,
                sentiment=random.choice(_SENTIMENTS),
                confidence=round(random.uniform(0.55, 0.99), 2),
            )
            for asp in random.sample(_ASPECTS, n_aspects)
        ]
        return PredictionResult(
            sentiment=random.choice(_SENTIMENTS),
            confidence=round(random.uniform(0.55, 0.99), 2),
            aspects=aspects,
            sarcasm_flag=random.random() < 0.10,
        )

    def _random_shap(self, text: str) -> SHAPResult:
        tokens = text.split()
        if not tokens:
            tokens = ["<empty>"]
        shap_values = [round(random.uniform(-0.5, 0.5), 3) for _ in tokens]
        base_value = round(random.uniform(0.1, 0.3), 3)
        return SHAPResult(tokens=tokens, shap_values=shap_values, base_value=base_value)

    # --- Interface implementation ---

    def predict_single(self, text: str, lang: str = "en") -> PredictionResult:
        self._check_language(lang)
        return self._random_prediction()

    def predict_batch(
        self, texts: list[str], lang: str = "en"
    ) -> list[PredictionResult]:
        self._check_language(lang)
        return [self._random_prediction() for _ in texts]

    def get_shap_explanation(self, text: str, lang: str = "en") -> SHAPResult:
        self._check_language(lang)
        return self._random_shap(text)
```

- [ ] **Step 3.4: Run tests to confirm they pass**

```bash
pytest tests/contracts/test_mock_model.py -v
```

Expected output:
```
tests/contracts/test_mock_model.py::test_is_model_inference_subclass PASSED
tests/contracts/test_mock_model.py::test_is_loaded_true PASSED
tests/contracts/test_mock_model.py::test_supported_languages_contains_en PASSED
tests/contracts/test_mock_model.py::test_predict_single_returns_prediction_result PASSED
tests/contracts/test_mock_model.py::test_predict_single_valid_sentiment PASSED
tests/contracts/test_mock_model.py::test_predict_single_confidence_in_range PASSED
tests/contracts/test_mock_model.py::test_predict_single_aspects_are_valid PASSED
tests/contracts/test_mock_model.py::test_predict_single_sarcasm_flag_is_bool PASSED
tests/contracts/test_mock_model.py::test_predict_single_unsupported_language_raises PASSED
tests/contracts/test_mock_model.py::test_predict_batch_returns_correct_length PASSED
tests/contracts/test_mock_model.py::test_predict_batch_all_prediction_results PASSED
tests/contracts/test_mock_model.py::test_predict_batch_empty_list PASSED
tests/contracts/test_mock_model.py::test_shap_returns_shap_result PASSED
tests/contracts/test_mock_model.py::test_shap_tokens_and_values_same_length PASSED
tests/contracts/test_mock_model.py::test_shap_tokens_are_nonempty PASSED
tests/contracts/test_mock_model.py::test_shap_base_value_is_float PASSED
16 passed
```

- [ ] **Step 3.5: Commit**

```bash
git add contracts/mock_model.py tests/contracts/test_mock_model.py
git commit -m "feat(contracts): add MockModelInference with fully random valid outputs"
```

---

## Task 4: Pydantic schemas for all 6 API endpoints

**Files:**
- Create: `contracts/schemas.py`
- Create: `tests/contracts/test_schemas.py`

- [ ] **Step 4.1: Write the failing tests**

```python
# tests/contracts/test_schemas.py
import pytest
from pydantic import ValidationError
from contracts.schemas import (
    PredictRequest,
    AspectSentimentOut,
    PredictResponse,
    ExplainRequest,
    ExplainResponse,
    BatchSubmitResponse,
    BatchStatusResponse,
    HealthResponse,
)


# --- PredictRequest ---

def test_predict_request_valid():
    req = PredictRequest(text="Great food", lang="en")
    assert req.text == "Great food"
    assert req.lang == "en"


def test_predict_request_default_lang():
    req = PredictRequest(text="Hello")
    assert req.lang == "en"


def test_predict_request_text_too_long():
    with pytest.raises(ValidationError):
        PredictRequest(text="x" * 2001)


def test_predict_request_empty_text():
    with pytest.raises(ValidationError):
        PredictRequest(text="")


# --- PredictResponse ---

def test_predict_response_parses_sample():
    data = {
        "text": "The food was great but service was slow",
        "sentiment": "positive",
        "confidence": 0.72,
        "aspects": [
            {"aspect": "food", "sentiment": "positive", "confidence": 0.95},
            {"aspect": "service", "sentiment": "negative", "confidence": 0.88},
        ],
        "sarcasm_flag": False,
        "latency_ms": 45.2,
    }
    resp = PredictResponse(**data)
    assert resp.sentiment == "positive"
    assert len(resp.aspects) == 2
    assert resp.sarcasm_flag is False


# --- ExplainResponse ---

def test_explain_response_parses_sample():
    data = {
        "tokens": ["The", "food", "was", "great"],
        "shap_values": [0.01, 0.45, 0.02, 0.52],
        "base_value": 0.15,
        "latency_ms": 125.0,
    }
    resp = ExplainResponse(**data)
    assert len(resp.tokens) == len(resp.shap_values)


# --- BatchSubmitResponse ---

def test_batch_submit_response():
    data = {
        "job_id": "batch_abc123",
        "status": "pending",
        "total_items": 150,
        "created_at": "2026-04-14T10:30:00Z",
    }
    resp = BatchSubmitResponse(**data)
    assert resp.job_id == "batch_abc123"
    assert resp.status == "pending"


# --- BatchStatusResponse ---

def test_batch_status_with_no_completed_at():
    data = {
        "job_id": "batch_abc123",
        "status": "processing",
        "progress": 0.65,
        "total_items": 150,
        "processed_items": 97,
        "created_at": "2026-04-14T10:30:00Z",
        "completed_at": None,
    }
    resp = BatchStatusResponse(**data)
    assert resp.completed_at is None
    assert resp.progress == 0.65


# --- HealthResponse ---

def test_health_response():
    data = {
        "status": "healthy",
        "model_loaded": True,
        "version": "0.1.0",
        "supported_languages": ["en"],
    }
    resp = HealthResponse(**data)
    assert resp.model_loaded is True
    assert "en" in resp.supported_languages
```

- [ ] **Step 4.2: Run tests to confirm they fail**

```bash
pytest tests/contracts/test_schemas.py -v
```

Expected: `ImportError: cannot import name 'PredictRequest' from 'contracts.schemas'`

- [ ] **Step 4.3: Implement `contracts/schemas.py`**

```python
# contracts/schemas.py
"""
Pydantic v2 models for all 6 API endpoints.

Quân wires these directly into FastAPI route signatures:
    @app.post("/api/v1/predict", response_model=PredictResponse)
    def predict(req: PredictRequest): ...

FastAPI generates Swagger UI from these automatically — Long reads the /docs page
to get all type information without reading this file directly.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

class AspectSentimentOut(BaseModel):
    aspect: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# POST /api/v1/predict
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    lang: str = Field(default="en")


class PredictResponse(BaseModel):
    text: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    aspects: list[AspectSentimentOut] = []
    sarcasm_flag: bool
    latency_ms: float


# ---------------------------------------------------------------------------
# POST /api/v1/explain
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    lang: str = Field(default="en")


class ExplainResponse(BaseModel):
    tokens: list[str]
    shap_values: list[float]
    base_value: float
    latency_ms: float


# ---------------------------------------------------------------------------
# POST /api/v1/batch  (multipart handled by FastAPI UploadFile, not Pydantic)
# ---------------------------------------------------------------------------

class BatchSubmitResponse(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    total_items: int
    created_at: str  # ISO 8601 string; FastAPI serializes datetime to str automatically


# ---------------------------------------------------------------------------
# GET /api/v1/batch/{job_id}
# ---------------------------------------------------------------------------

class BatchStatusResponse(BaseModel):
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    progress: float = Field(ge=0.0, le=1.0)
    total_items: int
    processed_items: int
    created_at: str
    completed_at: str | None = None


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    version: str
    supported_languages: list[str]
```

- [ ] **Step 4.4: Run tests to confirm they pass**

```bash
pytest tests/contracts/test_schemas.py -v
```

Expected output:
```
tests/contracts/test_schemas.py::test_predict_request_valid PASSED
tests/contracts/test_schemas.py::test_predict_request_default_lang PASSED
tests/contracts/test_schemas.py::test_predict_request_text_too_long PASSED
tests/contracts/test_schemas.py::test_predict_request_empty_text PASSED
tests/contracts/test_schemas.py::test_predict_response_parses_sample PASSED
tests/contracts/test_schemas.py::test_explain_response_parses_sample PASSED
tests/contracts/test_schemas.py::test_batch_submit_response PASSED
tests/contracts/test_schemas.py::test_batch_status_with_no_completed_at PASSED
tests/contracts/test_schemas.py::test_health_response PASSED
9 passed
```

- [ ] **Step 4.5: Commit**

```bash
git add contracts/schemas.py tests/contracts/test_schemas.py
git commit -m "feat(contracts): add Pydantic v2 schemas for all 6 API endpoints"
```

---

## Task 5: Sample batch CSV

**Files:**
- Create: `contracts/sample_batch_input.csv`

- [ ] **Step 5.1: Create the CSV file**

Create `contracts/sample_batch_input.csv` with the content below (25 rows, columns: `text,lang`):

```csv
text,lang
"The pizza was absolutely amazing, best I've ever had!",en
"Service was incredibly slow and the waiter was rude.",en
"The ambiance is nice but the food is just average.",en
"Prices are a bit high for what you get.",en
"Great location, right in the heart of the city.",en
"I loved the pasta but the tiramisu was disappointing.",en
"The staff went above and beyond to make our anniversary special.",en
"Food arrived cold and looked nothing like the menu photo.",en
"Decent place for a quick lunch, nothing spectacular.",en
"Best sushi in town, will definitely come back!",en
"The music was too loud and we couldn't hold a conversation.",en
"Very cozy atmosphere, perfect for a date night.",en
"The burger was overcooked and the fries were soggy.",en
"Exceptional value for money, huge portions at fair prices.",en
"Waited 45 minutes for our appetizers. Unacceptable.",en
"The chef personally came to our table to check on us — lovely touch.",en
"Drinks were watered down and overpriced.",en
"Fresh ingredients, authentic flavors, outstanding experience.",en
"The restrooms were dirty and the tables were sticky.",en
"Vegetarian options are plentiful and delicious.",en
"I asked for medium-rare steak and got well-done. Never again.",en
"Great service but the food was just okay.",en
"The dessert menu is to die for — try the molten lava cake!",en
"Noisy environment but the food makes up for it.",en
"Surprisingly good for the price. Highly recommend the fish tacos.",en
```

- [ ] **Step 5.2: Verify the CSV is well-formed**

```bash
python -c "
import csv
with open('contracts/sample_batch_input.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    assert len(rows) == 25, f'Expected 25 rows, got {len(rows)}'
    for row in rows:
        assert 'text' in row and 'lang' in row, f'Bad row: {row}'
        assert len(row['text']) > 0
print(f'OK — {len(rows)} rows, columns: {list(rows[0].keys())}')
"
```

Expected: `OK — 25 rows, columns: ['text', 'lang']`

- [ ] **Step 5.3: Commit**

```bash
git add contracts/sample_batch_input.csv
git commit -m "feat(contracts): add 25-row sample batch input CSV for backend and frontend testing"
```

---

## Task 6: Sample API responses JSON

**Files:**
- Create: `contracts/sample_responses.json`

- [ ] **Step 6.1: Create `contracts/sample_responses.json`**

```json
{
  "POST /api/v1/predict": {
    "request": {
      "text": "The food was great but service was slow",
      "lang": "en"
    },
    "response": {
      "text": "The food was great but service was slow",
      "sentiment": "positive",
      "confidence": 0.72,
      "aspects": [
        {"aspect": "food", "sentiment": "positive", "confidence": 0.95},
        {"aspect": "service", "sentiment": "negative", "confidence": 0.88}
      ],
      "sarcasm_flag": false,
      "latency_ms": 45.2
    }
  },
  "POST /api/v1/explain": {
    "request": {
      "text": "The food was great but service was slow",
      "lang": "en"
    },
    "response": {
      "tokens": ["The", "food", "was", "great", "but", "service", "was", "slow"],
      "shap_values": [0.01, 0.45, 0.02, 0.52, -0.05, -0.40, 0.01, -0.30],
      "base_value": 0.15,
      "latency_ms": 125.0
    }
  },
  "POST /api/v1/batch": {
    "note": "Multipart form upload. Attach sample_batch_input.csv as the file field.",
    "response": {
      "job_id": "batch_abc123",
      "status": "pending",
      "total_items": 25,
      "created_at": "2026-04-14T10:30:00Z"
    }
  },
  "GET /api/v1/batch/{job_id}": {
    "note": "Replace {job_id} with the value returned by POST /api/v1/batch",
    "response_processing": {
      "job_id": "batch_abc123",
      "status": "processing",
      "progress": 0.65,
      "total_items": 25,
      "processed_items": 16,
      "created_at": "2026-04-14T10:30:00Z",
      "completed_at": null
    },
    "response_completed": {
      "job_id": "batch_abc123",
      "status": "completed",
      "progress": 1.0,
      "total_items": 25,
      "processed_items": 25,
      "created_at": "2026-04-14T10:30:00Z",
      "completed_at": "2026-04-14T10:30:45Z"
    }
  },
  "GET /api/v1/batch/{job_id}/result": {
    "note": "Returns a CSV file download. HTTP 404 if job not completed. HTTP 410 if result expired.",
    "csv_columns": ["text", "lang", "sentiment", "confidence", "aspects_json", "sarcasm_flag"],
    "csv_example_row": {
      "text": "The food was great but service was slow",
      "lang": "en",
      "sentiment": "positive",
      "confidence": "0.72",
      "aspects_json": "[{\"aspect\":\"food\",\"sentiment\":\"positive\",\"confidence\":0.95},{\"aspect\":\"service\",\"sentiment\":\"negative\",\"confidence\":0.88}]",
      "sarcasm_flag": "false"
    }
  },
  "GET /api/v1/health": {
    "response": {
      "status": "healthy",
      "model_loaded": true,
      "version": "0.1.0",
      "supported_languages": ["en"]
    }
  }
}
```

- [ ] **Step 6.2: Verify the JSON is valid**

```bash
python -c "
import json
with open('contracts/sample_responses.json') as f:
    data = json.load(f)
expected_keys = [
    'POST /api/v1/predict',
    'POST /api/v1/explain',
    'POST /api/v1/batch',
    'GET /api/v1/batch/{job_id}',
    'GET /api/v1/batch/{job_id}/result',
    'GET /api/v1/health',
]
for key in expected_keys:
    assert key in data, f'Missing key: {key}'
print('OK — all 6 endpoints documented.')
"
```

Expected: `OK — all 6 endpoints documented.`

- [ ] **Step 6.3: Commit**

```bash
git add contracts/sample_responses.json
git commit -m "feat(contracts): add sample_responses.json covering all 6 API endpoints"
```

---

## Task 7: contracts/README.md — integration guide

**Files:**
- Create: `contracts/README.md`

- [ ] **Step 7.1: Create `contracts/README.md`**

```markdown
# contracts/

Shared interface package for the Sentiment Analysis Service.

**Owner:** Trung (AI Core) — do NOT edit these files without creating a PR and notifying Quan and Long.
**Breaking changes** must be announced ≥ 1 day in advance.

---

## Files

| File | Purpose |
|------|---------|
| `errors.py` | `ModelError` (→ HTTP 500), `UnsupportedLanguageError` (→ HTTP 400) |
| `model_interface.py` | Data classes (`AspectSentiment`, `PredictionResult`, `SHAPResult`) + abstract `ModelInference` |
| `mock_model.py` | `MockModelInference` — returns random valid data with fake latency |
| `schemas.py` | Pydantic v2 request/response models for all 6 endpoints |
| `sample_batch_input.csv` | 25 rows of `text,lang` for testing batch upload |
| `sample_responses.json` | Example request + response for each endpoint |

---

## For Quan (Backend)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Wire MockModelInference into FastAPI

```python
# app/dependencies.py
from contracts.mock_model import MockModelInference
from contracts.model_interface import ModelInference

_model: ModelInference = MockModelInference()

def get_model() -> ModelInference:
    return _model
```

```python
# app/routers/predict.py
from fastapi import APIRouter, Depends
from contracts.schemas import PredictRequest, PredictResponse
from contracts.errors import ModelError, UnsupportedLanguageError
from app.dependencies import get_model
import time

router = APIRouter()

@router.post("/api/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest, model=Depends(get_model)):
    start = time.time()
    try:
        result = model.predict_single(req.text, lang=req.lang)
    except UnsupportedLanguageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelError as e:
        raise HTTPException(status_code=500, detail=str(e))
    latency_ms = (time.time() - start) * 1000
    return PredictResponse(
        text=req.text,
        sentiment=result.sentiment,
        confidence=result.confidence,
        aspects=[{"aspect": a.aspect, "sentiment": a.sentiment, "confidence": a.confidence} for a in result.aspects],
        sarcasm_flag=result.sarcasm_flag,
        latency_ms=round(latency_ms, 2),
    )
```

### 3. /health endpoint

```python
@router.get("/api/v1/health", response_model=HealthResponse)
def health(model=Depends(get_model)):
    return HealthResponse(
        status="healthy" if model.is_loaded else "degraded",
        model_loaded=model.is_loaded,
        version="0.1.0",
        supported_languages=model.supported_languages,
    )
```

### 4. Swapping to the real model (when Trung is ready)

Replace `MockModelInference` with the real class. Everything else stays the same:

```python
# app/dependencies.py
from ai_core.model import RealModelInference   # Trung's implementation
_model = RealModelInference(model_path="weights/sentiment_model")
```

---

## For Long (Frontend)

### 1. Get all types from Swagger UI

Start the backend server and open `http://localhost:8000/docs`. FastAPI generates full OpenAPI documentation from `schemas.py` — this is your source of truth for request shapes and response types.

### 2. Batch upload test

Use `sample_batch_input.csv` to test the file-picker → upload → poll → download flow.
The file has 25 rows with columns `text` and `lang`.

### 3. SHAP visualization

Call `POST /api/v1/explain` with a text. The response contains:
- `tokens`: array of words (x-axis labels)
- `shap_values`: signed floats (bar heights; positive = pushes toward positive sentiment)
- `base_value`: baseline prediction before any token influence

Render as a horizontal bar chart. Positive values: green bars. Negative values: red bars.

---

## Running tests

```bash
pytest tests/contracts/ -v --cov=contracts --cov-report=term-missing
```
```

- [ ] **Step 7.2: Verify README renders correctly**

```bash
# Quick sanity check: ensure all file references in README exist
python -c "
import os
files = [
    'contracts/errors.py',
    'contracts/model_interface.py',
    'contracts/mock_model.py',
    'contracts/schemas.py',
    'contracts/sample_batch_input.csv',
    'contracts/sample_responses.json',
]
for f in files:
    assert os.path.exists(f), f'Missing: {f}'
    print(f'  OK: {f}')
print('All files present.')
"
```

Expected:
```
  OK: contracts/errors.py
  OK: contracts/model_interface.py
  OK: contracts/mock_model.py
  OK: contracts/schemas.py
  OK: contracts/sample_batch_input.csv
  OK: contracts/sample_responses.json
All files present.
```

- [ ] **Step 7.3: Commit**

```bash
git add contracts/README.md
git commit -m "docs(contracts): add integration README for Quan and Long"
```

---

## Task 8: Update `contracts/__init__.py` + full test suite run

**Files:**
- Modify: `contracts/__init__.py`

- [ ] **Step 8.1: Expose public symbols from `contracts/__init__.py`**

```python
# contracts/__init__.py
"""
contracts — shared interface package for the Sentiment Analysis Service.

Quick imports for Quân's FastAPI:
    from contracts import MockModelInference, PredictRequest, PredictResponse
    from contracts.errors import ModelError, UnsupportedLanguageError
"""
from contracts.errors import ModelError, UnsupportedLanguageError
from contracts.model_interface import (
    AspectSentiment,
    ModelInference,
    PredictionResult,
    SHAPResult,
)
from contracts.mock_model import MockModelInference
from contracts.schemas import (
    AspectSentimentOut,
    BatchStatusResponse,
    BatchSubmitResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

__all__ = [
    "ModelError",
    "UnsupportedLanguageError",
    "AspectSentiment",
    "ModelInference",
    "PredictionResult",
    "SHAPResult",
    "MockModelInference",
    "AspectSentimentOut",
    "BatchStatusResponse",
    "BatchSubmitResponse",
    "ExplainRequest",
    "ExplainResponse",
    "HealthResponse",
    "PredictRequest",
    "PredictResponse",
]
```

- [ ] **Step 8.2: Run full test suite with coverage**

```bash
pytest tests/contracts/ -v --cov=contracts --cov-report=term-missing
```

Expected final output (35 tests):
```
============================= test session results ==============================
tests/contracts/test_errors.py              4 passed
tests/contracts/test_model_interface.py     6 passed
tests/contracts/test_mock_model.py         16 passed
tests/contracts/test_schemas.py             9 passed

---------- coverage: contracts ----------
contracts/errors.py          100%
contracts/model_interface.py 100%
contracts/mock_model.py       98%
contracts/schemas.py         100%
contracts/__init__.py        100%

35 passed in <N>s
```

Coverage should be ≥ 95% for the `contracts/` package.

- [ ] **Step 8.3: Final commit**

```bash
git add contracts/__init__.py
git commit -m "feat(contracts): expose public API via __init__.py; all 35 tests passing"
```

---

## Self-Review Checklist

### Spec coverage

| Spec requirement | Task |
|-----------------|------|
| `ModelError`, `UnsupportedLanguageError` | Task 1 |
| `AspectSentiment`, `PredictionResult`, `SHAPResult` dataclasses | Task 2 |
| `ModelInference` abstract class with all 5 methods/properties | Task 2 |
| `MockModelInference` with random data, 10% sarcasm, 0.03–0.08s sleep | Task 3 |
| Pydantic schemas for all 6 endpoints (all constraints) | Task 4 |
| `sample_batch_input.csv` 20-30 rows | Task 5 |
| `sample_responses.json` one example per endpoint | Task 6 |
| `contracts/README.md` usage guide for Quân and Long | Task 7 |
| `contracts/__init__.py` package re-exports | Task 8 |

### Placeholder scan

No TBD, TODO, or "similar to Task N" placeholders. All steps contain exact file content.

### Type consistency

- `AspectSentiment` defined in Task 2, used in `PredictionResult` (Task 2), `MockModelInference` (Task 3), `PredictResponse` (Task 4) — all consistent.
- `_random_prediction()` in Task 3 constructs `AspectSentiment` exactly: `aspect=`, `sentiment=`, `confidence=`.
- `predict_batch` in Task 3 calls `_random_prediction()` not `predict_single` — no sleep stacking/recursion issue.
