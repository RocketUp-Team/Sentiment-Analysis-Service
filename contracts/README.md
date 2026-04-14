# contracts/

Shared interface package for the Sentiment Analysis Service.

Owner note: Trung (AI Core). Do not edit without a PR and notifying Quan and Long. Breaking changes must be announced at least 1 day in advance.

## Files

| File | Purpose |
| --- | --- |
| `errors.py` | Shared model-level exceptions, including `ModelError` and `UnsupportedLanguageError`. |
| `model_interface.py` | Abstract model contract plus shared dataclasses for prediction and SHAP output. |
| `mock_model.py` | Drop-in mock implementation for local development, tests, and API wiring examples. |
| `schemas.py` | FastAPI/Pydantic request and response schemas used by the HTTP layer and Swagger UI. |
| `sample_batch_input.csv` | Example batch upload input with `text` and `lang` columns. |
| `sample_responses.json` | Example API payloads for predict, explain, batch, and health responses. |

## For Quan (Backend)

Install the shared contracts and test dependencies first:

```bash
# run from the repository root
pip install -r requirements.txt
```

In this branch, `requirements.txt` only covers the shared contracts package and its test dependencies. The backend app still needs its own FastAPI/runtime dependencies installed separately before you can run the route examples below.

Use `MockModelInference` while wiring the API, but depend on the `ModelInference` interface so the real model can be swapped in later.

```python
from dataclasses import asdict
from time import perf_counter

from fastapi import Depends, FastAPI, HTTPException

from contracts import (
    HealthResponse,
    ModelError,
    ModelInference,
    MockModelInference,
    PredictRequest,
    PredictResponse,
    UnsupportedLanguageError,
)

app = FastAPI()


def get_model() -> ModelInference:
    return MockModelInference()


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    model: ModelInference = Depends(get_model),
) -> PredictResponse:
    start = perf_counter()
    try:
        result = model.predict_single(text=request.text, lang=request.lang)
    except UnsupportedLanguageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (perf_counter() - start) * 1000
    return PredictResponse(
        text=request.text,
        sentiment=result.sentiment,
        confidence=result.confidence,
        aspects=[asdict(aspect) for aspect in result.aspects],
        sarcasm_flag=result.sarcasm_flag,
        latency_ms=latency_ms,
    )


@app.get("/api/v1/health", response_model=HealthResponse)
def health(
    model: ModelInference = Depends(get_model),
) -> HealthResponse:
    return HealthResponse(
        status="healthy" if model.is_loaded else "degraded",
        model_loaded=model.is_loaded,
        version="0.1.0",
        supported_languages=model.supported_languages,
    )
```

For `/api/v1/predict`, map `UnsupportedLanguageError` to HTTP 400 and other `ModelError` failures to HTTP 500. Keep the latency measurement in the route so the API response reflects end-to-end request time.

When the production model is ready, replace `MockModelInference()` inside `get_model()` with the real implementation and keep the route code unchanged.

## For Long (Frontend)

Use Swagger UI at `http://localhost:8000/docs` as the source of truth for request and response shapes. The Pydantic schemas in `schemas.py` drive the docs that Long should rely on when building UI integration and validating payloads.
`contracts/sample_responses.json` also contains concrete sample payloads for predict, explain, batch, and health flows.

For batch upload testing, use `contracts/sample_batch_input.csv`. It already matches the expected CSV format with `text` and optional `lang` columns, so it is safe for manual end-to-end checks and UI file upload validation.

For SHAP visualization, read `tokens`, `shap_values`, and `base_value` together:

* `tokens` is the tokenized input text in the same order as the explanation values.
* `shap_values` shows each token’s contribution to the prediction.
* `base_value` is the baseline score before token contributions are applied.

Use green bars for positive contribution values and red bars for negative contribution values. Longer bars should indicate larger absolute impact, and the bars should align token-by-token with the values returned by the API.

## Running tests

```bash
# run from the repository root
pytest tests/contracts/ -v --cov=contracts --cov-report=term-missing
```
