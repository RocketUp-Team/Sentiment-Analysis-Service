import json
from pathlib import Path

import pytest
from pydantic import ValidationError

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


def test_predict_request_stores_text_and_lang():
    request = PredictRequest(text="Great food", lang="en")

    assert request.text == "Great food"
    assert request.lang == "en"


def test_predict_request_defaults_lang_to_none():
    request = PredictRequest(text="Hello")

    assert request.lang is None


def test_predict_request_rejects_text_longer_than_2000_chars():
    with pytest.raises(ValidationError):
        PredictRequest(text="a" * 2001)


def test_predict_request_rejects_empty_text():
    with pytest.raises(ValidationError):
        PredictRequest(text="")


def test_predict_response_parses_sample_data():
    response = PredictResponse(
        text="Great food",
        sentiment="positive",
        confidence=0.93,
        aspects=[
            {"aspect": "food", "sentiment": "positive", "confidence": 0.98},
            AspectSentimentOut(aspect="service", sentiment="neutral", confidence=0.61),
        ],
        sarcasm_flag=False,
        detected_lang="en",
        lang_confidence=1.0,
        latency_ms=12.5,
    )

    assert response.text == "Great food"
    assert response.sentiment == "positive"
    assert response.confidence == 0.93
    assert len(response.aspects) == 2
    assert response.aspects[0].aspect == "food"
    assert response.aspects[1].sentiment == "neutral"
    assert response.sarcasm_flag is False
    assert response.detected_lang == "en"
    assert response.lang_confidence == 1.0
    assert response.latency_ms == 12.5


def test_explain_response_parses_sample_data_and_lengths_match():
    response = ExplainResponse(
        tokens=["great", "food", "today"],
        shap_values=[0.12, 0.34, -0.05],
        base_value=0.01,
        latency_ms=8.2,
    )

    assert response.tokens == ["great", "food", "today"]
    assert response.shap_values == [0.12, 0.34, -0.05]
    assert len(response.tokens) == len(response.shap_values)
    assert response.base_value == 0.01
    assert response.latency_ms == 8.2


def test_explain_response_rejects_mismatched_token_and_shap_lengths():
    with pytest.raises(ValidationError):
        ExplainResponse(
            tokens=["great", "food", "today"],
            shap_values=[0.12, 0.34],
            base_value=0.01,
            latency_ms=8.2,
        )


def test_sample_explain_payload_validates_against_schema():
    sample_path = Path(__file__).resolve().parents[2] / "contracts" / "sample_responses.json"
    sample_data = json.loads(sample_path.read_text())

    response = ExplainResponse.model_validate(sample_data["POST /api/v1/explain"]["response"])

    assert response.tokens == ["great", "food", "today"]
    assert response.shap_values == [0.12, 0.34, -0.05]


def test_batch_submit_response_parses_job_metadata():
    response = BatchSubmitResponse(
        job_id="job_123",
        status="pending",
        total_items=42,
        created_at="2026-04-15T09:30:00Z",
    )

    assert response.job_id == "job_123"
    assert response.status == "pending"
    assert response.total_items == 42
    assert response.created_at == "2026-04-15T09:30:00Z"


def test_batch_status_response_parses_processing_state():
    response = BatchStatusResponse(
        job_id="job_123",
        status="processing",
        progress=0.5,
        total_items=42,
        processed_items=21,
        created_at="2026-04-15T09:30:00Z",
        completed_at=None,
    )

    assert response.job_id == "job_123"
    assert response.status == "processing"
    assert response.progress == 0.5
    assert response.total_items == 42
    assert response.processed_items == 21
    assert response.created_at == "2026-04-15T09:30:00Z"
    assert response.completed_at is None


def test_health_response_parses_service_metadata():
    response = HealthResponse(
        status="healthy",
        model_loaded=True,
        version="1.2.3",
        supported_languages=["en", "vi"],
    )

    assert response.status == "healthy"
    assert response.model_loaded is True
    assert response.version == "1.2.3"
    assert response.supported_languages == ["en", "vi"]
