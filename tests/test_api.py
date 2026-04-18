import pytest
import os
os.environ["MODEL_MODE"] = "baseline"
from fastapi.testclient import TestClient
from src.main import app

from unittest.mock import MagicMock
from contracts.schemas import PredictionResult, AspectSentimentOut, ExplainResult

@pytest.fixture
def mock_model():
    mock = MagicMock()
    mock.is_loaded = True
    
    mock.predict_single.return_value = PredictionResult(
        sentiment="positive",
        confidence=0.99,
        aspects=[AspectSentimentOut(aspect="foo", sentiment="positive", confidence=0.8)],
        sarcasm_flag=False,
    )
    
    mock.get_shap_explanation.return_value = ExplainResult(
        tokens=["I", "love", "this"],
        shap_values=[0.1, 0.8, 0.1],
        base_value=0.0
    )
    
    return mock

@pytest.fixture
def client(mock_model):
    from unittest.mock import patch
    from src.main import get_model
    
    # Mock the lifespan instantiation of the model to prevent massive HF downloads
    with patch('src.main.BaselineModelInference', return_value=mock_model):
        app.dependency_overrides[get_model] = lambda: mock_model
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.clear()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict(client):
    response = client.post(
        "/predict",
        json={"text": "I love this product!", "lang": "en"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data
    assert "aspects" in data
    assert data["text"] == "I love this product!"


def test_predict_without_lang_returns_detected_lang(client):
    response = client.post(
        "/predict",
        json={"text": "Dịch vụ này rất tuyệt vời và đáng tiền"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["detected_lang"] == "vi"
    assert data["lang_confidence"] > 0.8


def test_predict_with_explicit_lang_keeps_additive_detected_fields(client):
    response = client.post(
        "/predict",
        json={"text": "I love this product!", "lang": "en"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["detected_lang"] == "en"
    assert data["lang_confidence"] == 1.0

def test_explain(client):
    response = client.post(
        "/explain",
        json={"text": "I love this product!", "lang": "en"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert "shap_values" in data
    assert len(data["tokens"]) == len(data["shap_values"])

def test_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text
