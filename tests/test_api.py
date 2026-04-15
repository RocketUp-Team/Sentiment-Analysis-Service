from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
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

def test_explain():
    response = client.post(
        "/explain",
        json={"text": "I love this product!", "lang": "en"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert "shap_values" in data
    assert len(data["tokens"]) == len(data["shap_values"])

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text
