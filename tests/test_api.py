import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import unittest.mock as mock

client = TestClient(app)

def test_read_root():
    """Test the root endpoint status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "API is running"

def test_predict_positive():
    """Test a positive sentiment prediction."""
    payload = {"text": "I love this bank, the service is excellent!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"][0]["label"] == "POSITIVE"

def test_predict_negative():
    """Test a negative sentiment prediction."""
    payload = {"text": "This is the worst experience ever, I hate the app."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"][0]["label"] == "NEGATIVE"

def test_predict_server_error():
    """Simulate a server error (500) during prediction."""
    with mock.patch("src.api.main.model", None):
        payload = {"text": "Any text"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 500
        assert response.json()["detail"] == "Modelo no disponible."

def test_predict_inference_exception():
    """Simulate an exception during the inference process."""
    with mock.patch("src.api.main.tokenizer", side_effect=Exception("Inference failure")):
        payload = {"text": "Any text"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 500
        assert "Error en inferencia manual" in response.json()["detail"]
