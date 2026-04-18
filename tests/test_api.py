import pytest
from fastapi.testclient import TestClient
from unittest import mock
import torch
from src.api.main import app

client = TestClient(app)

class MockModelOutput:
    def __init__(self, logits):
        self.logits = logits

def test_read_root():
    """Prueba el endpoint de salud de la API."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "API is running"

@mock.patch("src.api.main.model")
@mock.patch("src.api.main.tokenizer")
def test_predict_positive(mock_tokenizer, mock_model):
    """Simula una predicción positiva sin cargar el modelo real."""
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]), 
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    
    mock_model.return_value = MockModelOutput(torch.tensor([[0.1, 5.0]]))
    
    payload = {"text": "I love this service!"}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    assert response.json()["prediction"][0]["label"] == "POSITIVE"

@mock.patch("src.api.main.model")
@mock.patch("src.api.main.tokenizer")
def test_predict_negative(mock_tokenizer, mock_model):
    """Simula una predicción negativa."""
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]), 
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    
    mock_model.return_value = MockModelOutput(torch.tensor([[5.0, 0.1]]))
    
    payload = {"text": "This is terrible."}
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    assert response.json()["prediction"][0]["label"] == "NEGATIVE"

def test_predict_no_model():
    """Verifica que la API responda 500 si el modelo no cargó."""
    with mock.patch("src.api.main.model", None):
        response = client.post("/predict", json={"text": "Test"})
        assert response.status_code == 500
        assert "Modelo no disponible" in response.json()["detail"]