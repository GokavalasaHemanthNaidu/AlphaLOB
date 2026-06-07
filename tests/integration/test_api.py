import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

# Mock the predictor so it doesn't fail on missing ONNX file
@pytest.fixture(autouse=True)
def mock_onnx_predictor():
    with patch("src.domain.inference.ONNXPredictor") as MockPredictor:
        instance = MockPredictor.return_value
        instance.predict.return_value = {
            "dir_up_prob": 0.6,
            "dir_flat_prob": 0.3,
            "dir_down_prob": 0.1,
            "spread_compress_prob": 0.8,
            "vol_imbalance_pred": -0.5
        }
        yield instance

from src.api.main import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "symbol": "BTCUSDT",
        "b": [["60000.0", "1.5"]],
        "a": [["60001.0", "2.0"]],
        "ts": 123456789
    }
    
    response = client.post("/v1/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["symbol"] == "BTCUSDT"
    
    preds = data["predictions"]
    assert preds["dir_up_prob"] == 0.6
    assert preds["vol_imbalance_pred"] == -0.5

def test_healthz_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "alpha_signals_queue_size" in response.json()

def test_root_demo_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AlphaLOB | Low-Latency" in response.text
