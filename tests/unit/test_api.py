import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    # Usually apps have a health endpoint. If it doesn't, this might fail.
    # We will test the root / instead since the landing page is there
    response = client.get("/")
    assert response.status_code == 200

def test_predict_endpoint_valid_payload():
    # Create dummy 10x4 matrix as expected by LOBSnapshot
    dummy_snapshot = [[0.0, 0.0, 0.0, 0.0] for _ in range(10)]
    payload = {"lob_snapshot": dummy_snapshot}
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "dir_5s_prob_up" in data
    assert "dir_30s_prob_up" in data
    assert "dir_5min_prob_up" in data
    assert "spread_compress" in data
    assert "vol_imbalance" in data

def test_regime_endpoint_valid_payload():
    payload = {
        "realized_vol": 0.05,
        "autocorrelation": -0.2
    }
    
    response = client.post("/regime", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "regime" in data
    assert "state_id" in data
    assert "probabilities" in data
