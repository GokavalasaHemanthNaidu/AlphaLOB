import pytest
import numpy as np
from src.domain.inference import ONNXPredictor

def test_inference_initialization():
    predictor = ONNXPredictor()
    assert predictor.session is not None
    assert predictor.model_path == "models/lob_transformer.onnx"

def test_inference_predict_dummy_data():
    predictor = ONNXPredictor()
    
    # Create dummy data: shape (1, 10, 2) for bids and asks
    # Because predict concatenates them to (1, 10, 4)
    bids = np.random.rand(1, 10, 2)
    asks = np.random.rand(1, 10, 2)
    
    # Run prediction
    result = predictor.predict(bids, asks)
    
    # Check that all keys are present
    assert "dir_up_prob" in result
    assert "dir_flat_prob" in result
    assert "dir_down_prob" in result
    assert "spread_compress_prob" in result
    assert "vol_imbalance_pred" in result
    
    # Assert values are floats in [0, 1] range (except maybe vol_imbalance_pred depending on activation, but mostly probabilities)
    assert 0.0 <= result["dir_up_prob"] <= 1.0
    assert 0.0 <= result["dir_down_prob"] <= 1.0
    assert result["dir_flat_prob"] == 0.0
    assert 0.0 <= result["spread_compress_prob"] <= 1.0
