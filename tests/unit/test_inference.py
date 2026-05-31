import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.domain.inference import ONNXPredictor

def test_onnx_predictor_logic():
    with patch("onnxruntime.InferenceSession") as mock_session, \
         patch("os.path.exists", return_value=True):
        # Configure the mock to return expected shapes for 3 heads:
        # [direction_logits (1, 3), spread_prob (1, 1), imbalance_pred (1, 1)]
        mock_instance = MagicMock()
        mock_instance.run.return_value = [
            np.array([[0.1, 0.5, 0.2]], dtype=np.float32),  # logits
            np.array([[0.85]], dtype=np.float32),           # spread compress prob
            np.array([[-1.5]], dtype=np.float32)            # vol imbalance
        ]
        mock_session.return_value = mock_instance
        
        # Initialize predictor (will call _load_model which uses the mock)
        predictor = ONNXPredictor(model_path="dummy_path.onnx")
        
        # Create valid dummy input (batch_size=1, num_levels=10, features=2)
        bids = np.random.randn(1, 10, 2).astype(np.float32)
        asks = np.random.randn(1, 10, 2).astype(np.float32)
        
        predictions = predictor.predict(bids, asks)
        
        assert "error" not in predictions
        
        # Check softmax logic on logits: [0.1, 0.5, 0.2]
        # Max is 0.5 -> exp([-0.4, 0.0, -0.3]) -> [0.67, 1.0, 0.74] -> sum ~ 2.41
        # Probs: ~ [0.27, 0.41, 0.30]
        assert np.isclose(predictions["dir_flat_prob"], np.exp(0.0) / (np.exp(-0.4) + np.exp(0.0) + np.exp(-0.3)), atol=0.01)
        
        # Check sum to 1
        sum_probs = predictions["dir_up_prob"] + predictions["dir_flat_prob"] + predictions["dir_down_prob"]
        assert np.isclose(sum_probs, 1.0)
        
        # Check direct passthrough heads
        assert np.isclose(predictions["spread_compress_prob"], 0.85, atol=1e-5)
        assert np.isclose(predictions["vol_imbalance_pred"], -1.5, atol=1e-5)
