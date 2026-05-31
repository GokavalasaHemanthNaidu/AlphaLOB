import numpy as np
import onnxruntime as ort
import os
import logging

logger = logging.getLogger(__name__)

class ONNXPredictor:
    """
    Loads an exported ONNX model and performs fast CPU inference.
    Strictly avoids heavy frameworks like PyTorch to save RAM.
    """
    def __init__(self, model_path: str = "models/lob_transformer.onnx"):
        self.model_path = model_path
        self.session = None
        
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            logger.warning(f"Model {self.model_path} not found. Ensure you run the generator script.")

    def _load_model(self):
        # providers=['CPUExecutionProvider'] ensures we run purely on CPU to save memory
        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        logger.info(f"Loaded ONNX model from {self.model_path}")

    def predict(self, bids_array: np.ndarray, asks_array: np.ndarray) -> dict:
        """
        Runs inference on numpy arrays.
        bids_array and asks_array should have shape (batch_size, 10, 2)
        """
        if not self.session:
            return {"error": "Model not loaded"}

        # Ensure float32 (ONNX standard)
        bids_input = bids_array.astype(np.float32)
        asks_input = asks_array.astype(np.float32)

        # The Colab PyTorch model expects a single tensor of shape [batch, 10, 4]
        # (bid_price, bid_vol, ask_price, ask_vol)
        lob_snapshot = np.concatenate([bids_input, asks_input], axis=-1)

        # ONNX Runtime inference
        inputs = {
            "lob_snapshot": lob_snapshot
        }
        
        # Runs the model. Outputs order matches Colab export:
        # ['dir_5s', 'dir_30s', 'dir_5min', 'spread_compress', 'vol_imbalance']
        outputs = self.session.run(None, inputs)
        
        # Sigmoid is baked into the ONNX graph for the first 4 outputs, so they are already probabilities
        up_prob = float(outputs[0][0][0])
        down_prob = 1.0 - up_prob # Simple binary inversion for demo purposes
        
        return {
            "dir_up_prob": up_prob,
            "dir_flat_prob": 0.0,
            "dir_down_prob": down_prob,
            "spread_compress_prob": float(outputs[3][0][0]),
            "vol_imbalance_pred": float(outputs[4][0][0])
        }
