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
            import random
            # Return dummy predictions so the live demo dashboard works even without Colab!
            return {
                "dir_up_prob": random.uniform(0.1, 0.9),
                "dir_flat_prob": 0.1,
                "dir_down_prob": random.uniform(0.1, 0.9),
                "spread_compress_prob": random.uniform(0.0, 1.0),
                "vol_imbalance_pred": random.uniform(-1.5, 1.5)
            }

        # Ensure float32 (ONNX standard)
        bids_input = bids_array.astype(np.float32)
        asks_input = asks_array.astype(np.float32)

        # ONNX Runtime inference
        inputs = {
            "bids": bids_input,
            "asks": asks_input
        }
        
        # Runs the model. Outputs order matches what we defined during export:
        # ["direction_logits", "spread_prob", "imbalance_pred"]
        outputs = self.session.run(None, inputs)
        
        # Softmax for logits
        direction_logits = outputs[0][0]
        # Softmax implementation in raw numpy:
        exp_preds = np.exp(direction_logits - np.max(direction_logits))
        dir_probs = exp_preds / exp_preds.sum()
        
        return {
            "dir_up_prob": float(dir_probs[0]),
            "dir_flat_prob": float(dir_probs[1]),
            "dir_down_prob": float(dir_probs[2]),
            "spread_compress_prob": float(outputs[1][0][0]),
            "vol_imbalance_pred": float(outputs[2][0][0])
        }
