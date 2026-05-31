import asyncio
import logging
import numpy as np
from src.domain.inference import ONNXPredictor

logger = logging.getLogger(__name__)

class ModelInferenceWorker:
    """
    Background worker that consumes enriched features, 
    runs ONNX inference, and outputs final Alpha Signals.
    """
    def __init__(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.predictor = ONNXPredictor()
        self.running = False

    def _parse_snapshot_to_numpy(self, snapshot: dict) -> tuple:
        """
        Converts the raw snapshot dict into the numpy arrays required by ONNX.
        Returns bids_array, asks_array of shape (1, 10, 2)
        """
        # Ensure we have exactly 10 levels
        raw_bids = snapshot.get("b", [])[:10]
        raw_asks = snapshot.get("a", [])[:10]
        
        # Pad with zeros if less than 10 (edge case)
        while len(raw_bids) < 10:
            raw_bids.append([0.0, 0.0])
        while len(raw_asks) < 10:
            raw_asks.append([0.0, 0.0])
            
        bids = np.array(raw_bids, dtype=np.float32).reshape(1, 10, 2)
        asks = np.array(raw_asks, dtype=np.float32).reshape(1, 10, 2)
        
        return bids, asks

    async def start(self):
        self.running = True
        logger.info("ModelInferenceWorker started.")
        
        while self.running:
            try:
                # Await enriched data from Phase 2
                data = await self.input_queue.get()
                
                snapshot = data.get("snapshot", {})
                bids_arr, asks_arr = self._parse_snapshot_to_numpy(snapshot)
                
                import time
                from src.infrastructure.mlflow_sqlite import log_metric, log_prediction_distribution
                
                start_time = time.time()
                # Run CPU inference in a separate thread so it doesn't starve the async event loop (fixing 502 Bad Gateway)
                predictions = await asyncio.to_thread(self.predictor.predict, bids_arr, asks_arr)
                latency_ms = (time.time() - start_time) * 1000
                
                if "error" not in predictions:
                    # Log monitoring metrics
                    log_metric("inference_latency_ms", latency_ms)
                    log_prediction_distribution(
                        up_prob=predictions.get("dir_up_prob", 0.5),
                        down_prob=predictions.get("dir_down_prob", 0.5),
                        spread_prob=predictions.get("spread_compress_prob", 0.5)
                    )
                    
                    alpha_signal = {
                        "ts": snapshot.get("ts"),
                        "features": data.get("features"),
                        "predictions": predictions
                    }
                    
                    if not self.output_queue.full():
                        self.output_queue.put_nowait(alpha_signal)
                    
                self.input_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("ModelInferenceWorker cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in inference worker: {e}")
                
    async def stop(self):
        self.running = False
        logger.info("ModelInferenceWorker stopped.")
