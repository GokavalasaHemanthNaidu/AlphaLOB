import asyncio
import logging
from typing import Dict, Any
from src.domain.features import FeatureEngine

logger = logging.getLogger(__name__)

class FeatureEngineeringWorker:
    """
    Background worker that consumes raw LOB snapshots from a queue,
    calculates quantitative features, and pushes them to a downstream queue.
    Strictly adheres to memory constraints using asyncio.Queue.
    """
    def __init__(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.engine = FeatureEngine(depth_levels=10)
        self.running = False
        
    async def start(self):
        self.running = True
        logger.info("FeatureEngineeringWorker started.")
        
        while self.running:
            try:
                # Await raw snapshot from Phase 1 ingestion
                snapshot = await self.input_queue.get()
                
                # Compute features
                features = self.engine.process(snapshot)
                
                if features:
                    # Enrich original snapshot with features or send features directly
                    enriched_data = {
                        "snapshot": snapshot,
                        "features": features
                    }
                    # Non-blocking push to Phase 3 queue
                    if not self.output_queue.full():
                        self.output_queue.put_nowait(enriched_data)
                    else:
                        logger.warning("Features output queue is full. Dropping data.")
                
                self.input_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("FeatureEngineeringWorker cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in feature processing: {e}")
                
    async def stop(self):
        self.running = False
        logger.info("FeatureEngineeringWorker stopped.")
