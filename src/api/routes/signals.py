from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/signals", tags=["signals"])

# We will lazily import the queue from main.py to avoid circular dependencies
# Alternatively, we could attach the queue to app.state

async def sse_generator(request_queue: asyncio.Queue):
    """
    Consumes alpha signals from the in-memory queue and formats them 
    as Server-Sent Events (SSE).
    """
    logger.info("New SSE client connected.")
    try:
        while True:
            # Wait for a new signal to be produced by the InferenceWorker
            signal = await request_queue.get()
            
            # SSE format requires "data: <json string>\n\n"
            data_str = json.dumps(signal)
            yield f"data: {data_str}\n\n"
            
            request_queue.task_done()
    except asyncio.CancelledError:
        logger.info("SSE client disconnected.")

@router.get("/live")
async def live_signals():
    """
    Subscribes to the real-time stream of Alpha Signals.
    Downstream systems (e.g., Backtesting Engine or Order Execution) 
    can listen to this endpoint.
    """
    from src.api.main import alpha_signals_queue
    
    return StreamingResponse(
        sse_generator(alpha_signals_queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
