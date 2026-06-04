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
            try:
                # Wait for a new signal — 30s timeout prevents infinite hang
                # if the inference worker dies
                signal = await asyncio.wait_for(request_queue.get(), timeout=30.0)
                data_str = json.dumps(signal)
                yield f"data: {data_str}\n\n"
                request_queue.task_done()
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
    except asyncio.CancelledError:
        logger.info("SSE client disconnected.")

@router.get("/live")
async def live_signals(request):
    """
    Subscribes to the real-time stream of Alpha Signals.
    Uses request.app.state to avoid circular import from main.py.
    """
    alpha_signals_queue = request.app.state.alpha_signals_queue
    
    return StreamingResponse(
        sse_generator(alpha_signals_queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
