import asyncio
import json
import logging
import time
import websockets

logger = logging.getLogger(__name__)

class BybitLiveLOBGenerator:
    """
    Connects to Bybit v5 WebSocket API and streams live Level 2 Limit Order Book (LOB)
    snapshots directly into the AlphaLOB inference pipeline.
    This replaces the SyntheticLOBGenerator for live production deployments.
    """
    def __init__(self, queue: asyncio.Queue, symbol: str = "BTCUSDT"):
        self.queue = queue
        self.symbol = symbol
        self.ws_url = "wss://stream.bybit.com/v5/public/linear"
        
    async def start(self):
        logger.info(f"Connecting to LIVE Bybit WebSocket for {self.symbol}...")
        
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    logger.info("Connected to Bybit v5 public linear stream.")
                    
                    # Subscribe to orderbook depth 50
                    subscribe_msg = {
                        "op": "subscribe",
                        "args": [f"orderbook.50.{self.symbol}"]
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    
                    while True:
                        msg_str = await ws.recv()
                        msg = json.loads(msg_str)
                        
                        # Process order book snapshots and deltas
                        if "topic" in msg and msg["topic"] == f"orderbook.50.{self.symbol}":
                            data = msg.get("data", {})
                            
                            # We only care if there are bids and asks present
                            if "b" in data and "a" in data:
                                # Bybit format matches our synthetic generator format exactly:
                                # b: [[price, size], ...], a: [[price, size], ...]
                                
                                formatted_data = {
                                    "s": self.symbol,
                                    "b": data["b"][:10], # Keep top 10 levels
                                    "a": data["a"][:10], # Keep top 10 levels
                                    "u": data.get("u", 0),
                                    "seq": data.get("seq", 0),
                                    "type": msg.get("type", "delta"), # "snapshot" or "delta"
                                    "ts": msg.get("ts", int(time.time() * 1000))
                                }
                                
                                await self.queue.put(formatted_data)
                                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Bybit WebSocket connection closed. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in BybitLiveLOBGenerator: {e}")
                await asyncio.sleep(5)
