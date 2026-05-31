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
        
        # Local order book state to maintain full depth from deltas
        self.local_bids = {}
        self.local_asks = {}
        
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
                            
                            # We must maintain the order book locally because Bybit sends deltas!
                            if "b" in data:
                                for price, size in data["b"]:
                                    if float(size) == 0:
                                        self.local_bids.pop(price, None)
                                    else:
                                        self.local_bids[price] = size
                                        
                            if "a" in data:
                                for price, size in data["a"]:
                                    if float(size) == 0:
                                        self.local_asks.pop(price, None)
                                    else:
                                        self.local_asks[price] = size
                                        
                            # Extract top 10 levels
                            sorted_bids = sorted(self.local_bids.items(), key=lambda x: float(x[0]), reverse=True)[:10]
                            sorted_asks = sorted(self.local_asks.items(), key=lambda x: float(x[0]))[:10]
                            
                            # Only emit to the model if we have a full 10-level book
                            if len(sorted_bids) >= 10 and len(sorted_asks) >= 10:
                                formatted_data = {
                                    "s": self.symbol,
                                    "b": sorted_bids,
                                    "a": sorted_asks,
                                    "u": data.get("u", 0),
                                    "seq": data.get("seq", 0),
                                    "type": "snapshot", # We have reconstructed it into a snapshot
                                    "ts": msg.get("ts", int(time.time() * 1000))
                                }
                                
                                await self.queue.put(formatted_data)
                                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Bybit WebSocket connection closed. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in BybitLiveLOBGenerator: {e}")
                await asyncio.sleep(5)
