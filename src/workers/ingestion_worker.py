import asyncio
import json
import logging
import websockets

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BybitIngestionWorker:
    def __init__(self, queue: asyncio.Queue, symbol: str = "BTCUSDT"):
        self.queue = queue
        self.symbol = symbol
        self.ws_url = "wss://stream.bybit.com/v5/public/spot"

    async def start(self):
        logger.info(f"Starting Bybit WebSocket ingestion for {self.symbol}...")
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    # Subscribe to the 50-level orderbook topic
                    subscribe_msg = {
                        "op": "subscribe",
                        "args": [f"orderbook.50.{self.symbol}"]
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info(f"Subscribed to orderbook.50.{self.symbol}")

                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        # Only process actual orderbook data (ignore ping/pong/auth/success msg)
                        if "topic" in data and "data" in data:
                            # Push the snapshot into our mock Kafka queue
                            await self.queue.put(data["data"])
                            if self.queue.qsize() % 100 == 0:
                                logger.info(f"Ingested 100 LOB snapshots. Queue size: {self.queue.qsize()}")
            
            except asyncio.CancelledError:
                logger.info("Ingestion worker cancelled.")
                break
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
