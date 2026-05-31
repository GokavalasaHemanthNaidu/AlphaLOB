import asyncio
import random
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SyntheticLOBGenerator:
    """
    Simulates a live Limit Order Book WebSocket stream.
    Generates realistic looking LOB snapshots (bids and asks) with statistical noise,
    allowing the entire AlphaLOB pipeline to be tested locally without internet.
    """
    def __init__(self, queue: asyncio.Queue, symbol: str = "BTCUSDT", ticks_per_second: int = 20):
        self.queue = queue
        self.symbol = symbol
        self.sleep_interval = 1.0 / ticks_per_second
        
        # Initial mid price
        self.mid_price = 65000.0
        self.tick_size = 0.5
        self.update_id = 1000000

    async def start(self):
        logger.info(f"Starting SYNTHETIC LOB Generator for {self.symbol} at {1/self.sleep_interval} ticks/sec...")
        try:
            while True:
                # Random walk for mid price
                if random.random() > 0.5:
                    self.mid_price += self.tick_size
                else:
                    self.mid_price -= self.tick_size
                
                # Introduce occasional "spread compression" anomaly
                # Usually spread is 1 tick (0.5), sometimes it widens
                spread = self.tick_size if random.random() > 0.1 else self.tick_size * 3

                bids = []
                asks = []
                
                best_bid = self.mid_price - (spread / 2)
                best_ask = self.mid_price + (spread / 2)

                # Generate 10 levels of depth
                for i in range(10):
                    bid_p = best_bid - (i * self.tick_size)
                    ask_p = best_ask + (i * self.tick_size)
                    
                    # Random volumes between 0.1 and 5.0 BTC
                    # Occasional volume spikes
                    bid_v = random.uniform(0.1, 5.0) if random.random() > 0.05 else random.uniform(10.0, 50.0)
                    ask_v = random.uniform(0.1, 5.0) if random.random() > 0.05 else random.uniform(10.0, 50.0)

                    bids.append([f"{bid_p:.2f}", f"{bid_v:.3f}"])
                    asks.append([f"{ask_p:.2f}", f"{ask_v:.3f}"])

                self.update_id += 1
                
                # Match Bybit WebSocket format
                synthetic_data = {
                    "s": self.symbol,
                    "b": bids,
                    "a": asks,
                    "u": self.update_id,
                    "seq": self.update_id
                }

                await self.queue.put(synthetic_data)
                
                if self.update_id % 100 == 0:
                    logger.debug(f"Generated 100 synthetic ticks. Queue size: {self.queue.qsize()}")

                await asyncio.sleep(self.sleep_interval)

        except asyncio.CancelledError:
            logger.info("Synthetic LOB Generator cancelled.")
