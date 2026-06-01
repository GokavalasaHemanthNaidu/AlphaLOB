import asyncio
import random
import time
import logging
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SyntheticLOBGenerator:
    """
    Simulates a live Limit Order Book WebSocket stream.
    Generates realistic LOB snapshots (bids and asks) with statistical noise.

    Two modes:
      1. Streaming (async): use __init__ + start() for the live FastAPI pipeline
      2. Batch (static):    use SyntheticLOBGenerator.generate_batch() for Colab training
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

                    # Random volumes between 0.1 and 5.0 BTC — occasional spikes
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

    @staticmethod
    def generate_batch(
        n_ticks: int = 5_000_000,
        symbol: str = "BTC-USDT",
        start_price: float = 65_000.0,
        tick_size: float = 0.5,
        n_levels: int = 10,
        annual_volatility: float = 0.80,
        annual_drift: float = 0.10,
        ticks_per_second: int = 10,
        spread_mean: float = 1.0,
        spread_theta: float = 0.05,
        spread_sigma: float = 0.1,
        chunk_size: int = 100_000,
        seed: int = 42,
        start_ts: datetime = None,
    ):
        """
        Batch mode: generates n_ticks LOB snapshots as a list of flat-column dicts.
        Used by Colab training notebooks to produce large datasets without asyncio overhead.

        Each chunk dict has keys:
          timestamp, symbol, mid_price, spread,
          bid_price_0..9, bid_vol_0..9, ask_price_0..9, ask_vol_0..9

        Price dynamics:  Geometric Brownian Motion (GBM)
        Spread dynamics: Ornstein-Uhlenbeck mean reversion
        Volumes:         Log-normal with fat-tailed liquidity shocks

        Example:
            chunks = SyntheticLOBGenerator.generate_batch(n_ticks=5_000_000)
            import polars as pl
            df = pl.concat([pl.DataFrame(c) for c in chunks])
        """
        np.random.seed(seed)
        if start_ts is None:
            start_ts = datetime(2023, 1, 1, 9, 30, 0)

        dt            = 1.0 / ticks_per_second
        annual_factor = 1.0 / (252 * 6.5 * 3600)   # per-second scaling
        mu            = annual_drift * annual_factor
        sigma         = annual_volatility * np.sqrt(annual_factor)

        current_price  = start_price
        current_spread = spread_mean
        current_ts     = start_ts
        all_chunks     = []
        n_full_chunks  = n_ticks // chunk_size
        remainder      = n_ticks % chunk_size

        total_chunks = n_full_chunks + (1 if remainder else 0)

        for chunk_idx in range(total_chunks):
            c_size = chunk_size if chunk_idx < n_full_chunks else remainder
            if c_size == 0:
                break

            # ── Mid-price: Geometric Brownian Motion ──────────────────────
            log_returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), c_size)
            prices = current_price * np.exp(np.cumsum(log_returns))
            prices = np.round(prices / tick_size) * tick_size  # quantise to tick grid

            # ── Spread: Ornstein-Uhlenbeck ────────────────────────────────
            spreads = np.zeros(c_size)
            s = current_spread
            noise = np.random.normal(0, 1, c_size)
            for i in range(c_size):
                s = s + spread_theta * (spread_mean - s) * dt + spread_sigma * np.sqrt(dt) * noise[i]
                spreads[i] = max(0.5, s)  # minimum 1 tick spread

            # Liquidity shocks (~0.5% of ticks)
            shock_mask = np.random.random(c_size) < 0.005
            if shock_mask.any():
                spreads[shock_mask] *= np.random.uniform(3, 10, shock_mask.sum())

            # ── Timestamps ────────────────────────────────────────────────
            timestamps = [current_ts + timedelta(milliseconds=100 * i) for i in range(c_size)]

            # ── Volumes: log-normal with fat-tailed spikes ────────────────
            base_vols  = np.random.lognormal(0.5, 1.0, (c_size, n_levels))
            vol_shocks = np.random.lognormal(3.0, 0.5, (c_size, n_levels))
            vol_mask   = np.random.random((c_size, n_levels)) < 0.02
            bid_vols   = np.where(vol_mask, vol_shocks, base_vols)
            ask_vols   = np.where(vol_mask, vol_shocks,
                                  np.random.lognormal(0.5, 1.0, (c_size, n_levels)))

            # ── Price levels ──────────────────────────────────────────────
            half_spread = spreads[:, None] / 2
            offsets     = np.arange(n_levels) * tick_size
            bid_prices  = prices[:, None] - half_spread - offsets[None, :]
            ask_prices  = prices[:, None] + half_spread + offsets[None, :]

            # ── Build flat-column dict (one column per LOB level) ─────────
            data = {
                'timestamp': timestamps,
                'symbol':    [symbol] * c_size,
                'mid_price': prices.tolist(),
                'spread':    spreads.tolist(),
            }
            for lvl in range(n_levels):
                data[f'bid_price_{lvl}'] = bid_prices[:, lvl].tolist()
                data[f'bid_vol_{lvl}']   = bid_vols[:, lvl].tolist()
                data[f'ask_price_{lvl}'] = ask_prices[:, lvl].tolist()
                data[f'ask_vol_{lvl}']   = ask_vols[:, lvl].tolist()

            all_chunks.append(data)

            # Advance state for next chunk (ensures price continuity)
            current_price  = float(prices[-1])
            current_spread = float(spreads[-1])
            current_ts    += timedelta(milliseconds=100 * c_size)

            if (chunk_idx + 1) % 10 == 0 or chunk_idx == total_chunks - 1:
                pct = min((chunk_idx + 1) * chunk_size, n_ticks) / n_ticks * 100
                logger.info(
                    f"SyntheticLOBGenerator.generate_batch: {pct:.0f}% "
                    f"| price=${current_price:,.0f} | chunk {chunk_idx+1}/{total_chunks}"
                )

        return all_chunks
