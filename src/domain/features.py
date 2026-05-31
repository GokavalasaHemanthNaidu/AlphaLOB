import math
from typing import Dict, Any, Tuple

class FeatureEngine:
    """
    Computes mathematically rigorous real-time microstructure features from
    Limit Order Book (LOB) snapshots. Memory safe: maintains O(1) state.
    """
    def __init__(self, depth_levels: int = 10):
        self.depth_levels = depth_levels
        
        # State for previous orderbook to compute WOFI (Weighted Order Flow Imbalance)
        self.prev_bids = {}
        self.prev_asks = {}

    def _calculate_spread_and_depth(self, bids: Dict[float, float], asks: Dict[float, float]) -> Tuple[float, float, float]:
        """
        Calculates the bid-ask spread and total depth across configured levels.
        """
        best_bid = max(bids.keys()) if bids else 0.0
        best_ask = min(asks.keys()) if asks else 0.0
        
        spread = best_ask - best_bid if best_ask > 0 and best_bid > 0 else 0.0
        bid_depth = sum(bids.values())
        ask_depth = sum(asks.values())
        
        return spread, bid_depth, ask_depth

    def _calculate_wofi(self, curr_bids: Dict[float, float], curr_asks: Dict[float, float]) -> float:
        """
        Calculates Weighted Order Flow Imbalance (WOFI).
        Tracks the changes in bid/ask volume, weighting levels closer to the mid-price higher.
        """
        if not self.prev_bids or not self.prev_asks:
            self.prev_bids = curr_bids
            self.prev_asks = curr_asks
            return 0.0

        wofi = 0.0
        
        # We need sorted prices to apply exponential decay weights based on depth level
        sorted_bids = sorted(curr_bids.items(), key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(curr_asks.items(), key=lambda x: x[0])
        
        best_prev_bid = max(self.prev_bids.keys()) if self.prev_bids else 0.0
        best_prev_ask = min(self.prev_asks.keys()) if self.prev_asks else float('inf')
        
        # Bid side imbalance
        for i, (price, size) in enumerate(sorted_bids):
            weight = math.exp(-0.5 * i)
            prev_size = self.prev_bids.get(price, 0.0)
            
            if price > best_prev_bid:
                wofi += size * weight
            elif price == best_prev_bid:
                wofi += (size - prev_size) * weight
            else:
                wofi -= prev_size * weight
                
        # Ask side imbalance
        for i, (price, size) in enumerate(sorted_asks):
            weight = math.exp(-0.5 * i)
            prev_size = self.prev_asks.get(price, 0.0)
            
            if price < best_prev_ask:
                wofi -= size * weight
            elif price == best_prev_ask:
                wofi -= (size - prev_size) * weight
            else:
                wofi += prev_size * weight

        self.prev_bids = curr_bids
        self.prev_asks = curr_asks
        
        return wofi

    def process(self, snapshot: Dict[str, Any]) -> Dict[str, float]:
        """
        Processes a raw LOB snapshot dictionary and returns the feature vector.
        Expected snapshot format:
        {
            "b": [["60000.0", "1.5"], ...],
            "a": [["60001.0", "0.5"], ...],
            "ts": 1629811200000
        }
        """
        raw_bids = snapshot.get("b", [])
        raw_asks = snapshot.get("a", [])
        
        if not raw_bids or not raw_asks:
            return {}
            
        # Parse to dicts for fast lookups, respecting depth limit
        curr_bids = {float(p): float(s) for p, s in raw_bids[:self.depth_levels]}
        curr_asks = {float(p): float(s) for p, s in raw_asks[:self.depth_levels]}
        
        spread, bid_depth, ask_depth = self._calculate_spread_and_depth(curr_bids, curr_asks)
        wofi = self._calculate_wofi(curr_bids, curr_asks)
        
        return {
            "spread": spread,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "wofi": wofi
        }
