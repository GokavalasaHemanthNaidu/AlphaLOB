from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class LOBSnapshotRequest(BaseModel):
    """
    Pydantic schema for incoming Limit Order Book snapshots.
    Validates that we receive exactly 10 levels of bids and asks.
    """
    symbol: str = Field(..., json_schema_extra={"example": "BTCUSDT"})
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Bids and asks expected as list of [price, volume]
    bids: List[List[float]] = Field(..., min_length=10, max_length=10, json_schema_extra={"example": [[60000.5, 1.5]]*10})
    asks: List[List[float]] = Field(..., min_length=10, max_length=10, json_schema_extra={"example": [[60001.0, 0.5]]*10})

class AlphaSignalResponse(BaseModel):
    """
    Pydantic schema for outgoing Alpha Signals (Model Predictions).
    """
    symbol: str
    timestamp: datetime
    
    # Probabilities for 3 classes: Up, Down, Flat
    dir_up_prob: float
    dir_down_prob: float
    dir_flat_prob: float
    
    # Binary classification: Will spread compress?
    spread_compress_prob: float
    
    # Regression: Expected volume imbalance magnitude
    vol_imbalance_pred: float
    
    # Metadata
    model_version: str = "v1.0.0"
    latency_ms: float
