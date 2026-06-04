from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel, field_validator
import numpy as np

# Lazily loaded predictor
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        from src.domain.inference import ONNXPredictor
        _predictor = ONNXPredictor()
    return _predictor

router = APIRouter(prefix="/v1/predict", tags=["prediction"])

class LOBSnapshot(BaseModel):
    symbol: str
    b: List[List[float]]  # bid prices and sizes — [price, size] per level
    a: List[List[float]]  # ask prices and sizes — [price, size] per level
    ts: int

    @field_validator('b', 'a')
    @classmethod
    def validate_levels(cls, v):
        if not v:
            raise ValueError('LOB levels cannot be empty')
        for i, row in enumerate(v):
            if len(row) != 2:
                raise ValueError(f'Level {i} must have exactly [price, size], got {len(row)} values')
            if any(not isinstance(x, (int, float)) or x < 0 for x in row):
                raise ValueError(f'Level {i} contains invalid (negative or non-numeric) values')
        return v

@router.post("")
async def predict_single(snapshot: LOBSnapshot):
    """
    Synchronous fallback endpoint for single LOB predictions.
    Used for batch testing or isolated inference checks.
    """
    predictor = get_predictor()
    
    # Process into correct shapes (1, 10, 2)
    raw_bids = snapshot.b[:10]
    raw_asks = snapshot.a[:10]
    
    while len(raw_bids) < 10:
        raw_bids.append(["0.0", "0.0"])
    while len(raw_asks) < 10:
        raw_asks.append(["0.0", "0.0"])
        
    bids_arr = np.array(raw_bids, dtype=np.float32).reshape(1, 10, 2)
    asks_arr = np.array(raw_asks, dtype=np.float32).reshape(1, 10, 2)
    
    predictions = predictor.predict(bids_arr, asks_arr)
    
    if "error" in predictions:
        raise HTTPException(status_code=500, detail=predictions["error"])
        
    return {
        "status": "success",
        "symbol": snapshot.symbol,
        "timestamp": snapshot.ts,
        "predictions": predictions
    }
