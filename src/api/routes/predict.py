from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel
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
    b: List[List[str]]  # bid prices and sizes
    a: List[List[str]]  # ask prices and sizes
    ts: int

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
