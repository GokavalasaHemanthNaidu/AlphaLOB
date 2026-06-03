"""
AlphaLOB Production FastAPI Service
Routes: /health, /predict, /regime
"""
import os
import numpy as np
import onnxruntime as ort
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ── Model paths (relative to app.py location) ────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONNX_PATH  = os.path.join(BASE_DIR, "models", "weights", "lobster_transformer.onnx")
HMM_PATH   = os.path.join(BASE_DIR, "models", "weights", "regime_hmm.pkl")

# ── Load models at startup ────────────────────────────────────────────────────
print("Loading ONNX model...")
ort_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
print(f"  ONNX loaded: {ONNX_PATH}")

print("Loading RegimeHMM...")
hmm_model = joblib.load(HMM_PATH)
assert hasattr(hmm_model, "regime_names"), "regime_names missing from HMM model!"
print(f"  HMM loaded: {HMM_PATH} | regimes: {list(hmm_model.regime_names.values())}")

# ── FastAPI application ───────────────────────────────────────────────────────
app = FastAPI(
    title="AlphaLOB Inference API",
    description="Real-time LOB inference: directional probability + regime detection",
    version="1.0.0",
)


# ── Request/Response schemas ─────────────────────────────────────────────────

class LOBSnapshot(BaseModel):
    """
    Input: one LOB snapshot as a 10×4 matrix.
    Rows = LOB levels 0..9 (best bid/ask first).
    Columns = [normalized_price_dist, log_normalized_vol, wofi_z, kyle_lambda_z]
    """
    lob_snapshot: List[List[float]]   # shape (10, 4)


class PredictResponse(BaseModel):
    dir_5s_prob_up:    float   # P(price UP in 5s)
    dir_30s_prob_up:   float   # P(price UP in 30s)  ← KEY metric
    dir_5min_prob_up:  float   # P(price UP in 5min)
    spread_compress:   float   # P(spread compresses) — sigmoid [0,1]
    vol_imbalance:     float   # order flow imbalance regression


class RegimeInput(BaseModel):
    realized_vol:    float   # rolling 100-tick std of log-returns
    autocorrelation: float   # rolling lag-1 autocorr of log-returns


class RegimeResponse(BaseModel):
    regime:       str          # "TRENDING", "MEAN_REVERTING", or "VOLATILE"
    state_id:     int          # integer HMM state (0, 1, or 2)
    probabilities: dict        # {regime_name: probability} for all states


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint. Returns 200 if models are loaded."""
    return {
        "status": "ok",
        "model":  "LOBTransformer",
        "version": "1.0.0",
        "onnx_loaded": ort_session is not None,
        "hmm_loaded":  hmm_model is not None,
        "regime_names": list(hmm_model.regime_names.values()),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(body: LOBSnapshot):
    """
    Accepts a single LOB snapshot (10 levels × 4 features).
    Returns 5 model outputs: directional probs (3 horizons) + spread + vol imbalance.
    """
    snapshot = body.lob_snapshot

    # Validate input shape
    if len(snapshot) != 10:
        raise HTTPException(status_code=422, detail=f"Expected 10 LOB levels, got {len(snapshot)}")
    for i, row in enumerate(snapshot):
        if len(row) != 4:
            raise HTTPException(status_code=422, detail=f"Level {i} has {len(row)} features, expected 4")

    # Convert to numpy and sanitize NaN/Inf BEFORE ONNX inference
    X = np.array(snapshot, dtype=np.float32).reshape(1, 10, 4)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ONNX inference
    outputs = ort_session.run(None, {"lob_snapshot": X})

    # Extract outputs (handle both (1,2) and (1,) shapes)
    def safe_prob(arr, idx):
        """Safely extract probability and ensure it is finite."""
        v = float(arr[0, idx]) if arr.ndim == 2 else float(arr[0])
        return v if np.isfinite(v) else 0.5

    def safe_scalar(arr):
        v = float(arr[0]) if arr.ndim == 1 else float(arr[0, 0])
        return v if np.isfinite(v) else 0.0

    return PredictResponse(
        dir_5s_prob_up    = safe_prob(outputs[0], 1),
        dir_30s_prob_up   = safe_prob(outputs[1], 1),
        dir_5min_prob_up  = safe_prob(outputs[2], 1),
        spread_compress   = safe_scalar(outputs[3]),
        vol_imbalance     = safe_scalar(outputs[4]),
    )


@app.post("/regime", response_model=RegimeResponse)
async def regime(body: RegimeInput):
    """
    Accepts realized_vol + autocorrelation.
    Returns: current market regime label + state probabilities.
    """
    # Build HMM input array
    X_hmm = np.array([[body.realized_vol, body.autocorrelation]], dtype=np.float64)
    X_hmm = np.nan_to_num(X_hmm, nan=0.0, posinf=0.0, neginf=0.0)

    # Predict state and probabilities
    state_id  = int(hmm_model.predict(X_hmm)[0])
    proba     = hmm_model.predict_proba(X_hmm)[0]   # (n_states,) array

    regime_name = hmm_model.regime_names[state_id]
    proba_dict  = {hmm_model.regime_names[i]: float(proba[i]) for i in range(len(proba))}

    return RegimeResponse(
        regime=regime_name,
        state_id=state_id,
        probabilities=proba_dict,
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
