from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
import numpy as np
from src.infrastructure.mlflow_sqlite import get_recent_metrics, get_recent_distributions
import math

router = APIRouter(prefix="/v1/health", tags=["Monitoring"])

@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """
    Exposes metrics in Prometheus format.
    """
    latencies = get_recent_metrics("inference_latency_ms", limit=100)
    
    if not latencies:
        return "# No data available yet\n"

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    avg = np.mean(latencies)

    lines = [
        "# HELP alphalob_inference_latency_ms Inference latency in milliseconds",
        "# TYPE alphalob_inference_latency_ms gauge",
        f"alphalob_inference_latency_ms{{quantile=\"0.5\"}} {p50}",
        f"alphalob_inference_latency_ms{{quantile=\"0.95\"}} {p95}",
        f"alphalob_inference_latency_ms{{quantile=\"0.99\"}} {p99}",
        f"alphalob_inference_latency_ms_avg {avg}"
    ]
    
    return "\n".join(lines) + "\n"

def compute_kl_divergence(p, q):
    """
    Computes KL Divergence D_KL(P || Q) for two discrete distributions.
    p: true distribution (training baseline)
    q: observed distribution (live)
    """
    # Add epsilon to prevent div by zero or log zero
    epsilon = 1e-5
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return np.sum(p * np.log(p / q))

@router.get("/drift")
async def detect_model_drift():
    """
    Compares live inference distribution against expected baseline distribution.
    If KL Divergence is high, it triggers a drift alert.
    """
    # Assuming during Colab training we found the average probabilities were balanced ~50/50
    # In reality, you'd load this from a baseline config.
    baseline_up_prob = 0.5
    baseline_down_prob = 0.5
    
    distributions = get_recent_distributions(limit=500)
    if not distributions:
        return {"status": "insufficient_data", "kl_divergence": 0.0}
        
    # Average the recent live probabilities
    live_up_avg = np.mean([row[0] for row in distributions])
    live_down_avg = np.mean([row[1] for row in distributions])
    
    # Normalize
    total = live_up_avg + live_down_avg
    if total > 0:
        live_up_avg /= total
        live_down_avg /= total
    else:
        live_up_avg, live_down_avg = 0.5, 0.5
        
    kl_div = compute_kl_divergence([baseline_up_prob, baseline_down_prob], [live_up_avg, live_down_avg])
    
    # Arbitrary threshold for demonstration
    drift_detected = kl_div > 0.05
    
    return {
        "status": "drift_detected" if drift_detected else "healthy",
        "kl_divergence": kl_div,
        "baseline": {"up": baseline_up_prob, "down": baseline_down_prob},
        "live": {"up": live_up_avg, "down": live_down_avg}
    }
