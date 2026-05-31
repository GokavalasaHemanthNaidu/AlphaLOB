import logging
import numpy as np
from src.infrastructure.mlflow_sqlite import get_recent_distributions

logger = logging.getLogger(__name__)

def calculate_psi(expected_dist, actual_dist, buckets=10):
    """
    Calculates Population Stability Index (PSI) to detect data/concept drift.
    PSI > 0.2 indicates significant drift requiring model retraining.
    """
    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    expected_percents = np.percentile(expected_dist, breakpoints)
    
    expected_fractions = np.histogram(expected_dist, expected_percents)[0] / len(expected_dist)
    actual_fractions = np.histogram(actual_dist, expected_percents)[0] / len(actual_dist)
    
    def sub_psi(e_perc, a_perc):
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001
        return (e_perc - a_perc) * np.log(e_perc / a_perc)
    
    psi_value = np.sum(sub_psi(expected_fractions, actual_fractions))
    return psi_value

def run_drift_check():
    """
    Nightly MLOps job to check for data drift using the Evidently AI methodology.
    """
    logger.info("Starting Nightly Data Drift Detection (PSI Check)...")
    
    recent_data = get_recent_distributions(limit=5000)
    if len(recent_data) < 100:
        logger.warning("Not enough data to perform drift detection.")
        return
        
    # Extract prediction arrays
    # recent_data rows: (dir_up_prob, dir_down_prob, spread_compress_prob)
    actual_up_probs = np.array([row[0] for row in recent_data])
    
    # In production, this would be loaded from your training validation set
    # For demo purposes, we simulate the expected training distribution
    expected_up_probs = np.random.normal(loc=0.5, scale=0.1, size=len(actual_up_probs))
    
    try:
        psi_score = calculate_psi(expected_up_probs, actual_up_probs)
        logger.info(f"Calculated Population Stability Index (PSI): {psi_score:.4f}")
        
        if psi_score > 0.2:
            logger.critical(f"CRITICAL DRIFT DETECTED! PSI ({psi_score:.4f}) > 0.2. Triggering retraining pipeline!")
        elif psi_score > 0.1:
            logger.warning(f"Minor drift detected. PSI ({psi_score:.4f}) > 0.1. Monitoring closely.")
        else:
            logger.info("No significant drift detected. Model is stable.")
    except Exception as e:
        logger.error(f"Error calculating PSI: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_drift_check()
