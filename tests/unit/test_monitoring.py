from src.api.routes.model_health import compute_kl_divergence
import pytest

def test_kl_divergence_identical():
    # If the distribution is identical, KL divergence should be ~0
    p = [0.5, 0.5]
    q = [0.5, 0.5]
    
    kl = compute_kl_divergence(p, q)
    assert kl < 1e-4

def test_kl_divergence_drifted():
    # If live distribution heavily diverges from baseline, KL should be larger
    baseline = [0.5, 0.5]
    live = [0.9, 0.1]
    
    kl = compute_kl_divergence(baseline, live)
    assert kl > 0.1

def test_kl_divergence_edge_case():
    # Should handle zero probabilities gracefully via epsilon
    baseline = [0.5, 0.5]
    live = [1.0, 0.0]
    
    kl = compute_kl_divergence(baseline, live)
    # The epsilon prevents infinity, but KL should still be substantial
    assert kl > 0.1
