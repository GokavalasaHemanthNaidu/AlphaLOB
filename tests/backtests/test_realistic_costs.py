import pytest
from src.domain.backtesting.impact_model import calculate_square_root_impact

def test_square_root_impact_scales_correctly():
    # market_impact = eta * sigma * sqrt(|Q| / V_avg_daily)
    eta = 0.1
    sigma = 0.02
    v_daily = 10000.0
    
    # Small trade (100 units)
    impact_small = calculate_square_root_impact(100.0, sigma, v_daily, eta)
    # 0.1 * 0.02 * sqrt(100 / 10000) = 0.002 * 0.1 = 0.0002
    assert pytest.approx(impact_small, 0.00001) == 0.0002
    
    # Large trade (400 units) - quantity is 4x, so impact should be 2x (square root)
    impact_large = calculate_square_root_impact(400.0, sigma, v_daily, eta)
    assert pytest.approx(impact_large, 0.00001) == 0.0004
    
    # Zero volume failsafe
    impact_zero = calculate_square_root_impact(100.0, sigma, 0.0, eta)
    assert impact_zero == 0.0
