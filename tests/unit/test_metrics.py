import pytest
import numpy as np
from src.domain.backtesting.metrics import (
    calculate_sharpe, 
    calculate_sortino, 
    calculate_max_drawdown, 
    calculate_calmar,
    calculate_omega
)

def test_calculate_sharpe():
    # Returns with mean 0.001, std 0.01, and let's use 10000 periods
    returns = np.random.normal(0.001, 0.01, 10000)
    sharpe = calculate_sharpe(returns, periods_per_year=10000)
    
    # Expected: sqrt(10000) * 0.001 / 0.01 = 100 * 0.1 = 10
    assert np.isclose(sharpe, 10.0, atol=1.0)

def test_calculate_max_drawdown():
    # Simulate a drop from 1.0 down to 0.8 (20% drawdown) and back to 1.1
    equity = np.array([1.0, 0.9, 0.8, 1.1, 1.0])
    mdd = calculate_max_drawdown(equity)
    assert np.isclose(mdd, 0.2)

def test_calculate_omega():
    returns = np.array([-0.02, -0.01, 0.01, 0.03, 0.05])
    # Above 0 threshold: 0.01, 0.03, 0.05 -> sum = 0.09
    # Below 0 threshold: -0.02, -0.01 -> sum = 0.03
    # Omega = 0.09 / 0.03 = 3.0
    omega = calculate_omega(returns, threshold=0.0)
    assert np.isclose(omega, 3.0)
