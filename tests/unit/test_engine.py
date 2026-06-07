import pytest
import polars as pl
import numpy as np
from src.domain.backtesting.engine import PolarsBacktester

def test_engine_empty_dataframe():
    engine = PolarsBacktester()
    df = pl.DataFrame()
    result = engine.run(df)
    
    assert result["total_trades"] == 0
    assert result["sharpe"] == 0.0

def test_engine_basic_returns():
    engine = PolarsBacktester(slippage_bps=0.0, use_impact_model=False)
    
    # Create a synthetic dataframe with perfectly predictable moves
    # Price goes 100 -> 101 -> 102 -> 101 -> 100
    # Model predicts correctly 1 step ahead
    data = {
        "ts": [0, 31536000000, 63072000000, 94608000000, 126144000000], # Yearly ms to avoid overflow
        "mid_price": [100.0, 101.0, 102.0, 101.0, 100.0],
        "dir_5min_prob": [0.9, 0.9, 0.1, 0.1, 0.5], # 0.9 = strong buy, 0.1 = strong sell
        "daily_volatility": [0.01, 0.01, 0.01, 0.01, 0.01],
        "avg_daily_volume": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    }
    df = pl.DataFrame(data)
    
    result = engine.run(df)
    
    # We should have some winning trades and a positive sharpe ratio
    assert result["total_trades"] > 0
    assert "sharpe" in result
    assert len(result["equity_curve"]) == 5
