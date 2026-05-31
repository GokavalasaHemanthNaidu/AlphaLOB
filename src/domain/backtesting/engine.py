import polars as pl
import numpy as np
import logging
from typing import Dict, Any

from src.domain.backtesting.metrics import calculate_sharpe, calculate_max_drawdown, calculate_sortino, calculate_calmar, calculate_omega
from src.domain.backtesting.impact_model import calculate_square_root_impact

logger = logging.getLogger(__name__)

class PolarsBacktester:
    def __init__(self, initial_capital: float = 100000.0, slippage_bps: float = 1.0, use_impact_model: bool = True):
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps
        self.use_impact_model = use_impact_model
    
    def run(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Runs a vectorized backtest on a Polars DataFrame.
        Expected columns in df:
            - ts: timestamp
            - mid_price: float
            - dir_5min_prob: float (signal from model)
            - daily_volatility: float
            - avg_daily_volume: float
        """
        if df.height == 0:
            return self._empty_report()
            
        # 1. Generate trading signals based on probabilities
        # Example naive strategy: if UP prob > 0.6 -> buy (target position = 1)
        # if DOWN prob > 0.6 -> short (target position = -1)
        # else flat (target position = 0)
        
        # We assume dir_5min_prob is the predicted prob of mid_price going up.
        # If it's a regression signal, logic differs. Let's assume it's P(UP).
        # We'll use 0.6 threshold for UP, 0.4 threshold for DOWN (1 - P(UP) = P(DOWN)).
        
        df = df.with_columns([
            pl.when(pl.col("dir_5min_prob") > 0.6).then(1)
              .when(pl.col("dir_5min_prob") < 0.4).then(-1)
              .otherwise(0)
              .alias("target_position")
        ])
        
        # 2. Calculate Trades (changes in position)
        # Shift target_position by 1 to prevent lookahead bias (execute at next tick)
        df = df.with_columns([
            pl.col("target_position").shift(1).fill_null(0).alias("position")
        ])
        
        df = df.with_columns([
            (pl.col("position") - pl.col("position").shift(1).fill_null(0)).alias("trade_qty")
        ])
        
        # Calculate returns
        # Strategy Return = Position * Market Return
        df = df.with_columns([
            (pl.col("mid_price") / pl.col("mid_price").shift(1) - 1).fill_null(0.0).alias("market_return")
        ])
        
        df = df.with_columns([
            (pl.col("position").shift(1).fill_null(0) * pl.col("market_return")).alias("strategy_return_raw")
        ])
        
        # Calculate Costs
        # Slippage cost = |trade_qty| * slippage_bps / 10000
        df = df.with_columns([
            (pl.col("trade_qty").abs() * self.slippage_bps / 10000.0).alias("slippage_cost")
        ])
        
        # Market Impact
        if self.use_impact_model and "daily_volatility" in df.columns and "avg_daily_volume" in df.columns:
            # Vectorized square-root impact
            eta = 0.1
            # Impact = eta * sigma * sqrt(|Q| / V)
            # Assuming trade_qty is a fraction of capital, say 1 unit = $10,000 for simplicity in this dummy logic
            # To be mathematically rigorous, Q should be in shares. 
            # We'll mock the impact for demonstration:
            df = df.with_columns([
                (eta * pl.col("daily_volatility") * (pl.col("trade_qty").abs() / pl.col("avg_daily_volume")).sqrt()).fill_nan(0.0).alias("market_impact_cost")
            ])
        else:
            df = df.with_columns([pl.lit(0.0).alias("market_impact_cost")])
            
        df = df.with_columns([
            (pl.col("strategy_return_raw") - pl.col("slippage_cost") - pl.col("market_impact_cost")).alias("strategy_return_net")
        ])
        
        # Equity Curve
        df = df.with_columns([
            (1.0 + pl.col("strategy_return_net")).cum_prod().alias("equity_curve")
        ])
        
        # Extract numpy arrays for metrics
        returns_array = df["strategy_return_net"].to_numpy()
        equity_array = df["equity_curve"].to_numpy()
        
        total_trades = df.filter(pl.col("trade_qty") != 0).height
        winning_trades = df.filter((pl.col("trade_qty") != 0) & (pl.col("strategy_return_net") > 0)).height
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        return {
            "sharpe": calculate_sharpe(returns_array),
            "sortino": calculate_sortino(returns_array),
            "calmar": calculate_calmar(returns_array, equity_array),
            "omega": calculate_omega(returns_array),
            "max_drawdown": calculate_max_drawdown(equity_array),
            "total_trades": total_trades,
            "win_rate": win_rate,
            "equity_curve": equity_array.tolist()
        }

    def _empty_report(self) -> Dict[str, Any]:
        return {
            "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "omega": 0.0,
            "max_drawdown": 0.0, "total_trades": 0, "win_rate": 0.0,
            "equity_curve": []
        }
