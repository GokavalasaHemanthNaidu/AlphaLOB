import numpy as np

def calculate_square_root_impact(quantity: float, daily_volatility: float, avg_daily_volume: float, eta: float = 0.1) -> float:
    """
    Almgren-Chriss Square-Root Market Impact Model.
    Calculates the expected slippage / market impact in basis points (bps) or absolute terms.
    
    market_impact = eta * sigma * sqrt(|Q| / V_avg_daily)
    
    :param quantity: Size of the order |Q|
    :param daily_volatility: Daily price volatility (sigma)
    :param avg_daily_volume: Average daily trading volume (V_avg_daily)
    :param eta: Empirical constant (default 0.1)
    :return: Estimated market impact cost
    """
    if avg_daily_volume <= 0:
        return 0.0
    return eta * daily_volatility * np.sqrt(abs(quantity) / avg_daily_volume)
