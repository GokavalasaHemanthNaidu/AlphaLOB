import numpy as np

def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 365 * 24 * 60) -> float:
    """
    Annualized Sharpe Ratio.
    periods_per_year defaults to minute-level data for crypto.
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)

def calculate_sortino(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 365 * 24 * 60) -> float:
    """
    Annualized Sortino Ratio using downside deviation.
    """
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    downside_dev = np.sqrt(np.mean(downside_returns**2))
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_dev

def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculates the maximum peak-to-trough drop in the equity curve.
    """
    if len(equity_curve) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max
    return np.max(drawdowns)

def calculate_calmar(returns: np.ndarray, equity_curve: np.ndarray, periods_per_year: int = 365 * 24 * 60) -> float:
    """
    Annualized Return / Max Drawdown
    """
    max_dd = calculate_max_drawdown(equity_curve)
    if max_dd == 0:
        return 0.0
    compounded_return = (equity_curve[-1] / equity_curve[0]) - 1
    # Simple annualization for brevity
    years = len(returns) / periods_per_year
    annualized_return = (1 + compounded_return) ** (1 / years) - 1 if years > 0 else 0
    return annualized_return / max_dd

def calculate_omega(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Sum of returns above threshold / Sum of returns below threshold (absolute)
    """
    above = returns[returns > threshold] - threshold
    below = threshold - returns[returns < threshold]
    if np.sum(below) == 0:
        return 0.0
    return np.sum(above) / np.sum(below)
