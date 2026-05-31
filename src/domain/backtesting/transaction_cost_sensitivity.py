import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simulate_transaction_cost_sweep():
    """
    Simulates the decay of Net Sharpe Ratio as slippage/transaction costs increase.
    For an HFT LOB model, the gross Sharpe might be high (e.g., 2.3), but 
    capacity is bottlenecked by the break-even transaction cost.
    """
    logger.info("Starting Transaction Cost Sensitivity Analysis Sweep...")
    
    # Baseline assumed gross performance (0 bps cost)
    gross_sharpe = 2.85 
    gross_annual_return = 0.42 # 42%
    gross_annual_vol = 0.147
    
    # Assumed average trade frequency per day
    trades_per_day = 120
    trading_days = 252
    
    costs_bps = np.arange(0.0, 10.5, 0.5)
    
    print("\n" + "="*70)
    print(f"{'Cost (bps)':<15} | {'Net Ann. Return':<20} | {'Net Sharpe Ratio':<20}")
    print("="*70)
    
    break_even_bps = None
    
    for cost in costs_bps:
        # Convert bps to decimal (1 bps = 0.0001)
        cost_decimal = cost / 10000.0
        
        # Annualized cost drag = cost per trade * trades per day * trading days
        annual_cost_drag = cost_decimal * trades_per_day * trading_days
        
        net_return = gross_annual_return - annual_cost_drag
        net_sharpe = net_return / gross_annual_vol
        
        if net_sharpe < 0 and break_even_bps is None:
            break_even_bps = cost - 0.5
            
        # Highlight the 5 bps mark which was requested by the reviewer
        marker = " <--- Reviewer Target (5 bps)" if cost == 5.0 else ""
        
        print(f"{cost:<15.1f} | {net_return*100:>12.2f}%         | {net_sharpe:>15.2f}{marker}")
        
    print("="*70)
    
    if break_even_bps:
        logger.info(f"Break-even transaction cost is approximately {break_even_bps} bps.")
    else:
        logger.info("Strategy remains profitable even at 10 bps cost (unrealistic for HFT!).")
        
if __name__ == "__main__":
    simulate_transaction_cost_sweep()
