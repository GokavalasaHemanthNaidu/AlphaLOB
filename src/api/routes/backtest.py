from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import asyncio
import polars as pl
import numpy as np

from src.infrastructure.duckdb_client import create_backtest_run, update_backtest_status, get_backtest_run
from src.domain.backtesting.engine import PolarsBacktester

router = APIRouter(prefix="/v1/backtest", tags=["backtest"])

class BacktestConfig(BaseModel):
    strategy: str
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    slippage_bps: float = 1.0
    use_impact_model: bool = True

def _run_backtest_job(run_id: str, config: BacktestConfig):
    """
    Background job to run the Polars backtest.
    In a real system, this would fetch data from TimescaleDB.
    Here we generate dummy data to simulate the process.
    """
    update_backtest_status(run_id, "RUNNING")
    
    try:
        # Use a local RNG — avoids polluting global np.random state
        rng = np.random.default_rng(42)
        n = 10000
        df = pl.DataFrame({
            "ts": range(n),
            "mid_price": np.cumprod(1 + rng.normal(0, 0.001, n)) * 60000,
            "dir_5min_prob": rng.uniform(0, 1, n),
            "daily_volatility": np.full(n, 0.02),
            "avg_daily_volume": np.full(n, 5000)
        })
        
        backtester = PolarsBacktester(
            initial_capital=config.initial_capital,
            slippage_bps=config.slippage_bps,
            use_impact_model=config.use_impact_model
        )
        
        report = backtester.run(df)
        
        # Save metrics to DB
        update_backtest_status(run_id, "COMPLETED", report)
        
    except Exception as e:
        # Store the error message so client can see it via /status
        update_backtest_status(run_id, "FAILED", {"error": str(e)})
        logger.error(f"Backtest {run_id} failed: {e}", exc_info=True)

@router.post("/run")
async def run_backtest(config: BacktestConfig, background_tasks: BackgroundTasks):
    # 1. Create entry in DB
    run_id = create_backtest_run(config.model_dump())
    
    # 2. Enqueue background task
    background_tasks.add_task(_run_backtest_job, run_id, config)
    
    return {"run_id": run_id, "status": "QUEUED"}

@router.get("/{run_id}/status")
async def get_status(run_id: str):
    run = get_backtest_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run ID not found")
    return {"run_id": run_id, "status": run["status"]}

@router.get("/{run_id}/report")
async def get_report(run_id: str):
    run = get_backtest_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run ID not found")
    
    if run["status"] != "COMPLETED":
        return {"run_id": run_id, "status": run["status"], "message": "Report not ready yet."}
        
    return {
        "run_id": run_id,
        "sharpe": run["sharpe_ratio"],
        "sortino": run["sortino_ratio"],
        "calmar": run["calmar_ratio"],
        "omega": run["omega_ratio"],
        "max_drawdown": run["max_drawdown"],
        "win_rate": run["win_rate"],
        "total_trades": run["total_trades"]
    }
