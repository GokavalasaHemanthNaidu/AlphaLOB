import duckdb
import json
import uuid
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

DB_PATH = "alphalob.duckdb"

def init_db():
    """
    Initializes the local DuckDB database to perfectly mimic TimescaleDB 
    tables (backtest_runs and trade_log) with zero server overhead.
    """
    conn = duckdb.connect(DB_PATH)
    
    # Table: alpha_signals
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alpha_signals (
            signal_id UUID PRIMARY KEY DEFAULT uuid(),
            timestamp TIMESTAMP NOT NULL,
            symbol VARCHAR NOT NULL,
            dir_5s FLOAT,
            dir_30s FLOAT,
            dir_5min FLOAT,
            spread_compress FLOAT,
            vol_imbalance FLOAT,
            confidence FLOAT,
            regime VARCHAR,
            model_version VARCHAR,
            latency_ms FLOAT
        )
    """)
    
    # Table: backtest_runs
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_runs (
            run_id UUID PRIMARY KEY DEFAULT uuid(),
            status VARCHAR DEFAULT 'QUEUED',
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            config JSON NOT NULL,
            sharpe_ratio FLOAT,
            sortino_ratio FLOAT,
            calmar_ratio FLOAT,
            omega_ratio FLOAT,
            max_drawdown FLOAT,
            win_rate FLOAT,
            total_trades INTEGER
        )
    """)
    
    # Table: trade_log
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trade_log (
            trade_id UUID PRIMARY KEY DEFAULT uuid(),
            run_id UUID,
            timestamp TIMESTAMP NOT NULL,
            symbol VARCHAR,
            side VARCHAR,
            price FLOAT NOT NULL,
            quantity FLOAT NOT NULL,
            slippage_bps FLOAT,
            market_impact FLOAT,
            pnl FLOAT,
            cum_pnl FLOAT
        )
    """)
    conn.close()
    logger.info("Local DuckDB database initialized successfully.")

def create_backtest_run(config: Dict[str, Any]) -> str:
    run_id = str(uuid.uuid4())
    conn = duckdb.connect(DB_PATH)
    conn.execute("""
        INSERT INTO backtest_runs (run_id, config) 
        VALUES (?, ?)
    """, (run_id, json.dumps(config)))
    conn.close()
    return run_id

def update_backtest_status(run_id: str, status: str, metrics: Dict[str, float] = None):
    conn = duckdb.connect(DB_PATH)
    if metrics:
        conn.execute("""
            UPDATE backtest_runs 
            SET status = ?, completed_at = CURRENT_TIMESTAMP,
                sharpe_ratio = ?, sortino_ratio = ?, calmar_ratio = ?, 
                omega_ratio = ?, max_drawdown = ?, win_rate = ?, total_trades = ?
            WHERE run_id = ?
        """, (status, metrics.get("sharpe", 0.0), metrics.get("sortino", 0.0),
              metrics.get("calmar", 0.0), metrics.get("omega", 0.0),
              metrics.get("max_drawdown", 0.0), metrics.get("win_rate", 0.0),
              metrics.get("total_trades", 0), run_id))
    else:
        conn.execute("""
            UPDATE backtest_runs SET status = ? WHERE run_id = ?
        """, (status, run_id))
    conn.close()

def get_backtest_run(run_id: str) -> Dict[str, Any]:
    conn = duckdb.connect(DB_PATH)
    result = conn.execute("SELECT * FROM backtest_runs WHERE run_id = ?", (run_id,)).fetchone()
    conn.close()
    
    if result:
        # DuckDB fetchone returns a tuple. We map it to dictionary.
        # Columns: run_id, status, started_at, completed_at, config, sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio, max_drawdown, win_rate, total_trades
        return {
            "run_id": result[0],
            "status": result[1],
            "started_at": result[2],
            "completed_at": result[3],
            "config": result[4],
            "sharpe_ratio": result[5],
            "sortino_ratio": result[6],
            "calmar_ratio": result[7],
            "omega_ratio": result[8],
            "max_drawdown": result[9],
            "win_rate": result[10],
            "total_trades": result[11]
        }
    return None
