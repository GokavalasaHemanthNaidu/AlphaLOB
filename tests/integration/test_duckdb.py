import pytest
import sys
from unittest.mock import patch, MagicMock

# Mock duckdb globally BEFORE any imports happen
sys.modules['duckdb'] = MagicMock()

# Now we can safely import our client
from src.infrastructure.duckdb_client import init_db, create_backtest_run
@patch("src.infrastructure.duckdb_client.duckdb")
def test_duckdb_client_init(mock_duckdb):
    mock_conn = MagicMock()
    mock_duckdb.connect.return_value = mock_conn
    
    init_db()
    
    # Check that execute was called at least 3 times (for the 3 tables)
    assert mock_conn.execute.call_count >= 3
    mock_conn.close.assert_called_once()

@patch("src.infrastructure.duckdb_client.duckdb")
def test_duckdb_client_create_run(mock_duckdb):
    mock_conn = MagicMock()
    mock_duckdb.connect.return_value = mock_conn
    
    run_id = create_backtest_run({"strategy": "test"})
    assert isinstance(run_id, str)
    
    # Check that insert was called
    mock_conn.execute.assert_called_once()
    sql_arg = mock_conn.execute.call_args[0][0]
    assert "INSERT INTO backtest_runs" in sql_arg
    mock_conn.close.assert_called_once()
