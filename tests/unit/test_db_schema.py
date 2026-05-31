from datetime import datetime, timezone
from src.infrastructure.timescaledb import LOBSnapshot
from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import postgresql

def test_lob_snapshot_schema_compilation():
    """
    Test that the SQLAlchemy LOBSnapshot model correctly compiles 
    to PostgreSQL statements with Arrays and the correct types.
    """
    # Create the CreateTable construct
    create_stmt = CreateTable(LOBSnapshot.__table__)
    
    # Compile the statement for PostgreSQL specifically
    compiled_stmt = str(create_stmt.compile(dialect=postgresql.dialect()))
    
    # Verify key columns exist in the generated SQL
    assert "lob_snapshots" in compiled_stmt
    assert "timestamp TIMESTAMP WITH TIME ZONE NOT NULL" in compiled_stmt
    assert "symbol VARCHAR(20) NOT NULL" in compiled_stmt
    assert "bid_prices FLOAT[] NOT NULL" in compiled_stmt
    assert "PRIMARY KEY (timestamp, symbol)" in compiled_stmt
