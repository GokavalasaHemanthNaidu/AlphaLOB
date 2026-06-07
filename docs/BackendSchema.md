# Backend Schema Document: AlphaLOB

*Note: This document is written retrospectively based on the actual schema definitions in `src/infrastructure/duckdb_client.py` and `src/infrastructure/mlflow_sqlite.py`.*

## 1. Core Entities
AlphaLOB relies on two separate local embedded databases:
- **DuckDB (`alphalob.duckdb`)**: Optimized for high-speed time-series analytics, backtesting, and trade logs.
  - `alpha_signals`
  - `backtest_runs`
  - `trade_log`
- **SQLite (`alphalob_mlflow.db`)**: Used as an MLflow-lite tracking server for drift detection and metrics.
  - `metrics`
  - `signal_distribution`
  - `backtest_runs` (MLflow tracking subset)

## 2. Entity Schemas

### DuckDB: `alpha_signals`
| Field | Type | Required? | Description |
|---|---|---|---|
| `signal_id` | UUID | Yes (Auto) | Primary Key. |
| `timestamp` | TIMESTAMP | Yes | Time the signal was generated. |
| `symbol` | VARCHAR | Yes | e.g., 'BTCUSDT'. |
| `dir_5s` / `dir_30s` / `dir_5min` | FLOAT | No | Directional up-probabilities for horizons. |
| `spread_compress` / `vol_imbalance` | FLOAT | No | Regression predictions. |
| `confidence` | FLOAT | No | Aggregated signal confidence. |
| `regime` | VARCHAR | No | Detected market regime (e.g., 'High Volatility'). |
| `model_version` | VARCHAR | No | Tracker for A/B testing models. |
| `latency_ms` | FLOAT | No | End-to-end inference latency. |

### DuckDB: `backtest_runs`
| Field | Type | Required? | Description |
|---|---|---|---|
| `run_id` | UUID | Yes (Auto) | Primary Key. |
| `status` | VARCHAR | No | Defaults to 'QUEUED'. |
| `started_at` / `completed_at` | TIMESTAMP | No | Execution timing. |
| `config` | JSON | Yes | The backtest parameters. |
| `sharpe_ratio` / `max_drawdown` / `win_rate` / etc. | FLOAT | No | Performance metrics. |
| `total_trades` | INTEGER | No | Trade count. |

### DuckDB: `trade_log`
| Field | Type | Required? | Description |
|---|---|---|---|
| `trade_id` | UUID | Yes (Auto) | Primary Key. |
| `run_id` | UUID | No | Logical Foreign Key (unenforced) linking to `backtest_runs`. |
| `timestamp` | TIMESTAMP | Yes | Trade execution time. |
| `symbol` / `side` | VARCHAR | No | Asset and 'BUY'/'SELL'. |
| `price` / `quantity` | FLOAT | Yes | Execution details. |
| `pnl` / `cum_pnl` | FLOAT | No | Profit and loss tracking. |

### SQLite (MLflow Tracking)
- **`metrics`**: `id` (INTEGER, PK), `timestamp` (REAL), `metric_name` (TEXT), `value` (REAL).
- **`signal_distribution`**: `id` (INTEGER, PK), `timestamp` (REAL), `dir_up_prob` (REAL), `dir_down_prob` (REAL), `spread_compress_prob` (REAL).
- **`backtest_runs`**: `id` (INTEGER, PK), `timestamp` (REAL), `model_version` (TEXT), `hyperparameters` (TEXT), `sharpe_ratio` (REAL), `max_drawdown` (REAL), `break_even_bps` (REAL).

## 3. Relationships
- **One-to-Many**: `backtest_runs.run_id` (1) → `trade_log.run_id` (N). Each backtest generates multiple trades.
- The MLflow SQLite tables are completely independent, append-only ledgers.

## 4. Indexing Strategy
While DuckDB auto-indexes primary keys, the following columns act as critical query filters and require (or will require) explicit indexes for fast lookups:
- `alpha_signals.timestamp` and `trade_log.timestamp` (Crucial for time-series range queries).
- `trade_log.run_id` (Crucial for fast PnL aggregation per backtest).
- `metrics.metric_name` (Used heavily by drift detector).

## 5. Authentication & Sessions
- **Provider**: None.
- **Sessions**: Stateless public API.
- **Security**: Handled purely via a custom FastAPI dictionary-based middleware (IP-based rate limiting, capped at 30 requests/min).

## 6. User Roles
- **Roles**: None. AlphaLOB is a single-tenant or server-to-server microservice. There is no concept of Admin vs. Guest within the application layer.

## 7. Row Level Security (RLS)
- **Rules**: None. Since the databases are local file-embedded (`alphalob.duckdb` and `alphalob_mlflow.db`), data isolation is handled at the container/filesystem level, not at the row level.

## 8. File / Media Storage
- **Storage Needed**: Yes, but only for system files, not user media.
- **Structure**:
  - `models/weights/lobster_transformer.onnx` (Deep Learning weights)
  - `models/weights/regime_hmm.bin` (HMM weights)
  - Root directory for `.duckdb` and `.db` local database files.

## 9. Sensitive Fields & Vaulting
- **Current State**: None. The system currently ingests *public* Limit Order Book data via WebSockets.
- **Future Need**: If live execution is integrated, Exchange API Keys and Secrets (e.g., Bybit Secret Key) will require external vaulting (e.g., HashiCorp Vault, AWS Secrets Manager) or heavily encrypted environment variables.

## 10. Webhooks & Event Triggers
- **External Webhooks**: None.
- **Internal Triggers**: The system relies heavily on asynchronous internal event triggering. The `BybitWSClient` acts as a producer, pushing snapshots into an `asyncio.Queue()`, which triggers the background worker thread to execute ONNX inference instantly without HTTP overhead.

---

## API Endpoint Summary Table

| Endpoint | Method | Purpose | Input / Body | Response |
|---|---|---|---|---|
| `/` | `GET` | Main Dashboard UI | None | HTML |
| `/health` | `GET` | System Diagnostics | None | JSON (Status, Model details) |
| `/predict` | `POST` | Core Inference Engine | `LOBSnapshot` (10x4 matrix) | JSON (Directional Probs, Latency) |
| `/regime` | `POST` | HMM Regime Detection | `RegimeInput` (vol, autocorrel) | JSON (Regime Label, Probabilities) |
