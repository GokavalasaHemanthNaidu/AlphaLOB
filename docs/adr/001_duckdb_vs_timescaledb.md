# ADR 001: Embedded OLAP vs. Dedicated TSDB

## Status
Accepted

## Context
High-Frequency Trading (HFT) platforms require the ingestion of high-velocity Limit Order Book (LOB) snapshots. We needed a storage solution to cache recent market ticks for feature engineering (e.g., rolling VWAP, Order Flow Imbalance) and to serialize tick data for offline PyTorch training. The primary options considered were **TimescaleDB** (a dedicated time-series database extension for PostgreSQL) and **DuckDB** (an embedded, in-process OLAP database).

## Decision
We elected to use **DuckDB** for the AlphaLOB architecture.

## Rationale
1. **Zero Network Latency**: DuckDB runs in-process. Querying rolling windows for feature engineering bypasses the TCP/IP stack entirely, which is critical when attempting to maintain sub-millisecond end-to-end inference latency.
2. **Columnar Vectorization**: LOB data is inherently analytical (calculating means, standard deviations, and diffs across time windows). DuckDB's vectorized query execution engine outperforms row-store caches (like standard SQLite) and rivals TimescaleDB for localized, single-node analytical workloads.
3. **Portability**: DuckDB allows the database to be stored as a single local file (`.db`) or run entirely in-memory (`:memory:`). This dramatically simplifies the MLOps deployment footprint (no Docker-compose required for a heavy Postgres instance) and aligns with the project's goal of being a lightweight, reproducible demo.

## Consequences
- **Pro**: Extremely fast local read/write performance suitable for HFT simulation.
- **Pro**: Frictionless deployment and setup for reviewers.
- **Con**: DuckDB is not designed for distributed, multi-writer concurrency. In a true enterprise exchange environment scaling horizontally across multiple Kubernetes pods, a centralized TimescaleDB or kdb+ instance would be required for global state synchronization.
