# Product Requirements Document (PRD): AlphaLOB

## 1. App Name and Tagline
**App Name:** AlphaLOB
**Tagline:** Low-Latency HFT Signal Engine (Simulation & Inference Framework).

## 2. Problem Statement
In high-frequency cryptocurrency trading, prices shift in milliseconds based on order book dynamics. Identifying complex, microsecond-level patterns of buyer/seller pressure in a Limit Order Book (LOB) is impossible for humans and traditional rule-based algorithms. 

**Who feels the pain:** Quantitative researchers, hedge funds, and algorithmic traders who need an edge to make highly accurate, automated buy/sell decisions before the broader market moves.

## 3. Target User
Target users are algorithmic traders, quantitative analysts, and ML engineers operating in high-frequency trading. They require a highly reliable, low-latency predictive pipeline that ingests live data and outputs actionable signals instantly. They prefer robust, single-container deployments without the headache of managing heavy distributed systems like Kafka.

## 4. Features & Current Status

| Feature | Description | Priority | Current Status |
|---|---|---|---|
| **Bybit WS Ingestion** | Real-time LOB snapshot ingestion via Bybit WebSockets. | Must Have | ✅ Built |
| **ONNX Inference** | Sub-15ms CPU-optimized inference using ONNX Runtime. | Must Have | ✅ Built |
| **3-Horizon Prediction** | Multi-horizon signal prediction (5s, 30s, 5m: UP/DOWN/FLAT). | Must Have | ✅ Built |
| **HMM Regime Detection** | Market regime detection via Hidden Markov Models. | Must Have | ✅ Built |
| **API & Streaming** | REST API for predictions and SSE for live data streaming. | Must Have | ✅ Built |
| **Web Dashboard** | Live web dashboard for real-time visualization of signals. | Must Have | ✅ Built |
| **Embedded DBs** | In-memory metric tracking (DuckDB + MLflow-lite). | Must Have | ✅ Built |
| **Single Container** | Fully containerized system via Docker. | Must Have | ✅ Built |
| **Multi-Symbol Support** | Expand support beyond BTCUSDT to ETH, SOL, BNB. | Nice to Have | 🚧 Planned |
| **RL Execution Layer** | Reinforcement learning execution for dynamic position sizing. | Nice to Have | 🚧 Planned |
| **Tick-Level Data** | Granular tick-level data processing. | Nice to Have | 🚧 Planned |

## 5. Out of Scope (For this version)
- **Trade Execution:** Actual trade execution and automated order routing back to the exchange (`ccxt` / `place_order`).
- **Heavy Infrastructure:** Managing distributed systems and message brokers like Redis, Kafka, or PostgreSQL.
- **Consumer Interfaces:** A mobile application or consumer-facing retail interface.

## 6. User Stories
1. **As an algorithmic trader,** I want to receive real-time predictive signals (UP/DOWN/FLAT) over a continuous stream (SSE) so that my trading bot can place orders milliseconds before the market moves. *(Status: Demoable via `/signals` endpoint)*
2. **As a quant researcher,** I want the system to identify the current market regime (e.g., Volatile vs. Trending) so that I can dynamically adjust my trading strategy's risk rules and confidence thresholds. *(Status: Demoable via HMM model output)*
3. **As an ML/Data engineer,** I want the entire pipeline deployed in a single Docker container so that I can spin up a production environment easily without managing complex message brokers. *(Status: Demoable via `Dockerfile`)*
4. **As an ML operator,** I want the system to automatically monitor feature distributions for data drift so that I am alerted immediately if the model's reliability drops. *(Status: Demoable via PSI/KL-Divergence checks and `/drift` endpoint)*

## 7. Success Metrics

| Metric | Target | Current Status |
|---|---|---|
| **System Latency** | End-to-end inference completing in <15 milliseconds (p99) on CPU. | ✅ Meeting Target |
| **System Reliability** | Uninterrupted API and SSE streaming uptime without dropping LOB snapshots. | ✅ Stable |
| **Predictive Accuracy** | High directional accuracy (>55%) across horizons. | ⚠️ Underperforming (<50% validation accuracy) |
| **Sharpe Ratio** | Consistently strong Sharpe Ratio (>1.5) in walk-forward historical backtesting. | 🚧 Tuning in Progress |
