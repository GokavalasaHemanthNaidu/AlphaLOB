---
title: AlphaLOB
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
---
<div align="center">

<a href="https://github.com/GokavalasaHemanthNaidu/AlphaLOB"><img src="https://img.shields.io/badge/AlphaLOB-HFT%20AI%20Engine-blueviolet?style=for-the-badge&logo=lightning&logoColor=white" height="40"/></a>

# ⚡ AlphaLOB
### Low-Latency Limit Order Book Signal Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnx.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=flat-square&logo=duckdb&logoColor=black)](https://duckdb.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-yellow?style=flat-square)](https://huggingface.co/spaces/hemanthnaidug/AlphaLOB)

**[🚀 Live Demo](https://huggingface.co/spaces/hemanthnaidug/AlphaLOB)** · **[🐛 Issues](https://github.com/GokavalasaHemanthNaidu/AlphaLOB/issues)**

---

*An end-to-end ML system that reads the live heartbeat of a financial market and predicts — in milliseconds — whether the next price move will be UP, DOWN, or FLAT.*

</div>

---

## 📖 Table of Contents

- [🧩 What Is This? (Layman's Guide)](#-what-is-this-laymans-guide)
- [🏗️ System Architecture Diagram](#️-system-architecture-diagram)
- [🔄 Full ML Pipeline](#-full-ml-pipeline)
- [🧠 Model Architecture](#-model-architecture)
- [📊 Feature Engineering](#-feature-engineering)
- [🌊 Market Regime Detection (HMM)](#-market-regime-detection-hmm)
- [⚡ Low-Latency Inference Stack](#-low-latency-inference-stack)
- [🌐 API Reference](#-api-reference)
- [🚀 Quick Start](#-quick-start)
- [📁 Repository Structure](#-repository-structure)
- [🎯 Design Decisions](#-design-decisions)
- [📈 MLOps & Drift Monitoring](#-mlops--drift-monitoring)
- [🛠️ Tech Stack Table](#️-tech-stack-table)
- [🗺️ Roadmap](#️-roadmap)
- [📚 Comprehensive Documentation (Docs Folder)](#-comprehensive-documentation-docs-folder)

---

<a id="-what-is-this-laymans-guide"></a>
## 🧩 What Is This? (Layman's Guide)

> **No finance background? No problem. Read this first.**

Imagine you're watching an eBay auction, but for Bitcoin, and it's happening **millions of times per second**.

At any moment, thousands of buyers say *"I'll pay $30,000"* and thousands of sellers say *"I'll sell for $30,005."* The full list of all these open offers — sorted by price — is called a **Limit Order Book (LOB)**.

```text
                    THE LIMIT ORDER BOOK
    ┌─────────────────────────────────────────────┐
    │         SELL ORDERS (Asks)                  │
    │  $30,010 ████████████████  500 BTC          │
    │  $30,007 ████████          200 BTC          │
    │  $30,005 ██               50  BTC  ← Best   │
    │  ─ ─ ─ ─ ─ ─ SPREAD ─ ─ ─ ─ ─ ─ ─         │
    │  $30,000 ████████████      300 BTC  ← Best  │
    │  $29,998 ████████████████  600 BTC          │
    │  $29,995 ████              100 BTC          │
    │         BUY ORDERS (Bids)                   │
    └─────────────────────────────────────────────┘
               ↑ This entire picture changes
                 thousands of times per second
```

**AlphaLOB watches this order book in real time** and uses a trained AI model to answer:

> *"Based on the last N snapshots of buyer/seller pressure — will the price go UP 📈, DOWN 📉, or SIDEWAYS ➡️ in the next 30 seconds?"*

That prediction — called an **alpha signal** — is the core output that hedge funds and algorithmic trading systems use to execute micro-trades ahead of the market.

---

<a id="️-system-architecture-diagram"></a>
## 🏗️ System Architecture Diagram

The system is split into two clean environments: a **Cloud Training Phase** (where massive data is processed) and a **Production Docker Container** (where latency is everything).

```text
╔══════════════════════════════════════════════════════════════════════════╗
║                     ALPHALOB SYSTEM ARCHITECTURE                         ║
╠═══════════════════════════╦══════════════════════════════════════════════╣
║  ☁️  CLOUD GPU ENVIRONMENT  ║  🖥️  PRODUCTION INFERENCE SERVER             ║
║  (Google Colab / Training) ║  (Docker Container)                          ║
║                            ║                                              ║
║  ┌─────────────────────┐  ║  ┌──────────────────────────────────────┐   ║
║  │  Bybit Historical   │  ║  │  Live Data Sources                   │   ║
║  │  LOB Data (BTCUSDT) │  ║  │  ┌──────────────┐ ┌──────────────┐  │   ║
║  └──────────┬──────────┘  ║  │  │ Bybit WS API │ │ Synthetic    │  │   ║
║             │              ║  │  │  (live feed) │ │ LOB Generator│  │   ║
║             ▼              ║  │  └──────┬───────┘ └──────┬───────┘  │   ║
║  ┌─────────────────────┐  ║  └─────────┼────────────────┼──────────┘   ║
║  │  Feature Eng.       │  ║            └────────┬────────┘              ║
║  │  (Phase 1 Notebook) │  ║                     ▼                       ║
║  └──────────┬──────────┘  ║  ┌──────────────────────────────────────┐   ║
║             │              ║  │  Feature Engineering Worker           │   ║
║             ▼              ║  │  · Mid-price, Spread, Imbalance      │   ║
║  ┌─────────────────────┐  ║  │  · WOFI, Depth Ratio, Volume δ       │   ║
║  │  PyTorch Transformer│  ║  └─────────────────┬────────────────────┘   ║
║  │  Multi-Task Head    │  ║                     │                       ║
║  │  (Phase 2 Notebook) │  ║                     ▼                       ║
║  └──────────┬──────────┘  ║  ┌──────────────────────────────────────┐   ║
║             │              ║  │  asyncio.Queue  (in-process bus)     │   ║
║             ▼              ║  └─────────────────┬────────────────────┘   ║
║  ┌─────────────────────┐  ║                     │                       ║
║  │  Export to ONNX     │──╬──────────────────── ▼                       ║
║  │  lob_transformer    │  ║  ┌──────────────────────────────────────┐   ║
║  │  .onnx              │  ║  │  Inference Worker                    │   ║
║  └─────────────────────┘  ║  │  · ONNX Runtime (CPU-optimized)      │   ║
║                            ║  │  · HMM Regime Classifier             │   ║
║  ┌─────────────────────┐  ║  │  · Drift Monitor (KL-Divergence)     │   ║
║  │  HMM Regime Model   │──╬──└─────────────────┬────────────────────┘   ║
║  │  (hmmlearn)         │  ║                     │                       ║
║  └─────────────────────┘  ║                     ▼                       ║
║                            ║  ┌──────────────────────────────────────┐   ║
║                            ║  │  FastAPI Application                 │   ║
║                            ║  │  · SSE /signals  (streaming)         │   ║
║                            ║  │  · REST /predict (snapshot)          │   ║
║                            ║  └─────────────────┬────────────────────┘   ║
║                            ║                     │                       ║
║                            ║                     ▼                       ║
║                            ║  ┌──────────────────────────────────────┐   ║
║                            ║  │  🌐 Live Web Dashboard               │   ║
║                            ║  │  Real-time terminal visualization    │   ║
║                            ║  └──────────────────────────────────────┘   ║
╚═══════════════════════════╩══════════════════════════════════════════════╝
```

---

<a id="-full-ml-pipeline"></a>
## 🔄 Full ML Pipeline

The ML lifecycle is divided into a strict 4-phase sequence.

### Phase 1: Data & Features
- Pulls Level-2 snapshot order book data (Bids/Asks 1-5) via Bybit REST API.
- Computes engineered indicators: WOFI, Imbalance, Spread Compression.
- Maps continuous price changes to strict ternary targets (+1, 0, -1) over 5s/30s/5m horizons.
- Archives features into an embedded columnar `alphalob.duckdb` database.

### Phase 2: Model Training
- Trains a custom PyTorch Transformer Encoder from scratch.
- Utilizes Multi-Task Learning (predicting direction alongside spread volatility).
- Traces and exports the dynamic PyTorch graph to an optimized `.onnx` graph.

### Phase 3: Backtesting Engine
- Reads historical hold-out test sets from DuckDB.
- Executes walk-forward simulation crossing the bid-ask spread.
- Generates hard trading metrics: Sharpe Ratio, Max Drawdown, and Win Rate.

### Phase 4: Production Inference
- Loads the `.onnx` file inside the standalone Docker container.
- Connects to Bybit WebSockets, piping live data through `asyncio.Queue`.
- Pushes continuous directional probabilities to the UI via Server-Sent Events (SSE).

---

<a id="-model-architecture"></a>
## 🧠 Model Architecture

The core predictor is a **Custom PyTorch Transformer**, built to process sequential financial feature vectors rather than NLP tokens.

| Component | Detail | Why |
|---|---|---|
| **Input Shape** | `[Batch × Seq_Len × N_Features]` | Matches sliding-window LOB snapshots. |
| **Embedding** | Linear projection (`d_model = 64`) | Maps continuous features to attention space. |
| **Positional Encoding** | Sinusoidal | Preserves the rigid chronological tick order. |
| **Encoder Layers** | Stacked Multi-Head Attention | Discovers temporal order flow sequences. |
| **Pooling** | CLS-token reduction | Compresses sequence state into a 1D vector. |

### The Multi-Task Output Head

Instead of just guessing "UP" or "DOWN", the model is forced to simultaneously solve three auxiliary problems. This creates stronger gradient signals.

```text
                    ┌────────────────────────────────────────────┐
                    │           CLS Token Representation         │
                    │          (compressed market state)         │
                    └──────────────┬─────────────────┬───────────┘
                                   │                 │
              ┌────────────────────▼─┐  ┌────────────▼──────────────┐
              │     Head 1           │  │       Head 2 & 3          │
              │  Direction Signal    │  │  Auxiliary Signals        │
              │                      │  │                           │
              │  3-class classifier  │  │ ▪ Spread Compression      │
              │  for each horizon:   │  │   (binary: tightening?)   │
              │  ▪ 5 second          │  │                           │
              │  ▪ 30 second   ← ★  │  │ ▪ Volume Imbalance        │
              │  ▪ 5 minute          │  │   (regression: buy/sell   │
              │                      │  │    pressure ratio)        │
              └──────────────────────┘  └───────────────────────────┘
```
**Why multi-task?** Predicting order book spread compression forces the internal layers to understand microstructural liquidity mechanics rather than blindly pattern-matching price jumps.

---

<a id="-feature-engineering"></a>
## 📊 Feature Engineering

Neural networks struggle with raw order book shapes. The system transforms the raw state into mathematically stationary features:

```text
RAW LOB SNAPSHOT
 bid_price_1, bid_size_1, ... bid_size_5
 ask_price_1, ask_size_1, ... ask_size_5
                │
                ▼
      FEATURE TRANSFORMATION
  ┌─────────────────────────────────────────┐
  │ • mid_price       = (best_bid+best_ask)/2
  │ • spread          = ask_1 - bid_1       │
  │ • order_imbalance = (bidV - askV)/(tot) │
  │ • WOFI            = depth-weighted imbalance
  │ • depth_ratio     = sum(bids)/sum(asks) │
  │ • rolling_vol_delta = change in volume  │
  └─────────────────────────────────────────┘
                │
                ▼
          Feature Vector
```

---

<a id="-market-regime-detection-hmm"></a>
## 🌊 Market Regime Detection (HMM)

An **HMM (Hidden Markov Model)** is a probabilistic classifier that figures out the unobservable state of a system based on visible observations.

We utilize a 3-state Gaussian HMM (via `hmmlearn`) to track the current meta-state of the order book:
1. **TRENDING (State 0):** Low spread, persistent directional order flow.
2. **MEAN-REVERTING (State 1):** Tight spread, balanced liquidity.
3. **VOLATILE (State 2):** Wide spread, erratic sweeps, high variance.

**Why?** A "BUY" signal from the Transformer in a *Trending* regime carries high confidence. The same signal in a *Volatile* regime represents noise and is filtered out.

---

<a id="-low-latency-inference-stack"></a>
## ⚡ Low-Latency Inference Stack

The core constraint of the project is ensuring predictions return in **<15ms on CPU**.

| Trait | PyTorch (Training) | ONNX Runtime (Production) |
|---|---|---|
| Footprint | ~2 GB (with CUDA) | ~50 MB |
| Graph | Dynamic, Autograd | Static, heavily optimized |
| Latency | ~50ms (CPU) | **1-5ms (CPU)** |
| Hardware | Requires GPU | Standard Cloud CPU |

### Concurrency Pattern

To ensure ONNX inference doesn't block the FastAPI HTTP event loop, the backend strictly uses an asynchronous handoff pattern:

```text
LOB Snapshot ──▶ [Feature Worker / asyncio task]
                         │
                         ▼
             [asyncio.Queue (zero-copy)]
                         │
                         ▼
             [Inference Worker / ThreadPoolExecutor]
             ONNX Session.run() executes here
                         │
                         ▼
             [FastAPI SSE Event Loop] ──▶ User Dashboard
```
**Why this matters:** This ensures the web server can concurrently serve thousands of HTTP connections without stuttering while the machine learning model monopolizes the CPU thread.

---

<a id="-api-reference"></a>
## 🌐 API Reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web Dashboard HTML/JS |
| `GET` | `/signals` | SSE continuous stream of predictions |
| `POST` | `/predict` | Immediate one-shot JSON prediction |
| `GET` | `/health` | Diagnostics and model memory statuses |

### `/predict` REST Example
```json
// POST /predict body
{
  "bid_prices": [30000, 29998, 29995, 29990, 29985],
  "bid_sizes":  [300,   600,   100,   450,   200],
  "ask_prices": [30005, 30007, 30010, 30015, 30020],
  "ask_sizes":  [50,    200,   500,   300,   150]
}

// Response
{
  "signal_30s": "UP",
  "confidence": 0.82,
  "regime": "TRENDING",
  "latency_ms": 2.1
}
```

### `/signals` SSE Snippet
```javascript
const source = new EventSource('/signals');
source.onmessage = (e) => {
    let data = JSON.parse(e.data);
    console.log(`Live Signal: ${data.signal_30s}`);
};
```

---

<a id="-quick-start"></a>
## 🚀 Quick Start

Run the entire system completely locally with synthetic mock data (no Bybit API keys needed).

```bash
git clone https://github.com/GokavalasaHemanthNaidu/AlphaLOB.git
cd AlphaLOB
pip install -r requirements.txt
python -m uvicorn src.api.main:app --port 8000 --reload
# Open http://localhost:8000
```

---

<a id="-repository-structure"></a>
## 📁 Repository Structure

```text
AlphaLOB/
├── README.md                      ← Entry point
├── docs/                          ← Comprehensive technical documentation
│   ├── PRD.md
│   ├── TRD.md
│   └── ... 
├── notebooks/colab/
│   ├── Phase1_DataPipeline.ipynb  ← Eng features
│   ├── Phase2_ModelTraining.ipynb ← Trains PyTorch model
│   └── Phase3_Backtesting.ipynb   ← Walk-forward simulator
├── src/
│   ├── api/main.py                ← FastAPI + asyncio loop
│   ├── workers/                   ← Inference and feature tasks
│   ├── domain/inference.py        ← ONNX/hmmlearn wrappers
│   ├── data/bybit_ws.py           ← Websocket client
│   └── infrastructure/duckdb.py   ← DB layer schemas
├── models/weights/
│   ├── lobster_transformer.onnx   ← Production static graph
│   └── regime_hmm.bin             ← Trained HMM model
├── tests/                         ← Pytest suites
├── Dockerfile                     ← Isolated container definition
└── requirements.txt               ← Strict pinned dependencies
```

---

<a id="-design-decisions"></a>
## 🎯 Design Decisions

*Decisions targeting high-performance financial systems:*

- **`asyncio.Queue` over Kafka/Redis:** Kafka implies immense network overhead and external container orchestration. By utilizing an in-memory Python `asyncio.Queue`, the system achieves zero-copy, sub-microsecond handoffs within a single process.
- **ONNX over TorchServe:** TorchServe brings heavy Java dependencies, massive CUDA binaries, and 50ms+ latency. Exporting a static graph to `.onnx` shrinks the container payload to <50MB and yields ~2ms CPU inference.
- **DuckDB over PostgreSQL:** Since financial tick data is purely time-series and append-only, DuckDB's columnar analytics engine queries historical ranges 10-100x faster than PostgreSQL’s row-store, without requiring a background daemon.
- **Single Container Deployment:** Because of the tight dependencies above, the entire end-to-end framework deploys trivially to free-tier CPU instances (like Hugging Face Spaces) with zero DevOps complexity.

---

<a id="-mlops--drift-monitoring"></a>
## 📈 MLOps & Drift Monitoring

To prevent "silent failures" where the market changes behavior but the model keeps trading, AlphaLOB utilizes a lightweight monitoring daemon.

```text
[Incoming LOB Feature Space] ────── KL-Divergence ────── [Baseline Train Space]
                                        │
                                        ▼
                             Drift Score > Threshold?
                            /                        \
                    [✅ Normal]                 [⚠️ ALERT SQL]
```

- **SQLite Tracker:** Used as an MLflow-lite instance. Logs `accuracy`, `latency_p99`, queue depth, and regime transition probabilities.
- **DuckDB Archive:** Houses massive numerical historical feature snapshots for deep post-mortem analysis.

---

<a id="️-tech-stack-table"></a>
## 🛠️ Tech Stack Table

| Layer | Technology | Why We Used It |
|---|---|---|
| **ML Framework** | PyTorch | Deep customization of Transformer Self-Attention |
| **Edge Inference** | ONNX Runtime | High-speed, GPU-independent C++ engine execution |
| **Unit tests** | ✅ 100% Core coverage via Pytest | High reliability |
| **CI/CD** | ✅ Fully implemented (Pytest + Security) | Automated deployment |
| **Regime Detection** | `hmmlearn` | Unsupervised Gaussian HMM state clustering |
| **API Server** | FastAPI | Async-first HTTP routing with built-in Pydantic |
| **Real-Time Data** | Server-Sent Events | Unidirectional push without WebSocket bloat |
| **Concurrency** | `asyncio.Queue` | Non-blocking inter-worker memory transfers |
| **Database** | DuckDB | Blistering fast embedded columnar storage |
| **Tracking** | SQLite | Serverless metric tracking |
| **Container** | Docker | 100% reproducible deployment context |

---

<a id="️-roadmap"></a>
## 🗺️ Roadmap

- ✅ **Phases 1-5 (Complete):** Data ingestion, feature engineering, Transformer training, ONNX export, FastAPI inference, Docker deployment.
- 🔜 **Phase 6:** Multi-symbol support (ETH, SOL, BNB simultaneously).
- 🔜 **Phase 7:** Reinforcement Learning execution layer (sizing based on signal).
- 🔜 **Phase 8:** Tick-level trade ingestion (upgrading from snapshot frequency).
- 🔜 **Phase 9:** Federated multi-exchange signal fusion.

---

<a id="-comprehensive-documentation-docs-folder"></a>
## 📚 Comprehensive Documentation (Docs Folder)

For an extraordinarily deep dive into how this system was architected, designed, and structured, review the core planning files located in the [`/docs`](docs/) directory:

| Document | Purpose | Link |
|---|---|---|
| **PRD** | Product Requirements, Use Cases & Personas | [`docs/PRD.md`](docs/PRD.md) |
| **TRD** | Technical Constraints & Stack Specifications | [`docs/TRD.md`](docs/TRD.md) |
| **App Flow** | Web Dashboard Navigation & Client State | [`docs/AppFlow.md`](docs/AppFlow.md) |
| **UI/UX Brief** | Terminal Theme Constants & WCAG Checks | [`docs/UIUXBrief.md`](docs/UIUXBrief.md) |
| **Backend Schema** | DuckDB & SQLite Table Mappings | [`docs/BackendSchema.md`](docs/BackendSchema.md) |
| **Implementation Plan** | Ordered 7-Phase Build Roadmap | [`docs/ImplementationPlan.md`](docs/ImplementationPlan.md) |

---

<div align="center">

**Built by [Gokavalasa Hemanth Naidu](https://github.com/GokavalasaHemanthNaidu)**

*Dual Degree (B.Tech + M.Tech) — Mathematics & Computing / CSE*

[![GitHub](https://img.shields.io/badge/GitHub-GokavalasaHemanthNaidu-181717?style=flat-square&logo=github)](https://github.com/GokavalasaHemanthNaidu)
[![HuggingFace](https://img.shields.io/badge/🤗-hemanthnaidug-yellow?style=flat-square)](https://huggingface.co/hemanthnaidug)

⭐ **Star this repo if you found it useful!**

</div>
