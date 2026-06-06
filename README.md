<div align="center">

<a href="https://github.com/GokavalasaHemanthNaidu/AlphaLOB"><img src="https://img.shields.io/badge/AlphaLOB-HFT%20AI%20Engine-blueviolet?style=for-the-badge&logo=lightning&logoColor=white" height="40"/></a>

# ⚡ AlphaLOB
### Real-Time High-Frequency Trading AI — Limit Order Book Signal Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnx.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=flat-square&logo=duckdb&logoColor=black)](https://duckdb.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-yellow?style=flat-square)](https://huggingface.co/spaces/hemanthnaidug/AlphaLOB)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

**[🚀 Live Demo](https://huggingface.co/spaces/hemanthnaidug/AlphaLOB)** · **[🐛 Issues](https://github.com/GokavalasaHemanthNaidu/AlphaLOB/issues)**

---

*An end-to-end ML system that reads the live heartbeat of a financial market and predicts — in milliseconds — whether the next price move will be UP, DOWN, or FLAT.*

</div>

---

## 📖 Table of Contents

- [🧩 What Is This? (Layman's Guide)](#-what-is-this-laymans-guide)
- [🏗️ System Architecture](#️-system-architecture)
- [🔄 Full ML Pipeline](#-full-ml-pipeline)
- [🧠 Model Architecture](#-model-architecture)
- [📊 Feature Engineering](#-feature-engineering)
- [🌊 Market Regime Detection (HMM)](#-market-regime-detection-hmm)
- [⚡ Low-Latency Inference Stack](#-low-latency-inference-stack)
- [🌐 API Reference](#-api-reference)
- [📁 Repository Structure](#-repository-structure)
- [🚀 Quick Start](#-quick-start)
- [🐳 Docker Deployment](#-docker-deployment)
- [📈 MLOps & Drift Monitoring](#-mlops--drift-monitoring)
- [🛠️ Tech Stack](#️-tech-stack)
- [🎯 Design Decisions](#-design-decisions)
- [🗺️ Roadmap](#️-roadmap)
- [🧯 VS Code Chat Model Error](#-vs-code-chat-model-error)

---

## 🧩 What Is This? (Layman's Guide)

> **No finance background? No problem. Read this first.**

Imagine you're watching an eBay auction, but for Bitcoin, and it's happening **millions of times per second**.

At any moment, thousands of buyers say *"I'll pay $30,000"* and thousands of sellers say *"I'll sell for $30,005."* The full list of all these open offers — sorted by price — is called a **Limit Order Book (LOB)**.

```
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

That prediction — called an **alpha signal** — is the core output that hedge funds and trading algorithms use to make buy/sell decisions.

---

## 🏗️ System Architecture

The system is split into two clean environments: a **Cloud Training Phase** and a **Local/Deployed Inference Phase**.

```
╔══════════════════════════════════════════════════════════════════════════╗
║                     ALPHALOB SYSTEM ARCHITECTURE                        ║
╠═══════════════════════════╦══════════════════════════════════════════════╣
║  ☁️  CLOUD GPU ENVIRONMENT  ║  🖥️  PRODUCTION INFERENCE SERVER             ║
║  (Google Colab / Training) ║  (Docker · HuggingFace Spaces)              ║
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
║                            ║  │  · /backtest, /health, /docs         │   ║
║                            ║  └─────────────────┬────────────────────┘   ║
║                            ║                     │                       ║
║                            ║                     ▼                       ║
║                            ║  ┌──────────────────────────────────────┐   ║
║                            ║  │  🌐 Live Web Dashboard               │   ║
║                            ║  │  Real-time signal stream (SSE)       │   ║
║                            ║  └──────────────────────────────────────┘   ║
╚═══════════════════════════╩══════════════════════════════════════════════╝
```

---

## 🔄 Full ML Pipeline

The pipeline is executed across **4 sequential Colab notebooks** in training, then compressed into a single Docker container for production.

```
PHASE 1: DATA & FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ┌─────────────────┐     ┌──────────────────────┐
 │  Bybit REST API │────▶│  Raw LOB Snapshots   │
 │  BTCUSDT Perp.  │     │  (bid/ask levels 1-5)│
 └─────────────────┘     └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  Feature Engineering │
                          │  ▪ Mid-price         │
                          │  ▪ Bid-Ask Spread    │
                          │  ▪ Order Imbalance   │
                          │  ▪ WOFI score        │
                          │  ▪ Depth ratio       │
                          │  ▪ Rolling δ volume  │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  Label Generation    │
                          │  ▪ +1  (UP   >0.02%) │
                          │  ▪  0  (FLAT ±0.02%) │
                          │  ▪ -1  (DOWN <0.02%) │
                          │  Horizons: 5s/30s/5m │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  DuckDB Storage      │
                          │  alphalob.duckdb     │
                          └──────────────────────┘

PHASE 2: MODEL TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ┌─────────────────────────────────────────────────┐
 │           LOB Transformer (PyTorch)             │
 │                                                 │
 │  Input: [Batch × Seq_Len × Features]            │
 │                      │                          │
 │            ┌─────────▼──────────┐               │
 │            │  Linear Embedding  │               │
 │            │  d_model = 64      │               │
 │            └─────────┬──────────┘               │
 │                      │                          │
 │            ┌─────────▼──────────┐               │
 │            │ Positional Encoding│               │
 │            └─────────┬──────────┘               │
 │                      │                          │
 │            ┌─────────▼──────────┐               │
 │            │  Transformer Enc.  │  × N layers   │
 │            │  Multi-Head Attn   │               │
 │            │  FFN + Layer Norm  │               │
 │            └─────────┬──────────┘               │
 │                      │                          │
 │            ┌─────────▼──────────┐               │
 │            │   CLS Token Pool   │               │
 │            └──┬──────┬──────┬───┘               │
 │               │      │      │                   │
 │         ┌─────▼┐  ┌──▼──┐  ┌▼─────┐            │
 │         │Head 1│  │Head2│  │Head 3│            │
 │         │Dir.  │  │Sprd │  │Vol.  │            │
 │         │5s/30s│  │Comp.│  │Imbal.│            │
 │         │/5min │  │     │  │      │            │
 │         └──────┘  └─────┘  └──────┘            │
 └─────────────────────────────────────────────────┘
              │
              ▼
 ┌─────────────────────────────────────────────────┐
 │           Export → ONNX Graph                   │
 │   torch.onnx.export(model, ...)                 │
 │   lob_transformer.onnx  (~few MB)               │
 └─────────────────────────────────────────────────┘

PHASE 3: BACKTESTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Historical Data ──▶ Backtesting Engine
                         ▪ Walk-forward simulation
                         ▪ Sharpe ratio, Max drawdown
                         ▪ Signal accuracy per horizon

PHASE 4: PRODUCTION INFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 (See System Architecture above)
```

---

## 🧠 Model Architecture

### LOB Transformer

The core prediction model is a **custom Transformer Encoder** built from scratch in PyTorch — not a pre-trained NLP model, but one specifically designed for time-series financial data.

| Component | Detail |
|---|---|
| **Input Shape** | `[Batch × Sequence_Length × N_Features]` |
| **Embedding** | Linear projection → `d_model = 64` |
| **Positional Encoding** | Sinusoidal (captures order of LOB snapshots over time) |
| **Encoder Layers** | Stacked `TransformerEncoderLayer` blocks |
| **Attention** | Multi-Head Self-Attention (each head learns different market patterns) |
| **Pooling** | CLS-token pooling (one summary vector from the full sequence) |
| **Output Heads** | 3 parallel task heads (see below) |

#### Multi-Task Output Heads

```
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
              │  Labels:             │  │                           │
              │   +1 = price UP      │  │  These help the model     │
              │    0 = price FLAT    │  │  learn richer market      │
              │   -1 = price DOWN    │  │  microstructure signals   │
              └──────────────────────┘  └───────────────────────────┘
```

> **Why multi-task?** Training on related tasks simultaneously (direction + spread + volume) forces the model to learn deeper representations of market microstructure rather than just pattern-matching price labels. This is a technique used by many professional quant shops.

---

## 📊 Feature Engineering

Raw LOB data is transformed into **engineered numerical features** before being fed to the model. Each feature captures a specific aspect of market microstructure:

```
RAW LOB SNAPSHOT
 bid_price_1, bid_size_1, bid_price_2, bid_size_2, ... bid_price_5, bid_size_5
 ask_price_1, ask_size_1, ask_price_2, ask_size_2, ... ask_price_5, ask_size_5
                                  │
                                  ▼
          ┌───────────────────────────────────────────────┐
          │            FEATURE ENGINEERING                 │
          │                                               │
          │  ┌──────────────────────────────────────────┐ │
          │  │  PRICE FEATURES                          │ │
          │  │  • mid_price   = (best_bid + best_ask)/2 │ │
          │  │  • spread      = best_ask - best_bid     │ │
          │  │  • spread_pct  = spread / mid_price      │ │
          │  └──────────────────────────────────────────┘ │
          │                                               │
          │  ┌──────────────────────────────────────────┐ │
          │  │  ORDER FLOW FEATURES                     │ │
          │  │  • order_imbalance = (bid_vol - ask_vol) │ │
          │  │                    / (bid_vol + ask_vol) │ │
          │  │    → +1 = all buyers, -1 = all sellers   │ │
          │  │                                          │ │
          │  │  • WOFI (Weighted Order Flow Imbalance)  │ │
          │  │    → Imbalance weighted by proximity     │ │
          │  │      to best bid/ask (levels 1-5)        │ │
          │  └──────────────────────────────────────────┘ │
          │                                               │
          │  ┌──────────────────────────────────────────┐ │
          │  │  DEPTH FEATURES                          │ │
          │  │  • depth_ratio  = total_bid / total_ask  │ │
          │  │  • depth_delta  = change in total depth  │ │
          │  │  • level_slopes = price gaps between     │ │
          │  │                   consecutive LOB levels │ │
          │  └──────────────────────────────────────────┘ │
          │                                               │
          │  ┌──────────────────────────────────────────┐ │
          │  │  TEMPORAL FEATURES                       │ │
          │  │  • rolling_vol_delta = Δ in traded vol   │ │
          │  │  • price_momentum    = mid_price change  │ │
          │  │    over last N ticks                     │ │
          │  └──────────────────────────────────────────┘ │
          └───────────────────────────────────────────────┘
                                  │
                                  ▼
             Feature Vector: [N_features] per snapshot
             Sliding Window:  last K snapshots → [K × N] tensor
```

---

## 🌊 Market Regime Detection (HMM)

In addition to the Transformer's price direction signal, AlphaLOB runs a parallel **Hidden Markov Model (HMM)** to classify the current market *regime*. This is important because the same buy/sell pressure means very different things in different market conditions.

```
                      MARKET REGIME CLASSIFIER
                      ━━━━━━━━━━━━━━━━━━━━━━━━

  Feature Stream ──▶ HMM (hmmlearn GaussianHMM)
                          │
                          ▼
              ┌───────────────────────────┐
              │  3 Hidden Market States   │
              │                           │
              │  State 0: 📈 TRENDING     │  Low spread, directional
              │           momentum-driven │  flow, one-sided book
              │                           │
              │  State 1: ↔️ MEAN-REVERTING│  Tight spread, balanced
              │           range-bound     │  book, high liquidity
              │                           │
              │  State 2: ⚡ VOLATILE      │  Wide spread, erratic
              │           noisy / news    │  order flow, thin book
              └───────────────────────────┘
                          │
                          ▼
              Combined Signal = Transformer Prediction
                              + HMM Regime Context

  Example: "BUY signal in VOLATILE regime" → lower confidence
           "BUY signal in TRENDING regime" → higher confidence
```

---

## ⚡ Low-Latency Inference Stack

One of the most critical engineering decisions in AlphaLOB is the path from a large PyTorch model to sub-millisecond CPU inference:

```
TRAINING TIME                           PRODUCTION TIME
━━━━━━━━━━━━━━━                         ━━━━━━━━━━━━━━━━

PyTorch Model                           ONNX Runtime
  ▪ Full autograd graph                   ▪ No Python overhead
  ▪ ~150MB with optimizer                 ▪ Static compute graph
  ▪ GPU dependent                         ▪ CPU-optimized kernels
  ▪ ~50ms inference                       ▪ ~1-5ms inference
                                          ▪ No GPU required
        │                                       ▲
        │   torch.onnx.export()                 │
        └──────────────────────────────────────▶│
                                          lob_transformer.onnx

INFERENCE PIPELINE (per tick)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 LOB Snapshot
     │
     ▼  [Feature Worker - asyncio task]
 Feature Vector
     │
     ▼  [asyncio.Queue - zero-copy handoff]
 Queue Buffer
     │
     ▼  [Inference Worker - ThreadPoolExecutor]
 ONNX Runtime Session
     │  InferenceSession.run()
     │  (CPU: OpenMP parallelism)
     ▼
 Prediction Dict
     │
     ▼  [FastAPI SSE Router]
 data: {"signal": "UP", "confidence": 0.82, ...}
     │
     ▼
 Web Dashboard (EventSource)
```

> **Why asyncio + ThreadPool?** ONNX inference is CPU-bound (not I/O-bound), so it blocks Python's event loop. We offload it to a ThreadPoolExecutor, keeping FastAPI fully responsive to incoming HTTP connections while the model computes — a critical pattern for production ML APIs.

---

## 🌐 API Reference

The FastAPI server exposes four categories of endpoints, all served from a single Docker container:

```
BASE URL: https://hemanthnaidug-alphalob.hf.space
          (or http://localhost:8000 locally)

┌─────────────────────────────────────────────────────────────────┐
│                       API ENDPOINTS                             │
├──────────┬──────────────────────┬──────────────────────────────┤
│ Method   │ Endpoint             │ Description                  │
├──────────┼──────────────────────┼──────────────────────────────┤
│ GET      │ /                    │ Live dashboard UI            │
│ GET      │ /signals             │ SSE stream of live signals   │
│ POST     │ /predict             │ One-shot snapshot inference  │
│ POST     │ /backtest            │ Launch historical simulation │
│ GET      │ /health              │ Model + queue metrics        │
│ GET      │ /docs                │ Auto-generated Swagger UI    │
└──────────┴──────────────────────┴──────────────────────────────┘
```

### `/predict` — REST Snapshot Inference

Send a raw LOB snapshot as JSON; receive an instant prediction.

```json
// REQUEST
POST /predict
Content-Type: application/json

{
  "bid_prices": [30000, 29998, 29995, 29990, 29985],
  "bid_sizes":  [300,   600,   100,   450,   200],
  "ask_prices": [30005, 30007, 30010, 30015, 30020],
  "ask_sizes":  [50,    200,   500,   300,   150],
  "timestamp":  1748700000000
}

// RESPONSE
{
  "signal_5s":     "UP",
  "signal_30s":    "UP",
  "signal_5min":   "FLAT",
  "confidence":    0.82,
  "regime":        "TRENDING",
  "spread_compression": true,
  "volume_imbalance":   0.43,
  "latency_ms":    2.3
}
```

### `/signals` — Server-Sent Events Stream

Connect once; receive a continuous stream of live predictions:

```javascript
// Browser / Client
const source = new EventSource('/signals');
source.onmessage = (event) => {
    const prediction = JSON.parse(event.data);
    console.log(prediction.signal_30s); // "UP", "DOWN", "FLAT"
};

// Server emits every tick:
// data: {"signal_30s": "UP", "confidence": 0.79, "regime": "TRENDING", ...}
// data: {"signal_30s": "FLAT", "confidence": 0.61, "regime": "MEAN-REVERTING", ...}
```

---

## 📁 Repository Structure

```
AlphaLOB/
│
├── 📓 notebooks/colab/
│   ├── Phase1_DataPipeline.ipynb      ← Data ingestion + feature engineering
│   ├── Phase2_ModelTraining.ipynb     ← Transformer training + ONNX export
│   ├── Phase3_Backtesting.ipynb       ← Walk-forward historical simulation
│   └── Phase4_LiveInference.ipynb     ← End-to-end live pipeline demo
│
├── 🐍 src/
│   ├── api/
│   │   ├── main.py                    ← FastAPI app + asyncio event loop
│   │   ├── schemas.py                 ← Pydantic type-safe models
│   │   └── routes/
│   │       ├── signals.py             ← SSE live streaming endpoint
│   │       ├── predict.py             ← REST one-off inference endpoint
│   │       ├── backtest.py            ← Historical simulation endpoint
│   │       └── model_health.py        ← Health checks + queue metrics
│   │
│   ├── workers/
│   │   ├── feature_worker.py          ← Real-time feature computation
│   │   └── inference_worker.py        ← CPU-bound ONNX threadpool worker
│   │
│   ├── domain/
│   │   ├── features.py                ← Pure LOB feature engineering logic
│   │   ├── inference.py               ← ONNX Runtime session wrapper
│   │   └── models/
│   │       ├── lob_transformer.py     ← PyTorch model definition
│   │       └── regime_hmm.py          ← HMM regime classifier
│   │   └── backtesting/
│   │       ├── engine.py              ← Walk-forward backtest engine
│   │       └── metrics.py             ← Sharpe, drawdown, accuracy calc
│   │
│   ├── data/
│   │   ├── bybit_ws.py                ← Live Bybit WebSocket ingestion
│   │   └── synthetic_lob.py           ← Offline synthetic data generator
│   │
│   └── infrastructure/
│       ├── duckdb_client.py           ← Embedded analytics DB client
│       └── mlflow_sqlite.py           ← Lightweight metric tracking
│
├── 🤖 models/
│   └── lob_transformer.onnx           ← Compiled model weights (production)
│
├── 📋 docs/adr/                       ← Architecture Decision Records
├── 🧪 tests/                          ← Unit & integration tests
├── 📜 scripts/                        ← Utility scripts
├── 🐳 Dockerfile                      ← Single-container build
├── 📦 requirements.txt                ← Pinned Python dependencies
├── ⚙️  render.yaml                     ← Render.com deployment config
└── 🦆 alphalob.duckdb                 ← Embedded analytics database
```

---

## 🚀 Quick Start

### Option 1: Use the Live Demo (No Setup Required)

**[→ Open AlphaLOB on HuggingFace Spaces](https://huggingface.co/spaces/hemanthnaidug/AlphaLOB)**

The live deployment runs 24/7 with a **Synthetic LOB Generator** — so you can see real predictions without needing any API keys.

---

### Option 2: Run Locally

#### Prerequisites

- Python 3.11+
- Git

#### Steps

```bash
# 1. Clone the repository
git clone https://github.com/GokavalasaHemanthNaidu/AlphaLOB.git
cd AlphaLOB

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the inference server
#    (Uses built-in Synthetic LOB — no API keys needed!)
python -m uvicorn src.api.main:app --port 8000 --reload

# 5. Open the live dashboard
#    Navigate to: http://localhost:8000
```

You'll immediately see Server-Sent Events streaming live predictions to the dashboard — no external connections required.

#### Running Notebooks (Training)

To train the model from scratch, run the Colab notebooks in order:

```
1. notebooks/colab/Phase1_DataPipeline.ipynb
   → Fetches historical Bybit data, engineers features, saves to DuckDB

2. notebooks/colab/Phase2_ModelTraining.ipynb
   → Trains the LOB Transformer, exports lob_transformer.onnx

3. notebooks/colab/Phase3_Backtesting.ipynb
   → Walk-forward backtesting on held-out historical data

4. notebooks/colab/Phase4_LiveInference.ipynb
   → Full end-to-end demo of the live inference pipeline
```

---

## 🐳 Docker Deployment

The entire pipeline — API server, feature worker, inference worker — runs in a **single Docker container**.

```bash
# Build
docker build -t alphalob:latest .

# Run
docker run -p 8000:7860 alphalob:latest

# Visit: http://localhost:8000
```

**Why single container?** See [Design Decisions](#-design-decisions) below.

---

## 📈 MLOps & Drift Monitoring

AlphaLOB includes a lightweight production monitoring system without heavy infrastructure:

```
DRIFT MONITORING PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━
 Incoming Feature Vector
         │
         ▼
 KL-Divergence Calculator
 (compare current distribution vs training baseline)
         │
         ├── KL < threshold → ✅ Normal — model reliable
         │
         └── KL > threshold → ⚠️ Drift Detected
                                  → Log alert to SQLite
                                  → Flag predictions with low_confidence

METRIC STORAGE
━━━━━━━━━━━━━━
 SQLite (mlflow_sqlite.py)
  ▪ Prediction accuracy per horizon
  ▪ Feature distribution stats
  ▪ Inference latency percentiles (p50, p95, p99)
  ▪ Queue depth metrics
  ▪ Drift alert history

DuckDB (alphalob.duckdb)
  ▪ Historical feature snapshots (for replay/analysis)
  ▪ Backtest results archive
  ▪ Fast columnar queries for dashboard analytics
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **ML Framework** | PyTorch | Industry standard for custom architecture research |
| **Production Inference** | ONNX Runtime | 10-50x faster than PyTorch for CPU inference |
| **Market Regime** | hmmlearn (GaussianHMM) | Probabilistic unsupervised regime detection |
| **API Server** | FastAPI + Uvicorn | Async-first, auto Swagger docs, production ready |
| **Real-Time Streaming** | Server-Sent Events (SSE) | Lightweight push protocol; no WebSocket complexity |
| **Async Concurrency** | asyncio + ThreadPoolExecutor | Non-blocking I/O with CPU offloading |
| **Analytics DB** | DuckDB | Embedded columnar DB; fast aggregations, no server |
| **Metric Tracking** | SQLite (custom MLflow-lite) | Zero-dependency experiment tracking |
| **Data Ingestion** | Bybit WebSocket API | Sub-second LOB updates from crypto exchange |
| **Data Validation** | Pydantic v2 | Type-safe schemas at API boundary |
| **Containerization** | Docker | Single-container reproducible deployment |
| **Cloud Hosting** | HuggingFace Spaces | Free GPU/CPU hosting with Docker SDK |
| **Notebooks** | Google Colab | Free GPU for training |
| **Data Engineering** | Pandas + NumPy | Feature matrix construction |

---

## 🎯 Design Decisions

### Why Single Docker Container? (No Kafka, No Redis)

Many production trading systems use distributed message buses like **Apache Kafka** or **Redis Streams** to connect pipeline components. AlphaLOB deliberately replaces these with Python's built-in `asyncio.Queue`:

```
Traditional HFT Architecture:
  Feature Worker ──▶ [Kafka Topic] ──▶ Inference Worker ──▶ [Redis] ──▶ API
  ↑ Heavy, requires separate containers, network overhead, ops complexity

AlphaLOB Architecture:
  Feature Worker ──▶ [asyncio.Queue] ──▶ Inference Worker ──▶ [Queue] ──▶ API
  ↑ Zero-copy in-process handoff, sub-microsecond, single container
```

**Trade-off**: This collapses the entire pipeline into one process, which is perfectly suitable for a single-symbol, single-exchange system running on a single machine — which is exactly this use case. Kafka would be warranted for multi-exchange, multi-symbol, multi-consumer architectures.

### Why ONNX over PyTorch Serving (TorchServe)?

| Concern | PyTorch / TorchServe | ONNX Runtime |
|---|---|---|
| Container size | ~2GB (CUDA drivers) | ~50MB |
| CPU inference | ~50ms | ~1-5ms |
| External deps | Needs GPU for speed | Runs on any CPU |
| Free tier hosting | ❌ Impractical | ✅ Perfect fit |

### Why DuckDB over PostgreSQL?

DuckDB is an **embedded analytical database** — it runs inside the Python process with no server. For read-heavy analytics (querying millions of historical LOB rows for backtesting), DuckDB's columnar engine is **10-100x faster** than row-oriented PostgreSQL, and requires zero infrastructure.

---

## 🗺️ Roadmap

```
✅ Phase 1  Data Pipeline + Feature Engineering (Complete)
✅ Phase 2  PyTorch Transformer + ONNX Export (Complete)
✅ Phase 3  Backtesting Engine (Complete)
✅ Phase 4  Live FastAPI + SSE Dashboard (Complete)
✅ Phase 5  Docker + HuggingFace Deployment (Complete)

🔜 Phase 6  Multi-symbol support (ETH, SOL, BNB)
🔜 Phase 7  Reinforcement Learning execution layer
             (position sizing based on signal confidence)
🔜 Phase 8  Tick-level data (upgrade from snapshot LOB)
🔜 Phase 9  Federated multi-exchange signal fusion
```

---

## 🧯 VS Code Chat Model Error

If VS Code chat shows this on every message:

> `400 Model 'moonshotai/kimi-k2' is not in the catalog. Use 'auto' (or omit the 'model' field) ...`

it means your chat client is pinned to a model that your provider does not expose anymore.

Quick fix:
- In your chat/client config, set `model` to `auto` (or remove the `model` field).
- If your extension keeps old state, reload VS Code after saving config.
- If needed, query your provider's `/v1/models` endpoint and pick only a listed model.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built by [Gokavalasa Hemanth Naidu](https://github.com/GokavalasaHemanthNaidu)**

*Dual Degree (B.Tech + M.Tech) — Mathematics & Computing / CSE*

[![GitHub](https://img.shields.io/badge/GitHub-GokavalasaHemanthNaidu-181717?style=flat-square&logo=github)](https://github.com/GokavalasaHemanthNaidu)
[![HuggingFace](https://img.shields.io/badge/🤗-hemanthnaidug-yellow?style=flat-square)](https://huggingface.co/hemanthnaidug)

⭐ **Star this repo if you found it useful!**

</div>
