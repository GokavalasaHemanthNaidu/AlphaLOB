# Technical Requirements Document (TRD): AlphaLOB

## 1. Frontend Framework / Language
- **Stack:** Plain HTML5, Tailwind CSS (via CDN), and Vanilla JavaScript.
- **Architecture:** The frontend is a lightweight, single-page dashboard directly embedded and served via FastAPI's `HTMLResponse`. It features custom CSS for micro-animations and terminal-like aesthetics, keeping the footprint close to zero.

## 2. Backend Setup
- **Stack:** Python 3.11 + FastAPI.
- **Architecture:** 
  - Asynchronous HTTP handling via `uvicorn`.
  - Producer-Consumer pattern using `asyncio.Queue` and threading for concurrent feature engineering and ONNX inference.
  - Zero look-ahead bias pipeline architecture.

## 3. Database & Provider
- **Database:** Embedded file-based local databases.
- **Providers:** 
  - **DuckDB:** Local file-based storage (`alphalob.duckdb`) for tracking high-speed metrics and signal backtesting.
  - **MLflow-lite:** Local SQLite-based file tracking (`alphalob_mlflow.db`) for model metrics and drift detection.
- **Constraint:** Intentionally avoids heavy external distributed databases (e.g., PostgreSQL, MongoDB) to maintain the single-container philosophy.

## 4. Authentication
- **Method:** No user authentication (Public-facing API).
- **Security:** Implements basic IP-based rate limiting via a custom FastAPI dictionary-based middleware (capped at 30 requests per minute per IP). No external rate-limiting dependencies (like `slowapi`) are used to maintain zero-bloat.

## 5. Hosting & Deployment
- **Deployment Strategy:** Fully containerized via Docker.
- **Primary Host:** Hugging Face Spaces (CPU tier).
- **Secondary Host:** Render (via `render.yaml`).

## 6. Third-Party APIs & Services
- **Bybit WebSocket API:** Used for live, real-time Limit Order Book (LOB) snapshot ingestion via `websockets`. Streams 10-level deep order book data for real-world inference.

## 7. Key Libraries & Packages
- **fastapi** (>=0.110.0): High-performance API routing and data validation.
- **uvicorn** (>=0.27.0): ASGI web server.
- **onnxruntime** (>=1.17.0): CPU-optimized machine learning inference.
- **hmmlearn** (>=0.3.0) / **joblib** (>=1.3.0): Hidden Markov Model training and serialization.
- **numpy** (>=1.24.0): High-performance mathematical operations.

## 8. Environment Variables
- `ALPHALOB_DB_PATH`: Defines the path for the DuckDB storage (defaults to `alphalob.duckdb`).

## 9. Hard Constraints
- **Latency:** End-to-end inference MUST complete in <15 milliseconds (p99 latency) strictly on a CPU.
- **Infrastructure:** Must operate as a standalone, single-container application. Cannot rely on external message brokers (Kafka/RabbitMQ) or memory caches (Redis).
- **Resource Limits:** Must operate efficiently within free-tier cloud environments (e.g., Hugging Face Spaces 2vCPU / 16GB RAM limit).

---

## Folder Structure Diagram

```text
AlphaLOB/
├── .github/                # CI/CD workflows
├── docs/                   # ADRs and project documentation
├── models/
│   └── weights/            # ONNX and HMM serialized model files
├── notebooks/              # Jupyter notebooks for EDA and model training
├── scripts/                # Utility and deployment scripts
├── src/
│   ├── api/
│   │   ├── main.py         # FastAPI application entrypoint and embedded UI
│   │   └── routes/         # API endpoints (e.g., /predict, /health)
│   ├── data/               # Data ingestion (Bybit WS, synthetic generation)
│   ├── domain/             # Core business logic (Features, Backtesting, Inference wrapper)
│   ├── infrastructure/     # Utilities (Logging, MLflow, DuckDB, Drift Detection)
│   └── workers/            # Asyncio background task workers
├── tests/
│   ├── integration/        # E2E API tests
│   └── unit/               # Unit tests for domain logic
├── .gitignore
├── Dockerfile              # Containerization instructions
├── LICENSE
├── PRD.md                  # Product Requirements Document
├── README.md               # Main repository documentation
├── render.yaml             # Render deployment configuration
├── requirements.txt        # Python package dependencies
└── TRD.md                  # Technical Requirements Document
```
