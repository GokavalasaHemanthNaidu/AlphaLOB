# Implementation Plan: AlphaLOB

*Note: This is a retrospective implementation plan mapping the sequenced roadmap that was used to build AlphaLOB.*

## Phase 1: Environment & Foundation Setup
**Goal:** Establish the core Python environment, infrastructure files, and single-container Docker setup.

- [x] Initialize Git repository and Python 3.11 virtual environment.
- [x] Create `requirements.txt` with `fastapi`, `uvicorn`, `onnxruntime`, `hmmlearn`, `duckdb`.
- [x] Scaffold project directory structure (`src/api`, `src/infrastructure`, `src/workers`, `models/weights`).
- [x] Set up basic `Dockerfile` optimized for Hugging Face Spaces / Render.
- [x] **Done Criteria:** The repository is structured, dependencies install without errors, and `docker build` completes successfully.

## Phase 2: Database & MLflow Infrastructure
**Goal:** Configure the local file-based databases to store signals and track model metrics before touching any business logic.

- [x] Implement `src/infrastructure/duckdb_client.py` for high-speed time-series storage.
- [x] Write `CREATE TABLE` definitions for `alpha_signals`, `backtest_runs`, and `trade_log`.
- [x] Implement `src/infrastructure/mlflow_sqlite.py` for lightweight ML tracking without an external server.
- [x] Create SQLite schemas for `metrics` and `signal_distribution`.
- [x] **Done Criteria:** Python scripts can connect to `.duckdb` and `.db` files, insert dummy rows, and query them successfully.

## Phase 3: Model Ingestion & Pipeline Scaffolding
**Goal:** Load the pre-trained ML models into memory and configure the asynchronous data passing queues.

- [x] Place `lobster_transformer.onnx` and `regime_hmm.bin` inside `models/weights/`.
- [x] Write `src/domain/inference.py` to wrap `onnxruntime.InferenceSession` and `hmmlearn`.
- [x] Setup `asyncio.Queue()` for the producer-consumer pipeline.
- [x] Create the `BybitWSClient` stub to prepare for Limit Order Book snapshot ingestion.
- [x] **Done Criteria:** Python can successfully load both models from disk and push/pull a mock numpy array through the queue without crashing.

## Phase 4: Core Features (FastAPI & Inference)
**Goal:** Build the main application logic connecting the models to HTTP endpoints. *(CRITICAL DEPENDENCY: Models from Phase 3 MUST be loaded into memory before initializing the FastAPI `app` object to prevent cold-start crashes).*

- [x] Create `src/api/main.py` and initialize the FastAPI app instance.
- [x] Build the custom dictionary-based rate limiting middleware (30 req/min).
- [x] Develop `/health` endpoint to verify model readiness.
- [x] Develop `/predict` endpoint (Ingests 10x4 LOB matrix → Runs ONNX inference → Returns 3 horizon probabilities).
- [x] Develop `/regime` endpoint (Ingests vol + autocorrelation → Runs HMM → Returns market state).
- [x] **Done Criteria:** Executing `curl` against `/predict` and `/regime` returns valid JSON probabilities in under 15ms.

## Phase 5: Dashboard Frontend
**Goal:** Build the single-page HTML UI embedded directly in the FastAPI response.

- [x] Create `HTMLResponse` string containing Tailwind CSS via CDN.
- [x] Build the sticky top navbar and "AlphaLOB" hero section with pulsing cursor.
- [x] Build the 4-column Metrics Grid with pure CSS hover tooltips and neon glows.
- [x] Implement the interactive Pipeline Timeline (6 clickable nodes).
- [x] Build the API Sandbox with hidden `<pre>` blocks.
- [x] Write Vanilla JS to execute `fetch()` requests and map responses to the terminal UI (including latency calculation).
- [x] **Done Criteria:** The root URL `/` renders beautifully in dark mode, and clicking "Send Request" visually returns the live JSON response.

## Phase 6: Testing & Polish
**Goal:** Ensure the system is robust, handles errors gracefully, and is ready for public deployment.

- [x] Write integration tests in `tests/integration/test_api.py` targeting the API endpoints.
- [x] Ensure the JS `.catch()` block correctly handles rate limits and offline states.
- [x] Validate UI/UX mathematically (Verify contrast ratio > 7:1).
- [x] Sanitize ONNX input layers (e.g., handling `NaN` and `Inf` inputs securely).
- [x] **Done Criteria:** `pytest` passes with 100% coverage on core routes, and manually forcing a bad input returns a `422 Unprocessable Entity` rather than a server crash.

## Phase 7: Pre-Launch & Deployment Checks
**Goal:** Ship the application to production environments.

- [x] **Code Review:** Ensure no hardcoded secrets or path leaks exist.
- [x] **Latency Check:** Confirm end-to-end inference stays under the 15ms CPU threshold constraint.
- [x] **Container Check:** Run `docker run -p 8000:8000` locally to ensure embedded DBs build correctly in the container volume.
- [x] **Deploy:** Push container to Hugging Face Spaces.
- [x] **Done Criteria:** The live URL is accessible globally, and the API Sandbox works seamlessly for end users.
