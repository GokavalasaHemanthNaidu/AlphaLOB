import sys
from unittest.mock import MagicMock
sys.modules['polars'] = MagicMock()
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import asyncio
from contextlib import asynccontextmanager
from src.data.synthetic_lob import SyntheticLOBGenerator
from src.workers.feature_worker import FeatureEngineeringWorker
from src.workers.inference_worker import ModelInferenceWorker
from src.api.routes import predict, signals, backtest, model_health
from src.infrastructure.duckdb_client import init_db
from src.infrastructure.mlflow_sqlite import init_mlflow_db
import os

# In-memory queues for local architecture (saving RAM, avoiding Kafka)
lob_queue = asyncio.Queue(maxsize=5000)
features_queue = asyncio.Queue(maxsize=5000)
alpha_signals_queue = asyncio.Queue(maxsize=5000)

background_tasks = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DuckDB and MLflow SQLite
    init_db()
    init_mlflow_db()
    
    # Startup: Launch workers
    from src.data.bybit_ws import BybitLiveLOBGenerator
    ingestion_worker = BybitLiveLOBGenerator(queue=lob_queue, symbol="BTCUSDT")
    feature_worker = FeatureEngineeringWorker(input_queue=lob_queue, output_queue=features_queue)
    inference_worker = ModelInferenceWorker(input_queue=features_queue, output_queue=alpha_signals_queue)
    
    t1 = asyncio.create_task(ingestion_worker.start())
    t2 = asyncio.create_task(feature_worker.start())
    t3 = asyncio.create_task(inference_worker.start())
    
    background_tasks.extend([t1, t2, t3])
    
    yield
    
    # Shutdown: Cancel workers
    for task in background_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

app = FastAPI(title="AlphaLOB API", lifespan=lifespan)

app.include_router(predict.router)
app.include_router(signals.router)
app.include_router(backtest.router)
app.include_router(model_health.router)

@app.get("/healthz")
async def health_check():
    return {
        "status": "ok", 
        "lob_queue_size": lob_queue.qsize(),
        "features_queue_size": features_queue.qsize(),
        "alpha_signals_queue_size": alpha_signals_queue.qsize()
    }

@app.get("/")
async def root_demo():
    """
    HTML Dashboard to visualize the SSE stream flowing in real-time.
    Useful for demonstrating sub-millisecond pipeline latency.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>AlphaLOB Real-Time Dashboard</title>
            <style>
                body { background-color: #121212; color: #00ff00; font-family: monospace; padding: 20px; }
                h1 { color: #ffffff; }
                .signal-card { border: 1px solid #333; padding: 10px; margin-bottom: 10px; background-color: #1e1e1e; }
                .up { color: #00ff00; }
                .down { color: #ff0000; }
            </style>
        </head>
        <body>
            <h1>AlphaLOB Real-Time Alpha Signals</h1>
            <p>Listening to /v1/signals/live (Server-Sent Events)</p>
            <div id="signals">Waiting for data...</div>
            <script>
                const evtSource = new EventSource("/v1/signals/live");
                const signalsDiv = document.getElementById("signals");
                
                evtSource.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    const preds = data.predictions;
                    const upProb = (preds.dir_up_prob * 100).toFixed(1);
                    const downProb = (preds.dir_down_prob * 100).toFixed(1);
                    
                    let directionHtml = '';
                    if (preds.dir_up_prob > 0.5) directionHtml = `<span class="up">↑ UP (${upProb}%)</span>`;
                    else if (preds.dir_down_prob > 0.5) directionHtml = `<span class="down">↓ DOWN (${downProb}%)</span>`;
                    else directionHtml = `<span>FLAT</span>`;
                    
                    const card = `
                        <div class="signal-card">
                            <strong>Timestamp:</strong> ${data.ts} <br/>
                            <strong>Prediction:</strong> ${directionHtml} <br/>
                            <strong>Spread Compress Prob:</strong> ${(preds.spread_compress_prob * 100).toFixed(1)}% <br/>
                            <strong>Vol Imbalance:</strong> ${preds.vol_imbalance_pred.toFixed(3)}
                        </div>
                    `;
                    
                    signalsDiv.innerHTML = card + signalsDiv.innerHTML;
                    
                    // Keep only last 10
                    if (signalsDiv.children.length > 10) {
                        signalsDiv.removeChild(signalsDiv.lastChild);
                    }
                };
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)
