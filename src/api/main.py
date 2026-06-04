"""
AlphaLOB Production FastAPI Service
Routes: /health, /predict, /regime
"""
import os
import numpy as np
import onnxruntime as ort
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import time
from collections import defaultdict

# ── Model paths (relative to app.py location) ────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ONNX_PATH  = os.path.join(BASE_DIR, "models", "weights", "lobster_transformer.onnx")
HMM_PATH   = os.path.join(BASE_DIR, "models", "weights", "regime_hmm.bin")

# ── Load models at startup ────────────────────────────────────────────────────
print("Loading ONNX model...")
ort_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
print(f"  ONNX loaded: {ONNX_PATH}")

print("Loading RegimeHMM...")
hmm_model = joblib.load(HMM_PATH)
assert hasattr(hmm_model, "regime_names"), "regime_names missing from HMM model!"
print(f"  HMM loaded: {HMM_PATH} | regimes: {list(hmm_model.regime_names.values())}")

# ── FastAPI application ───────────────────────────────────────────────────────
app = FastAPI(
    title="AlphaLOB Inference API",
    description="Real-time LOB inference: directional probability + regime detection",
    version="1.0.0",
)

# ── CORS — allow interviewers/Postman to call the API from any origin ─────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Rate Limiting — protect free-tier CPU (30 req/min per IP) ─────────────────
_request_log: dict = defaultdict(list)

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = [t for t in _request_log[ip] if now - t < 60]
    if len(window) >= 30:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Max 30 requests/min."})
    window.append(now)
    _request_log[ip] = window
    return await call_next(request)


# ── Request/Response schemas ─────────────────────────────────────────────────

class LOBSnapshot(BaseModel):
    """
    Input: one LOB snapshot as a 10×4 matrix.
    Rows = LOB levels 0..9 (best bid/ask first).
    Columns = [normalized_price_dist, log_normalized_vol, wofi_z, kyle_lambda_z]
    """
    lob_snapshot: List[List[float]]   # shape (10, 4)


class PredictResponse(BaseModel):
    dir_5s_prob_up:    float   # P(price UP in 5s)
    dir_30s_prob_up:   float   # P(price UP in 30s)  ← KEY metric
    dir_5min_prob_up:  float   # P(price UP in 5min)
    spread_compress:   float   # P(spread compresses) — sigmoid [0,1]
    vol_imbalance:     float   # order flow imbalance regression


class RegimeInput(BaseModel):
    realized_vol:    float   # rolling 100-tick std of log-returns
    autocorrelation: float   # rolling lag-1 autocorr of log-returns


class RegimeResponse(BaseModel):
    regime:       str          # "TRENDING", "MEAN_REVERTING", or "VOLATILE"
    state_id:     int          # integer HMM state (0, 1, or 2)
    probabilities: dict        # {regime_name: probability} for all states


# ── Endpoints ─────────────────────────────────────────────────────────────────

from fastapi.responses import HTMLResponse

LANDING_HTML = """
<!DOCTYPE html>
<html class="dark" lang="en"><head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📊</text></svg>"/>
<meta name="description" content="AlphaLOB — Real-time Limit Order Book alpha signals via ONNX Transformer + HMM regime detection."/>
<meta property="og:title" content="AlphaLOB | Real-Time LOB Alpha Signals"/>
<meta property="og:description" content="51.25% directional accuracy. 5.47ms p99 latency. Zero look-ahead bias."/>
<title>AlphaLOB | Real-Time Limit Order Book Alpha Signals</title>
<!-- Material Symbols -->
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&amp;display=swap" rel="stylesheet"/>
<!-- Google Fonts: JetBrains Mono -->
<link href="https://fonts.googleapis.com" rel="preconnect"/>
<link crossorigin="" href="https://fonts.gstatic.com" rel="preconnect"/>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&amp;display=swap" rel="stylesheet"/>
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&amp;display=swap" rel="stylesheet"/>
<!-- Tailwind CSS -->
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<!-- Tailwind Config injected from Design System -->
<style type="text/tailwindcss">
        @layer utilities {
          .pipeline-step { cursor: pointer; transition: opacity 0.2s; }
          .pipeline-step:hover { opacity: 0.8; }
        }
</style>
<script id="tailwind-config">
        tailwind.config = {
          darkMode: "class",
          theme: {
            extend: {
              "colors": {
                      "tertiary-fixed": "#ffdad6",
                      "secondary": "#67df70",
                      "on-secondary-container": "#00320a",
                      "primary-container": "#58a6ff",
                      "tertiary": "#ffb4ac",
                      "outline": "#8b919d",
                      "surface-container-lowest": "#060f16",
                      "inverse-primary": "#0060aa",
                      "tertiary-container": "#ff7b70",
                      "primary-fixed": "#d3e4ff",
                      "surface-container": "#182028",
                      "on-tertiary-fixed-variant": "#93000d",
                      "secondary-container": "#27a640",
                      "on-error-container": "#ffdad6",
                      "surface-variant": "#2d363e",
                      "surface-container-low": "#141c24",
                      "on-secondary": "#00390d",
                      "on-secondary-fixed": "#002105",
                      "secondary-fixed-dim": "#67df70",
                      "surface-bright": "#313a43",
                      "surface-container-high": "#222b33",
                      "inverse-surface": "#dae3ee",
                      "surface-dim": "#0b141c",
                      "on-primary-container": "#003a6b",
                      "on-error": "#690005",
                      "inverse-on-surface": "#29313a",
                      "outline-variant": "#414752",
                      "on-surface": "#dae3ee",
                      "on-tertiary-fixed": "#410002",
                      "on-secondary-fixed-variant": "#005317",
                      "error-container": "#93000a",
                      "surface-tint": "#a2c9ff",
                      "primary-fixed-dim": "#a2c9ff",
                      "secondary-fixed": "#83fc89",
                      "on-primary": "#00315c",
                      "on-surface-variant": "#c0c7d4",
                      "on-tertiary-container": "#790009",
                      "on-primary-fixed": "#001c38",
                      "background": "#0d1117",
                      "primary": "#a2c9ff",
                      "on-tertiary": "#690007",
                      "tertiary-fixed-dim": "#ffb4ac",
                      "on-background": "#dae3ee",
                      "surface-container-highest": "#2d363e",
                      "on-primary-fixed-variant": "#004882",
                      "surface": "#0d1117",
                      "error": "#ffb4ab"
              },
              "borderRadius": {
                      "DEFAULT": "0.125rem",
                      "lg": "0.25rem",
                      "xl": "0.5rem",
                      "full": "0.75rem"
              },
              "spacing": {
                      "unit": "4px",
                      "stack-default": "12px",
                      "stack-compact": "4px",
                      "gutter": "8px",
                      "container-margin": "16px"
              },
              "fontFamily": {
                      "title-sm": [
                              "JetBrains Mono"
                      ],
                      "headline-md": [
                              "JetBrains Mono"
                      ],
                      "body-md": [
                              "JetBrains Mono"
                      ],
                      "display-lg": [
                              "JetBrains Mono"
                      ],
                      "body-sm": [
                              "JetBrains Mono"
                      ],
                      "data-mono": [
                              "JetBrains Mono"
                      ],
                      "label-caps": [
                              "JetBrains Mono"
                      ]
              },
              "fontSize": {
                      "title-sm": [
                              "18px",
                              {
                                      "lineHeight": "24px",
                                      "fontWeight": "600"
                              }
                      ],
                      "headline-md": [
                              "24px",
                              {
                                      "lineHeight": "32px",
                                      "letterSpacing": "-0.01em",
                                      "fontWeight": "600"
                              }
                      ],
                      "body-md": [
                              "14px",
                              {
                                      "lineHeight": "20px",
                                      "fontWeight": "400"
                              }
                      ],
                      "display-lg": [
                              "32px",
                              {
                                      "lineHeight": "40px",
                                      "letterSpacing": "-0.02em",
                                      "fontWeight": "700"
                              }
                      ],
                      "body-sm": [
                              "12px",
                              {
                                      "lineHeight": "16px",
                                      "fontWeight": "400"
                              }
                      ],
                      "data-mono": [
                              "13px",
                              {
                                      "lineHeight": "18px",
                                      "fontWeight": "500"
                              }
                      ],
                      "label-caps": [
                              "11px",
                              {
                                      "lineHeight": "14px",
                                      "letterSpacing": "0.05em",
                                      "fontWeight": "700"
                              }
                      ]
              }
      },
          },
        }
    </script>
<style type="text/tailwindcss">
        /* Custom styles for terminal/high-performance feel */
        body {
            background-color: theme('colors.background');
            color: theme('colors.on-background');
        }
        
        .technical-border {
            border: 1px solid theme('colors.surface-container-highest');
        }
        
        .glow-hover:hover {
            box-shadow: 0 0 8px rgba(162, 201, 255, 0.3); /* primary-fixed-dim at 30% */
            background-color: theme('colors.surface-container-high');
        }
        
        /* Terminal Typing Effect */
        .typing-effect {
            overflow: hidden;
            white-space: nowrap;
            border-right: 2px solid theme('colors.primary');
            animation: typing 3.5s steps(40, end), blink-caret .75s step-end infinite;
        }

        @keyframes typing {
            from { width: 0 }
            to { width: 100% }
        }

        @keyframes blink-caret {
            from, to { border-color: transparent }
            50% { border-color: theme('colors.primary') }
        }

        /* Tooltip styling */
        .tooltip {
            position: relative;
            display: inline-block;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: theme('colors.surface-container-highest');
            color: theme('colors.on-surface');
            text-align: center;
            border-radius: theme('borderRadius.DEFAULT');
            padding: theme('spacing.stack-compact');
            position: absolute;
            z-index: 10;
            bottom: 125%; 
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
            border: 1px solid theme('colors.outline');
            font-size: 11px;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* Custom Scrollbar for Pre blocks */
        pre::-webkit-scrollbar {
            height: 4px;
            width: 4px;
        }
        pre::-webkit-scrollbar-track {
            background: theme('colors.surface-container-low');
        }
        pre::-webkit-scrollbar-thumb {
            background: theme('colors.outline');
            border-radius: 2px;
        }

        /* Flow diagram lines */
        .flow-line {
            height: 2px;
            background: repeating-linear-gradient(
                90deg,
                theme('colors.outline-variant'),
                theme('colors.outline-variant') 4px,
                transparent 4px,
                transparent 8px
            );
            flex-grow: 1;
            min-width: 20px;
        }
        
        @media (max-width: 768px) {
            .flow-line {
                width: 2px;
                height: 20px;
                min-height: 20px;
                background: repeating-linear-gradient(
                    180deg,
                    theme('colors.outline-variant'),
                    theme('colors.outline-variant') 4px,
                    transparent 4px,
                    transparent 8px
                );
            }
        }
        html { scroll-behavior: smooth; }
    </style>
</head>
<body class="bg-background text-on-background font-body-md text-body-md min-h-screen flex flex-col antialiased">
<!-- Top Navigation -->
<header class="flex justify-between items-center py-4 border-b border-gray-800 px-6 max-w-7xl mx-auto w-full">
  <div class="text-xl font-bold text-white tracking-tight">AlphaLOB</div>
  <nav class="space-x-6 text-sm flex items-center">
    <span id="api-status" class="text-xs font-mono">● Checking...</span>
    <a href="#api" class="text-gray-400 hover:text-white transition">API</a>
    <a href="https://github.com/GokavalasaHemanthNaidu/AlphaLOB" target="_blank" class="text-gray-400 hover:text-white transition">GitHub</a>
  </nav>
</header>
<main class="flex-grow p-4 md:p-8 space-y-12">
<!-- Section 1: Hero -->
<section class="max-w-7xl mx-auto w-full pt-8">
  <div class="flex gap-2 mb-4">
    <span class="px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-bold">● LIVE</span>
    <span class="px-2 py-0.5 rounded bg-blue-500/10 border border-blue-500/30 text-blue-400 text-xs font-bold">ONNX Runtime</span>
    <span class="px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-bold">Zero Look-Ahead Bias</span>
  </div>
  
  <h1 class="text-4xl md:text-5xl font-mono font-bold text-white mb-2">
    > AlphaLOB<span class="animate-pulse">_</span>
  </h1>
  <p class="text-xl text-gray-400 font-light">Real-Time Limit Order Book Alpha Signals</p>
</section>

<!-- Mathematical Rigor Banner -->
<section class="max-w-7xl mx-auto w-full">
  <div class="mb-10 p-4 rounded-lg border border-emerald-500/20 bg-emerald-500/5">
    <p class="text-sm text-emerald-300">
      <span class="font-bold">✓ Mathematical Rigor:</span> 51.25% directional accuracy matches the theoretical ceiling for a WOFI-return correlation of 0.044. This validates zero data leakage and a causally sound pipeline.
    </p>
  </div>
</section>
<!-- Section 2: Metrics Grid -->
<section class="max-w-7xl mx-auto w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
<!-- Accuracy -->
<div class="bg-gray-800/30 border border-gray-700 p-6 rounded-DEFAULT hover:shadow-lg hover:shadow-blue-500/10 hover:-translate-y-1 transition-all duration-300">
<div class="flex items-center justify-between mb-4">
<span class="text-xs text-gray-400 uppercase tracking-wider">Predictive Edge</span>
<span class="material-symbols-outlined text-blue-400 text-sm">analytics</span>
</div>
<div class="text-2xl text-white mb-1 flex items-center">
<span class="text-emerald-400 text-sm mr-1">▲</span> 51.25%
</div>
<div class="text-sm text-gray-500 border-b border-dashed border-gray-600 inline-block group relative cursor-help">
Honest Ceiling
  <div class="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block w-48 p-2 bg-gray-900 text-gray-300 text-xs rounded border border-gray-700 shadow-xl z-20 text-center">
    Theoretical ceiling for highly stochastic process (ρ=0.044) without look-ahead bias.
  </div>
</div>
</div>
<!-- Latency -->
<div class="bg-gray-800/30 border border-gray-700 p-6 rounded-DEFAULT hover:shadow-lg hover:shadow-blue-500/10 hover:-translate-y-1 transition-all duration-300 relative overflow-hidden group/card">
<div class="flex items-center justify-between mb-4">
<span class="text-xs text-gray-400 uppercase tracking-wider">Inference Latency</span>
<span class="material-symbols-outlined text-blue-400 text-sm">speed</span>
</div>
<div class="text-2xl text-white mb-1 relative z-10">
5.47ms
</div>
<div class="text-xs text-gray-500 border-b border-dashed border-gray-600 inline-block mt-1 group relative cursor-help z-10">
  p50: 0.82ms · p90: 1.18ms · p95: 1.29ms · <span class="text-emerald-400">p99: 5.47ms</span>
  <div class="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block w-48 p-2 bg-gray-900 text-gray-300 text-xs rounded border border-gray-700 shadow-xl z-20 text-center">
    p99 latency measured on standard CPU. Target SLA is &lt;15ms.
  </div>
</div>
<!-- Sparkline Chart -->
<div class="absolute bottom-0 left-0 w-full h-10 flex items-end px-2 gap-1 opacity-30 group-hover/card:opacity-80 transition-opacity z-0 pointer-events-none">
  <div class="w-1/4 bg-blue-500/40 rounded-t-sm" style="height: 20%;"></div>
  <div class="w-1/4 bg-blue-500/60 rounded-t-sm" style="height: 30%;"></div>
  <div class="w-1/4 bg-blue-500/80 rounded-t-sm" style="height: 40%;"></div>
  <div class="w-1/4 bg-emerald-500/80 rounded-t-sm" style="height: 80%;"></div>
</div>
</div>
<!-- Architecture -->
<div class="bg-gray-800/30 border border-gray-700 p-6 rounded-DEFAULT hover:shadow-lg hover:shadow-blue-500/10 hover:-translate-y-1 transition-all duration-300">
<div class="flex items-center justify-between mb-4">
<span class="text-xs text-gray-400 uppercase tracking-wider">Core Engine</span>
<span class="material-symbols-outlined text-blue-400 text-sm">memory</span>
</div>
<div class="text-2xl text-white mb-1">
6-Layer Transformer
</div>
<div class="text-sm text-gray-500 border-b border-dashed border-gray-600 inline-block group relative cursor-help">
8-Head Attention
  <div class="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block w-48 p-2 bg-gray-900 text-gray-300 text-xs rounded border border-gray-700 shadow-xl z-20 text-center">
    Multi-head attention captures complex order book spatial features.
  </div>
</div>
</div>
<!-- Regime -->
<div class="bg-gray-800/30 border border-gray-700 p-6 rounded-DEFAULT hover:shadow-lg hover:shadow-blue-500/10 hover:-translate-y-1 transition-all duration-300">
<div class="flex items-center justify-between mb-4">
<span class="text-xs text-gray-400 uppercase tracking-wider">Market State</span>
<span class="material-symbols-outlined text-blue-400 text-sm">waves</span>
</div>
<div class="text-2xl text-white mb-1">
3-State HMM
</div>
<div class="text-sm text-gray-500 border-b border-dashed border-gray-600 inline-block group relative cursor-help">
Gaussian HMM Regimes
  <div class="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block w-48 p-2 bg-gray-900 text-gray-300 text-xs rounded border border-gray-700 shadow-xl z-20 text-center">
    Unsupervised segmentation of market conditions into Trending, Mean-Reverting, and Volatile states.
  </div>
</div>
</div>
</section>

<!-- Quick Live Demo -->
<section class="max-w-7xl mx-auto w-full mt-12 border border-gray-700 rounded-lg p-6 bg-gray-800/30">
  <h3 class="text-lg font-semibold text-white mb-4">🔴 Live API Test</h3>
  <div class="flex gap-4">
    <button onclick="quickTestHealth()" class="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-white text-sm font-medium transition">
      Test /health
    </button>
    <button onclick="quickTestPredict()" class="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded text-white text-sm font-medium transition">
      Test /predict
    </button>
  </div>
  <pre id="quick-api-output" class="mt-4 p-3 bg-black rounded text-xs text-green-400 font-mono hidden overflow-x-auto"></pre>
</section>

<script>
async function quickTestHealth() {
  const out = document.getElementById('quick-api-output');
  out.classList.remove('hidden');
  out.textContent = 'Loading /health...';
  try {
    const res = await fetch('/health');
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = 'Error: ' + e.message;
  }
}
async function quickTestPredict() {
  const out = document.getElementById('quick-api-output');
  out.classList.remove('hidden');
  out.textContent = 'Loading /predict...';
  try {
    const lob_snapshot = Array.from({length: 10}, () => [Math.random()*0.2-0.1, Math.random()*0.1, 100+Math.random(), 100+Math.random()]);
    const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lob_snapshot })
    });
    const data = await res.json();
    out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = 'Error: ' + e.message;
  }
}
</script>
</section>
<!-- Pipeline Visualization -->
<!-- PROFESSIONAL PIPELINE SECTION -->
<section id="pipeline" class="max-w-7xl mx-auto w-full px-6 py-16">
    <div class="text-center mb-10">
        <h2 class="text-3xl font-bold text-white mb-2">End-to-End Pipeline</h2>
        <p class="text-gray-400 text-sm">From synthetic data generation to sub-15ms production inference</p>
    </div>

    <!-- Desktop: Horizontal Flow | Mobile: Vertical Stack -->
    <div class="relative">
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-8 lg:gap-4">

            <!-- STEP 01 -->
            <div class="group cursor-pointer relative" onclick="togglePipeline(1)">
                <div class="hidden md:block absolute top-7 left-[50%] w-full h-0.5 bg-gradient-to-r from-blue-500 to-indigo-500 opacity-40 z-0"></div>
                <div class="flex flex-col items-center text-center relative z-10">
                    <div class="relative w-14 h-14 rounded-2xl bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-blue-500/20 group-hover:shadow-blue-500/50 group-hover:scale-110 transition-all duration-300 border border-blue-400/40 ring-2 ring-transparent group-hover:ring-blue-400/30">
                        <span class="group-hover:hidden">01</span>
                        <svg class="w-6 h-6 hidden group-hover:block" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7"></path></svg>
                    </div>
                    <div class="mt-3">
                        <h3 class="text-white font-semibold text-sm group-hover:text-blue-400 transition-colors">Synthetic LOB</h3>
                        <p class="text-gray-500 text-xs mt-0.5">Data Generation</p>
                    </div>
                    <div class="mt-2 px-2 py-0.5 rounded-full bg-green-500/10 border border-green-500/30 text-green-400 text-[10px] font-bold tracking-wider">COMPLETED</div>
                </div>
                <div id="pipe-1" class="hidden mt-3 p-3 bg-gray-800/60 rounded-lg border border-gray-700/50 text-xs text-gray-300 space-y-1 text-left relative z-10">
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> 5M synthetic ticks</div>
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> OU mean-reversion</div>
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> Volume microstructure</div>
                </div>
            </div>

            <!-- STEP 02 -->
            <div class="group cursor-pointer relative" onclick="togglePipeline(2)">
                <div class="hidden lg:block absolute top-7 left-[50%] w-full h-0.5 bg-gradient-to-r from-indigo-500 to-emerald-500 opacity-40 z-0"></div>
                <div class="flex flex-col items-center text-center relative z-10">
                    <div class="relative w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-500 to-indigo-700 flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-indigo-500/20 group-hover:shadow-indigo-500/50 group-hover:scale-110 transition-all duration-300 border border-indigo-400/40 ring-2 ring-transparent group-hover:ring-indigo-400/30">
                        <span class="group-hover:hidden">02</span>
                        <svg class="w-6 h-6 hidden group-hover:block" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7"></path></svg>
                    </div>
                    <div class="mt-3">
                        <h3 class="text-white font-semibold text-sm group-hover:text-indigo-400 transition-colors">Feature Eng</h3>
                        <p class="text-gray-500 text-xs mt-0.5">WOFI · Hawkes · Kyle</p>
                    </div>
                    <div class="mt-2 px-2 py-0.5 rounded-full bg-green-500/10 border border-green-500/30 text-green-400 text-[10px] font-bold tracking-wider">COMPLETED</div>
                </div>
                <div id="pipe-2" class="hidden mt-3 p-3 bg-gray-800/60 rounded-lg border border-gray-700/50 text-xs text-gray-300 space-y-1 text-left relative z-10">
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> WOFI (order flow imbalance)</div>
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> Hawkes intensity</div>
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> Kyle's Lambda</div>
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> Amihud ILLIQ</div>
                </div>
            </div>

            <!-- STEP 03 -->
            <div class="group cursor-pointer relative" onclick="togglePipeline(3)">
                <div class="hidden md:block absolute top-7 left-[50%] w-full h-0.5 bg-gradient-to-r from-emerald-500 to-amber-500 opacity-40 z-0"></div>
                <div class="flex flex-col items-center text-center relative z-10">
                    <div class="relative w-14 h-14 rounded-2xl bg-gradient-to-br from-emerald-500 to-emerald-700 flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-emerald-500/20 group-hover:shadow-emerald-500/50 group-hover:scale-110 transition-all duration-300 border border-emerald-400/40 ring-2 ring-transparent group-hover:ring-emerald-400/30">
                        <span class="group-hover:hidden">03</span>
                        <svg class="w-6 h-6 hidden group-hover:block" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7"></path></svg>
                    </div>
                    <div class="mt-3">
                        <h3 class="text-white font-semibold text-sm group-hover:text-emerald-400 transition-colors">LOBTransformer</h3>
                        <p class="text-gray-500 text-xs mt-0.5">6-Layer · 8 Heads</p>
                    </div>
                    <div class="mt-2 px-2 py-0.5 rounded-full bg-green-500/10 border border-green-500/30 text-green-400 text-[10px] font-bold tracking-wider">COMPLETED</div>
                </div>
                <div id="pipe-3" class="hidden mt-3 p-3 bg-gray-800/60 rounded-lg border border-gray-700/50 text-xs text-gray-300 space-y-1 text-left relative z-10">
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> 51.25% Val Acc</div>
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> Kendall uncertainty</div>
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> Multi-task heads</div>
                </div>
            </div>

            <!-- STEP 04 -->
            <div class="group cursor-pointer relative" onclick="togglePipeline(4)">
                <div class="hidden lg:block absolute top-7 left-[50%] w-full h-0.5 bg-gradient-to-r from-amber-500 to-violet-500 opacity-40 z-0"></div>
                <div class="flex flex-col items-center text-center relative z-10">
                    <div class="relative w-14 h-14 rounded-2xl bg-gradient-to-br from-amber-500 to-amber-700 flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-amber-500/20 group-hover:shadow-amber-500/50 group-hover:scale-110 transition-all duration-300 border border-amber-400/40 ring-2 ring-transparent group-hover:ring-amber-400/30">
                        <span class="group-hover:hidden">04</span>
                        <svg class="w-6 h-6 hidden group-hover:block" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7"></path></svg>
                    </div>
                    <div class="mt-3">
                        <h3 class="text-white font-semibold text-sm group-hover:text-amber-400 transition-colors">Regime HMM</h3>
                        <p class="text-gray-500 text-xs mt-0.5">3-State Gaussian</p>
                    </div>
                    <div class="mt-2 px-2 py-0.5 rounded-full bg-green-500/10 border border-green-500/30 text-green-400 text-[10px] font-bold tracking-wider">COMPLETED</div>
                </div>
                <div id="pipe-4" class="hidden mt-3 p-3 bg-gray-800/60 rounded-lg border border-gray-700/50 text-xs text-gray-300 space-y-1 text-left relative z-10">
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> TRENDING regime</div>
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> MEAN_REVERTING</div>
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> VOLATILE</div>
                </div>
            </div>

            <!-- STEP 05 -->
            <div class="group cursor-pointer relative" onclick="togglePipeline(5)">
                <div class="hidden md:block absolute top-7 left-[50%] w-full h-0.5 bg-gradient-to-r from-violet-500 to-rose-500 opacity-40 z-0"></div>
                <div class="flex flex-col items-center text-center relative z-10">
                    <div class="relative w-14 h-14 rounded-2xl bg-gradient-to-br from-violet-500 to-violet-700 flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-violet-500/20 group-hover:shadow-violet-500/50 group-hover:scale-110 transition-all duration-300 border border-violet-400/40 ring-2 ring-transparent group-hover:ring-violet-400/30">
                        <span class="group-hover:hidden">05</span>
                        <svg class="w-6 h-6 hidden group-hover:block" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7"></path></svg>
                    </div>
                    <div class="mt-3">
                        <h3 class="text-white font-semibold text-sm group-hover:text-violet-400 transition-colors">WF Backtest</h3>
                        <p class="text-gray-500 text-xs mt-0.5">Walk-Forward OOS</p>
                    </div>
                    <div class="mt-2 px-2 py-0.5 rounded-full bg-green-500/10 border border-green-500/30 text-green-400 text-[10px] font-bold tracking-wider">COMPLETED</div>
                </div>
                <div id="pipe-5" class="hidden mt-3 p-3 bg-gray-800/60 rounded-lg border border-gray-700/50 text-xs text-gray-300 space-y-1 text-left relative z-10">
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> 3 expanding windows</div>
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> Causal daily vol</div>
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> 51.3% OOS accuracy</div>
                </div>
            </div>

            <!-- STEP 06 -->
            <div class="group cursor-pointer relative" onclick="togglePipeline(6)">
                <div class="flex flex-col items-center text-center relative z-10">
                    <div class="relative w-14 h-14 rounded-2xl bg-gradient-to-br from-rose-500 to-rose-700 flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-rose-500/20 group-hover:shadow-rose-500/50 group-hover:scale-110 transition-all duration-300 border border-rose-400/40 ring-2 ring-transparent group-hover:ring-rose-400/30">
                        <span class="group-hover:hidden">06</span>
                        <svg class="w-6 h-6 hidden group-hover:block" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7"></path></svg>
                    </div>
                    <div class="mt-3">
                        <h3 class="text-white font-semibold text-sm group-hover:text-rose-400 transition-colors">ONNX Deploy</h3>
                        <p class="text-gray-500 text-xs mt-0.5">p99: 5.47ms</p>
                    </div>
                    <div class="mt-2 px-2 py-0.5 rounded-full bg-green-500/10 border border-green-500/30 text-green-400 text-[10px] font-bold tracking-wider">LIVE</div>
                </div>
                <div id="pipe-6" class="hidden mt-3 p-3 bg-gray-800/60 rounded-lg border border-gray-700/50 text-xs text-gray-300 space-y-1 text-left">
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> ONNX Runtime</div>
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> FastAPI</div>
                    <div class="flex items-center gap-2"><span class="text-green-400">✓</span> Hugging Face Spaces</div>
                </div>
            </div>

        </div>
    </div>
</section>

<script>
function togglePipeline(n) {
    const el = document.getElementById('pipe-' + n);
    const all = document.querySelectorAll('[id^="pipe-"]');
    all.forEach(d => { if (d !== el) d.classList.add('hidden'); });
    el.classList.toggle('hidden');
}
</script>
<!-- Interactive API Documentation -->
<section class="max-w-7xl mx-auto w-full space-y-6" id="api">
<h2 class="text-2xl font-bold text-white mb-6">API Documentation &amp; Sandbox</h2>

<!-- /health -->
<div class="border border-gray-700 rounded-lg bg-gray-800/30 p-6">
  <div class="flex items-center gap-3 mb-3">
    <span class="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 text-xs font-bold">GET</span>
    <h3 class="text-white font-semibold">/health</h3>
  </div>
  <p class="text-sm text-gray-400 mb-4">Check API and model readiness.</p>
  
  <div class="relative group mb-4">
    <pre id="curl-health" class="text-xs bg-black/50 p-3 rounded border border-gray-700 font-mono text-blue-300 whitespace-pre-wrap break-all">curl -X GET https://hemanthnaidug-alphalob.hf.space/health</pre>
    <button onclick="copyToClipboard('curl-health')" class="absolute top-2 right-2 px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-xs text-gray-300 opacity-0 group-hover:opacity-100 transition">Copy</button>
  </div>
  
  <button onclick="testHealth()" class="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded text-white text-sm font-medium transition flex items-center gap-2">
    <span>▶</span> Ping API
  </button>
  
  <div id="health-output" class="mt-4 hidden">
    <div class="flex items-center gap-2 mb-2">
      <span id="health-status" class="px-2 py-0.5 rounded text-xs font-bold bg-gray-700 text-gray-300">--</span>
      <span id="health-latency" class="text-xs text-gray-500 font-mono">0ms</span>
    </div>
    <pre id="health-json" class="text-xs bg-black/70 p-3 rounded border border-gray-700 font-mono text-green-400 overflow-x-auto"></pre>
  </div>
</div>

<!-- /predict -->
<div class="border border-gray-700 rounded-lg bg-gray-800/30 p-6">
  <div class="flex items-center gap-3 mb-3">
    <span class="px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 text-xs font-bold">POST</span>
    <h3 class="text-white font-semibold">/predict</h3>
  </div>
  <p class="text-sm text-gray-400 mb-4">Generate alpha signals from a 10-level LOB snapshot. Returns directional probabilities + spread/vol signals.</p>
  
  <div class="relative group mb-4">
    <pre id="curl-predict" class="text-xs bg-black/50 p-3 rounded border border-gray-700 font-mono text-blue-300 whitespace-pre-wrap break-all">
curl -X POST https://hemanthnaidug-alphalob.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"lob_snapshot": [
    [0.0, 1.2, 0.5, 0.3],
    [0.0, 1.1, 0.5, 0.3],
    [0.0, 1.0, 0.5, 0.3],
    [0.0, 0.9, 0.5, 0.3],
    [0.0, 0.8, 0.5, 0.3],
    [0.0, 0.7, 0.5, 0.3],
    [0.0, 0.6, 0.5, 0.3],
    [0.0, 0.5, 0.5, 0.3],
    [0.0, 0.4, 0.5, 0.3],
    [0.0, 0.3, 0.5, 0.3]
  ]}'</pre>
    <button onclick="copyToClipboard('curl-predict')" class="absolute top-2 right-2 px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-xs text-gray-300 opacity-0 group-hover:opacity-100 transition">Copy</button>
  </div>
  
  <button onclick="testPredict()" class="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-white text-sm font-medium transition flex items-center gap-2">
    <span>▶</span> Send Request
  </button>
  
  <div id="predict-output" class="mt-4 hidden">
    <div class="flex items-center gap-2 mb-2">
      <span id="predict-status" class="px-2 py-0.5 rounded text-xs font-bold bg-gray-700 text-gray-300">--</span>
      <span id="predict-latency" class="text-xs text-gray-500 font-mono">0ms</span>
    </div>
    <pre id="predict-json" class="text-xs bg-black/70 p-3 rounded border border-gray-700 font-mono text-green-400 overflow-x-auto"></pre>
  </div>
</div>

<!-- /regime -->
<div class="border border-gray-700 rounded-lg bg-gray-800/30 p-6">
  <div class="flex items-center gap-3 mb-3">
    <span class="px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 text-xs font-bold">POST</span>
    <h3 class="text-white font-semibold">/regime</h3>
  </div>
  <p class="text-sm text-gray-400 mb-4">Determine market regime using HMM.</p>
  
  <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-2">
    <div>
      <label class="block text-xs font-bold text-gray-400 mb-1 uppercase">Realized Volatility</label>
      <input class="w-full bg-black/50 border border-gray-700 rounded p-2 text-sm text-blue-300 font-mono focus:border-amber-500 outline-none" id="input-vol" step="0.01" type="number" value="0.015"/>
    </div>
    <div>
      <label class="block text-xs font-bold text-gray-400 mb-1 uppercase">Autocorrelation</label>
      <input class="w-full bg-black/50 border border-gray-700 rounded p-2 text-sm text-blue-300 font-mono focus:border-amber-500 outline-none" id="input-auto" step="0.01" type="number" value="-0.1"/>
    </div>
  </div>
  <p class="text-xs text-gray-500 mb-4 italic">(Hint: Try Volatility &lt; 0.0018 for Volatile, 0.005 for Trending, and &gt; 0.009 for Mean-Reverting)</p>
  
  <div class="relative group mb-4">
    <pre id="curl-regime" class="text-xs bg-black/50 p-3 rounded border border-gray-700 font-mono text-blue-300 whitespace-pre-wrap break-all">
curl -X POST https://hemanthnaidug-alphalob.hf.space/regime \
  -H "Content-Type: application/json" \
  -d '{"realized_vol": 0.015, "autocorrelation": -0.1}'</pre>
    <button onclick="copyToClipboard('curl-regime')" class="absolute top-2 right-2 px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-xs text-gray-300 opacity-0 group-hover:opacity-100 transition">Copy</button>
  </div>
  
  <button onclick="testRegime()" class="px-4 py-2 bg-amber-600 hover:bg-amber-500 rounded text-white text-sm font-medium transition flex items-center gap-2">
    <span>▶</span> Detect Regime
  </button>
  
  <div id="regime-output" class="mt-4 hidden">
    <div class="flex items-center gap-2 mb-2">
      <span id="regime-status" class="px-2 py-0.5 rounded text-xs font-bold bg-gray-700 text-gray-300">--</span>
      <span id="regime-latency" class="text-xs text-gray-500 font-mono">0ms</span>
    </div>
    <pre id="regime-json" class="text-xs bg-black/70 p-3 rounded border border-gray-700 font-mono text-green-400 overflow-x-auto"></pre>
  </div>
</div>
</section>
</main>
<!-- Footer -->
<footer class="mt-16 pt-8 pb-8 border-t border-gray-800 text-center text-sm text-gray-500 w-full px-4">
  <p>Built by <strong class="text-gray-300">Hemanth Naidu Gokavalasa</strong> · Dual Degree(B Tech + M Tech) in Mathematics and Computing Technology · National Institute of Technology Patna</p>
  <div class="mt-2 space-x-4">
    <a href="https://github.com/GokavalasaHemanthNaidu/AlphaLOB" class="text-blue-400 hover:underline">GitHub</a>
    <a href="https://hemanthnaidug-alphalob.hf.space" class="text-blue-400 hover:underline">Hugging Face</a>
  </div>
</footer>
<script>
        // API Interactions
        const API_BASE = 'https://hemanthnaidug-alphalob.hf.space';

        function copyToClipboard(elementId) {
            const text = document.getElementById(elementId).textContent.trim();
            navigator.clipboard.writeText(text);
        }

        window.addEventListener('DOMContentLoaded', async () => {
            try {
                const res = await fetch('/health', { signal: AbortSignal.timeout(3000) });
                if (!res.ok) throw new Error('Not OK');
                const statusEl = document.getElementById('api-status');
                statusEl.classList.remove('text-amber-400');
                statusEl.classList.add('text-emerald-400');
                statusEl.textContent = '● API Online';
            } catch {
                const statusEl = document.getElementById('api-status');
                statusEl.classList.add('text-amber-400');
                statusEl.textContent = '● Waking up...';
            }
        });

        async function executeRequest(endpoint, outputPrefix, fetchArgs = {}) {
            const out = document.getElementById(`${outputPrefix}-output`);
            const status = document.getElementById(`${outputPrefix}-status`);
            const latency = document.getElementById(`${outputPrefix}-latency`);
            const json = document.getElementById(`${outputPrefix}-json`);
            
            out.classList.remove('hidden');
            status.textContent = 'LOADING';
            status.className = 'px-2 py-0.5 rounded text-xs font-bold bg-yellow-500/20 text-yellow-400';
            json.textContent = 'Waiting for response...';
            
            const t0 = performance.now();
            try {
                const res = await fetch(API_BASE + endpoint, fetchArgs);
                const ms = Math.round(performance.now() - t0);
                
                status.textContent = res.status;
                status.className = res.ok 
                    ? 'px-2 py-0.5 rounded text-xs font-bold bg-emerald-500/20 text-emerald-400'
                    : 'px-2 py-0.5 rounded text-xs font-bold bg-red-500/20 text-red-400';
                latency.textContent = `${ms}ms`;
                
                const data = await res.json();
                json.textContent = JSON.stringify(data, null, 2);
            } catch (e) {
                status.textContent = 'ERR';
                status.className = 'px-2 py-0.5 rounded text-xs font-bold bg-red-500/20 text-red-400';
                latency.textContent = '--';
                json.textContent = `Error: ${e.message}\n\nIf the Space was asleep, wait 3 seconds and retry.`;
            }
        }

        function testHealth() {
            executeRequest('/health', 'health');
        }

        function testPredict() {
            const lob_snapshot = Array.from({length: 10}, () => 
                [Math.random() * 0.2 - 0.1, Math.random() * 0.1, 100 + Math.random(), 100 + Math.random()]
            );
            executeRequest('/predict', 'predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ lob_snapshot })
            });
        }

        function testRegime() {
            const vol = parseFloat(document.getElementById('input-vol').value);
            const auto = parseFloat(document.getElementById('input-auto').value);
            executeRequest('/regime', 'regime', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ realized_vol: vol, autocorrelation: auto })
            });
        }
    </script>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return LANDING_HTML


@app.get("/health")
async def health():
    """Health check endpoint. Returns 200 if models are loaded."""
    return {
        "status": "ok",
        "model":  "LOBTransformer",
        "version": "1.0.0",
        "onnx_loaded": ort_session is not None,
        "hmm_loaded":  hmm_model is not None,
        "regime_names": list(hmm_model.regime_names.values()),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(body: LOBSnapshot):
    """
    Accepts a single LOB snapshot (10 levels × 4 features).
    Returns 5 model outputs: directional probs (3 horizons) + spread + vol imbalance.
    """
    snapshot = body.lob_snapshot

    # Validate input shape
    if len(snapshot) != 10:
        raise HTTPException(status_code=422, detail=f"Expected 10 LOB levels, got {len(snapshot)}")
    for i, row in enumerate(snapshot):
        if len(row) != 4:
            raise HTTPException(status_code=422, detail=f"Level {i} has {len(row)} features, expected 4")

    # Convert to numpy and sanitize NaN/Inf BEFORE ONNX inference
    X = np.array(snapshot, dtype=np.float32).reshape(1, 10, 4)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ONNX inference
    outputs = ort_session.run(None, {"lob_snapshot": X})

    # Extract outputs (handle both (1,2) and (1,) shapes)
    def safe_prob(arr, idx):
        """Safely extract probability and ensure it is finite."""
        v = float(arr[0, idx]) if arr.ndim == 2 else float(arr[0])
        return v if np.isfinite(v) else 0.5

    def safe_scalar(arr):
        v = float(arr[0]) if arr.ndim == 1 else float(arr[0, 0])
        return v if np.isfinite(v) else 0.0

    return PredictResponse(
        dir_5s_prob_up    = safe_prob(outputs[0], 1),
        dir_30s_prob_up   = safe_prob(outputs[1], 1),
        dir_5min_prob_up  = safe_prob(outputs[2], 1),
        spread_compress   = safe_scalar(outputs[3]),
        vol_imbalance     = safe_scalar(outputs[4]),
    )


@app.post("/regime", response_model=RegimeResponse)
async def regime(body: RegimeInput):
    """
    Accepts realized_vol + autocorrelation.
    Returns: current market regime label + state probabilities.
    """
    # Build HMM input array
    X_hmm = np.array([[body.realized_vol, body.autocorrelation]], dtype=np.float64)
    X_hmm = np.nan_to_num(X_hmm, nan=0.0, posinf=0.0, neginf=0.0)

    # Predict state and probabilities
    state_id  = int(hmm_model.predict(X_hmm)[0])
    proba     = hmm_model.predict_proba(X_hmm)[0]   # (n_states,) array

    regime_name = hmm_model.regime_names[state_id]
    proba_dict  = {hmm_model.regime_names[i]: float(proba[i]) for i in range(len(proba))}

    return RegimeResponse(
        regime=regime_name,
        state_id=state_id,
        probabilities=proba_dict,
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
