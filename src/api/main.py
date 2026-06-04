"""
AlphaLOB Production FastAPI Service
Routes: /health, /predict, /regime
"""
import os
import numpy as np
import onnxruntime as ort
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

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
<style>
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
            border-right: 2px solid theme('colors.primary');
            white-space: nowrap; 
            margin: 0;
            animation: typing 2s steps(40, end), blink-caret .75s step-end infinite;
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
<body class="font-body-md text-body-md min-h-screen flex flex-col antialiased">
<!-- Top Navigation (Generated from JSON) -->
<header class="bg-surface dark:bg-surface text-primary dark:text-primary docked full-width top-0 sticky border-b border-outline-variant dark:border-outline-variant flat no shadows flex justify-between items-center w-full px-container-margin py-stack-compact max-w-full z-50 bg-surface/95 backdrop-blur-sm">
<div class="flex items-center gap-gutter">
<span class="text-title-sm font-title-sm font-bold text-on-surface dark:text-on-surface">AlphaLOB</span>
</div>
<nav class="hidden md:flex gap-container-margin items-center">
<a class="text-on-surface-variant dark:text-on-surface-variant hover:text-primary dark:hover:text-primary transition-colors duration-200" href="#documentation">Documentation</a>
<a class="text-primary dark:text-primary border-b-2 border-primary pb-1" href="#api">API</a>
<a class="text-on-surface-variant dark:text-on-surface-variant hover:text-primary dark:hover:text-primary transition-colors duration-200" href="https://github.com/GokavalasaHemanthNaidu/AlphaLOB" target="_blank">GitHub</a>
</nav>
<div class="flex items-center">
<button class="text-primary dark:text-primary hover:text-primary dark:hover:text-primary transition-colors duration-200">
<span class="material-symbols-outlined" style="font-variation-settings: 'FILL' 0;">sensors</span>
</button>
</div>
</header>
<main class="flex-grow p-container-margin md:p-8 space-y-12">
<!-- Section 1: Hero -->
<section class="flex flex-col items-start gap-stack-default max-w-7xl mx-auto w-full">
<div class="flex items-center gap-gutter flex-wrap mb-2">
<span class="px-2 py-1 bg-secondary/15 border border-secondary text-secondary font-label-caps text-label-caps rounded-DEFAULT flex items-center gap-1">
<span class="w-1.5 h-1.5 rounded-full bg-secondary animate-pulse"></span>
                    LIVE
                </span>
<span class="px-2 py-1 bg-primary/15 border border-primary text-primary font-label-caps text-label-caps rounded-DEFAULT">
                    ONNX Runtime
                </span>
<span class="px-2 py-1 bg-tertiary/15 border border-tertiary text-tertiary font-label-caps text-label-caps rounded-DEFAULT">
                    Zero Look-Ahead Bias
                </span>
</div>
<h1 class="font-display-lg text-display-lg text-on-surface typing-effect inline-block pr-2">
                &gt; AlphaLOB_
            </h1>
<p class="font-headline-md text-headline-md text-on-surface-variant mt-stack-compact">
                Real-Time Limit Order Book Alpha Signals
            </p>
</section>
<!-- Scientific Integrity Banner -->
<section class="max-w-7xl mx-auto w-full">
<div class="border border-tertiary bg-tertiary/5 p-4 rounded-DEFAULT flex items-start gap-4">
<span class="material-symbols-outlined text-tertiary mt-1">warning</span>
<p class="text-body-sm text-on-surface-variant">
<strong class="text-tertiary font-medium">Scientific Integrity:</strong> 51.25% is the theoretical maximum accuracy for a WOFI-return correlation of 0.044. Any model claiming &gt;55% on this synthetic data would indicate data leakage or label injection. This result validates that the pipeline is mathematically sound and leak-free.
            </p>
</div>
</section>
<!-- Section 2: Metrics Grid -->
<section class="max-w-7xl mx-auto w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-gutter">
<!-- Accuracy -->
<div class="bg-surface-container border border-surface-container-highest p-stack-default rounded-DEFAULT glow-hover transition-colors">
<div class="flex items-center justify-between mb-stack-compact">
<span class="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">Predictive Edge</span>
<span class="material-symbols-outlined text-primary text-sm" style="font-variation-settings: 'FILL' 0;">analytics</span>
</div>
<div class="font-headline-md text-headline-md text-on-surface mb-1 flex items-center">
<span class="text-secondary text-sm mr-1">▲</span> 51.25%
                </div>
<div class="font-body-sm text-body-sm text-outline tooltip cursor-help border-b border-dashed border-outline">
                    Honest Ceiling
                    <span class="tooltiptext">Theoretical ceiling for highly stochastic process (ρ=0.044) without look-ahead bias.</span>
</div>
</div>
<!-- Latency -->
<div class="bg-surface-container border border-surface-container-highest p-stack-default rounded-DEFAULT glow-hover transition-colors relative overflow-hidden">
<div class="flex items-center justify-between mb-stack-compact">
<span class="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">Inference Latency</span>
<span class="material-symbols-outlined text-primary text-sm" style="font-variation-settings: 'FILL' 0;">speed</span>
</div>
<div class="font-headline-md text-headline-md text-on-surface mb-1">
                    5.47ms
                </div>
<div class="font-body-sm text-body-sm text-outline tooltip cursor-help border-b border-dashed border-outline mb-3">
                    p99 Threshold
                    <span class="tooltiptext">p99 latency measured on standard CPU. Target SLA is &lt;15ms.</span>
</div>
<!-- Sparkline Chart -->
<div class="absolute bottom-0 left-0 w-full h-10 flex items-end px-2 pb-1 gap-1 opacity-80">
<div class="w-1/4 bg-primary/40 rounded-t-sm tooltip" style="height: 10%;">
<span class="tooltiptext mb-6">p50: 0.82ms</span>
</div>
<div class="w-1/4 bg-primary/60 rounded-t-sm tooltip" style="height: 15%;">
<span class="tooltiptext mb-6">p90: 1.18ms</span>
</div>
<div class="w-1/4 bg-primary/80 rounded-t-sm tooltip" style="height: 20%;">
<span class="tooltiptext mb-6">p95: 1.29ms</span>
</div>
<div class="w-1/4 bg-primary rounded-t-sm tooltip" style="height: 80%;">
<span class="tooltiptext mb-6">p99: 5.47ms</span>
</div>
<!-- 15ms target line -->
<div class="absolute top-2 left-0 w-full border-t border-error border-dashed pointer-events-none"></div>
</div>
</div>
<!-- Architecture -->
<div class="bg-surface-container border border-surface-container-highest p-stack-default rounded-DEFAULT glow-hover transition-colors">
<div class="flex items-center justify-between mb-stack-compact">
<span class="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">Core Engine</span>
<span class="material-symbols-outlined text-primary text-sm" style="font-variation-settings: 'FILL' 0;">memory</span>
</div>
<div class="font-headline-md text-headline-md text-on-surface mb-1">
                    6-L Transformer
                </div>
<div class="font-body-sm text-body-sm text-outline tooltip cursor-help border-b border-dashed border-outline">
                    Attention Mech
                    <span class="tooltiptext">8 Attention Heads, d_model=64. Optimized for sequence modeling.</span>
</div>
</div>
<!-- Regime -->
<div class="bg-surface-container border border-surface-container-highest p-stack-default rounded-DEFAULT glow-hover transition-colors">
<div class="flex items-center justify-between mb-stack-compact">
<span class="font-label-caps text-label-caps text-on-surface-variant uppercase tracking-wider">Market State</span>
<span class="material-symbols-outlined text-primary text-sm" style="font-variation-settings: 'FILL' 0;">waves</span>
</div>
<div class="font-headline-md text-headline-md text-on-surface mb-1">
                    3-State HMM
                </div>
<div class="font-body-sm text-body-sm text-outline tooltip cursor-help border-b border-dashed border-outline">
                    Context Switcher
                    <span class="tooltiptext">Hidden Markov Model conditioning on realized volatility and autocorrelation.</span>
</div>
</div>
</section>
<!-- Pipeline Visualization -->
<section id="documentation" class="max-w-7xl mx-auto w-full">
<h2 class="text-title-sm font-semibold text-on-surface mb-4">Architecture Pipeline</h2>
<div class="bg-surface-container border border-surface-container-highest p-6 rounded-DEFAULT overflow-x-auto">
<div class="flex flex-col md:flex-row items-center justify-between min-w-[800px]">
<!-- Step 1 -->
<div class="flex flex-col items-center gap-2 group tooltip">
<div class="w-12 h-12 rounded-full border border-primary bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary group-hover:text-surface transition-colors">
<span class="font-data-mono font-bold">01</span>
</div>
<span class="text-body-sm font-medium text-on-surface-variant whitespace-nowrap">Synthetic LOB</span>
<span class="tooltiptext">Data generation mimicking L2 order book dynamics.</span>
</div>
<div class="flow-line"></div>
<!-- Step 2 -->
<div class="flex flex-col items-center gap-2 group tooltip">
<div class="w-12 h-12 rounded-full border border-primary bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary group-hover:text-surface transition-colors">
<span class="font-data-mono font-bold">02</span>
</div>
<span class="text-body-sm font-medium text-on-surface-variant whitespace-nowrap">Feature Eng</span>
<span class="tooltiptext">Extracting WOFI, spread, imbalances.</span>
</div>
<div class="flow-line"></div>
<!-- Step 3 -->
<div class="flex flex-col items-center gap-2 group tooltip">
<div class="w-12 h-12 rounded-full border border-secondary bg-secondary/10 flex items-center justify-center text-secondary group-hover:bg-secondary group-hover:text-surface transition-colors">
<span class="font-data-mono font-bold">03</span>
</div>
<span class="text-body-sm font-medium text-on-surface-variant whitespace-nowrap">LOBTransformer</span>
<span class="tooltiptext">Deep attention network for temporal patterns.</span>
</div>
<div class="flow-line"></div>
<!-- Step 4 -->
<div class="flex flex-col items-center gap-2 group tooltip">
<div class="w-12 h-12 rounded-full border border-tertiary bg-tertiary/10 flex items-center justify-center text-tertiary group-hover:bg-tertiary group-hover:text-surface transition-colors">
<span class="font-data-mono font-bold">04</span>
</div>
<span class="text-body-sm font-medium text-on-surface-variant whitespace-nowrap">Regime HMM</span>
<span class="tooltiptext">Contextualizing based on market regime.</span>
</div>
<div class="flow-line"></div>
<!-- Step 5 -->
<div class="flex flex-col items-center gap-2 group tooltip">
<div class="w-12 h-12 rounded-full border border-primary bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary group-hover:text-surface transition-colors">
<span class="font-data-mono font-bold">05</span>
</div>
<span class="text-body-sm font-medium text-on-surface-variant whitespace-nowrap">WF Backtest</span>
<span class="tooltiptext">Walk-forward validation without look-ahead bias.</span>
</div>
<div class="flow-line"></div>
<!-- Step 6 -->
<div class="flex flex-col items-center gap-2 group tooltip">
<div class="w-12 h-12 rounded-full border border-secondary bg-secondary/10 flex items-center justify-center text-secondary group-hover:bg-secondary group-hover:text-surface transition-colors">
<span class="font-data-mono font-bold">06</span>
</div>
<span class="text-body-sm font-medium text-on-surface-variant whitespace-nowrap">ONNX Deploy</span>
<span class="tooltiptext">Low-latency inference via ONNX runtime.</span>
</div>
</div>
</div>
</section>
<!-- Interactive API Documentation -->
<section class="max-w-7xl mx-auto w-full space-y-6" id="api">
<h2 class="text-title-sm font-semibold text-on-surface">API Documentation &amp; Sandbox</h2>
<!-- /health -->
<div class="bg-surface-container border border-surface-container-highest p-6 rounded-DEFAULT">
<div class="flex items-center gap-3 mb-4">
<span class="px-2 py-1 bg-secondary/20 text-secondary font-bold text-xs rounded">GET</span>
<h3 class="text-body-md font-bold text-on-surface font-data-mono">/health</h3>
</div>
<p class="text-body-sm text-on-surface-variant mb-4">Check API and model readiness.</p>
<div class="bg-surface p-4 rounded-DEFAULT border border-outline-variant mb-4 font-data-mono text-sm overflow-x-auto">
<code class="text-primary" id="curl-health">curl -X GET https://hemanthnaidug-alphalob.hf.space/health</code>
</div>
<div class="flex gap-4">
<button class="px-4 py-2 bg-primary/10 border border-primary text-primary rounded-DEFAULT hover:bg-primary/20 transition-colors font-medium text-sm flex items-center gap-2" onclick="testHealth()">
<span class="material-symbols-outlined text-sm">play_arrow</span> Test
                </button>
</div>
<div class="hidden mt-4" id="health-result-container">
<pre class="bg-surface p-4 rounded-DEFAULT border border-outline-variant text-xs font-data-mono text-on-surface overflow-x-auto" id="health-result"></pre>
</div>
</div>
<!-- /predict -->
<div class="bg-surface-container border border-surface-container-highest p-6 rounded-DEFAULT">
<div class="flex items-center gap-3 mb-4">
<span class="px-2 py-1 bg-primary/20 text-primary font-bold text-xs rounded">POST</span>
<h3 class="text-body-md font-bold text-on-surface font-data-mono">/predict</h3>
</div>
<p class="text-body-sm text-on-surface-variant mb-4">Generate alpha signals from order book features. Requires a sequence of 10 timesteps with 4 features (WOFI, spread, microprice_mid, vwap_mid).</p>
<div class="bg-surface p-4 rounded-DEFAULT border border-outline-variant mb-4 font-data-mono text-sm overflow-x-auto whitespace-pre">
<code class="text-primary">curl -X POST https://hemanthnaidug-alphalob.hf.space/predict \
-H "Content-Type: application/json" \
-d '{"lob_snapshot": [[0.0,1.2,0.5,0.3], ... (10 steps) ]}'</code>
</div>
<div class="flex gap-4">
<button class="px-4 py-2 bg-primary/10 border border-primary text-primary rounded-DEFAULT hover:bg-primary/20 transition-colors font-medium text-sm flex items-center gap-2" onclick="testPredict()">
<span class="material-symbols-outlined text-sm">play_arrow</span> Try It
                </button>
</div>
<div class="hidden mt-4" id="predict-result-container">
<pre class="bg-surface p-4 rounded-DEFAULT border border-outline-variant text-xs font-data-mono text-on-surface overflow-x-auto" id="predict-result"></pre>
</div>
</div>
<!-- /regime -->
<div class="bg-surface-container border border-surface-container-highest p-6 rounded-DEFAULT">
<div class="flex items-center gap-3 mb-4">
<span class="px-2 py-1 bg-primary/20 text-primary font-bold text-xs rounded">POST</span>
<h3 class="text-body-md font-bold text-on-surface font-data-mono">/regime</h3>
</div>
<p class="text-body-sm text-on-surface-variant mb-4">Determine market regime using HMM.</p>
<div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
<div>
<label class="block text-label-caps text-on-surface-variant mb-1">Realized Volatility</label>
<input class="w-full bg-surface border border-outline-variant rounded-DEFAULT px-3 py-2 text-sm text-on-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none" id="input-vol" step="0.01" type="number" value="0.015"/>
</div>
<div>
<label class="block text-label-caps text-on-surface-variant mb-1">Autocorrelation</label>
<input class="w-full bg-surface border border-outline-variant rounded-DEFAULT px-3 py-2 text-sm text-on-surface focus:border-primary focus:ring-1 focus:ring-primary outline-none" id="input-auto" step="0.01" type="number" value="-0.1"/>
</div>
</div>
<div class="flex gap-4">
<button class="px-4 py-2 bg-primary/10 border border-primary text-primary rounded-DEFAULT hover:bg-primary/20 transition-colors font-medium text-sm flex items-center gap-2" onclick="testRegime()">
<span class="material-symbols-outlined text-sm">play_arrow</span> Run Inference
                </button>
</div>
<div class="hidden mt-4" id="regime-result-container">
<pre class="bg-surface p-4 rounded-DEFAULT border border-outline-variant text-xs font-data-mono text-on-surface overflow-x-auto" id="regime-result"></pre>
</div>
</div>
</section>
</main>
<!-- Footer (Generated from JSON) -->
<footer class="bg-surface-container-low dark:bg-surface-container-low border-t border-outline-variant dark:border-outline-variant flat no shadows w-full px-container-margin py-stack-default flex flex-col md:flex-row justify-between items-center gap-stack-default mt-auto">
<div class="text-body-sm font-body-sm text-primary dark:text-primary">
            Built by Hemanth Naidu | 4th Year CSE | Quantitative Finance &amp; Deep Learning
        </div>
<div class="flex items-center gap-container-margin">
<a class="text-label-caps font-label-caps text-on-surface-variant dark:text-on-surface-variant hover:text-on-surface dark:hover:text-on-surface transition-colors" href="https://github.com/GokavalasaHemanthNaidu/AlphaLOB" target="_blank">GitHub</a>
<a class="text-label-caps font-label-caps text-on-surface-variant dark:text-on-surface-variant hover:text-on-surface dark:hover:text-on-surface transition-colors" href="https://hemanthnaidug-alphalob.hf.space" target="_blank">Hugging Face Space</a>
</div>
</footer>
<script>
        // API Interactions
        const API_BASE = 'https://hemanthnaidug-alphalob.hf.space';

        function showLoading(elementId) {
            const container = document.getElementById(elementId + '-container');
            const resultEl = document.getElementById(elementId);
            container.classList.remove('hidden');
            resultEl.innerHTML = '<span class="animate-pulse">Executing request...</span>';
            resultEl.className = 'bg-surface p-4 rounded-DEFAULT border border-outline-variant text-xs font-data-mono text-outline overflow-x-auto';
        }

        function showResult(elementId, data, isError = false) {
            const resultEl = document.getElementById(elementId);
            resultEl.innerHTML = JSON.stringify(data, null, 2);
            if (isError) {
                resultEl.className = 'bg-error/10 p-4 rounded-DEFAULT border border-error text-xs font-data-mono text-error overflow-x-auto';
            } else {
                resultEl.className = 'bg-surface p-4 rounded-DEFAULT border border-outline-variant text-xs font-data-mono text-secondary overflow-x-auto';
            }
        }

        async function testHealth() {
            showLoading('health-result');
            try {
                const response = await fetch(`${API_BASE}/health`);
                const data = await response.json();
                if (!response.ok) { throw new Error(JSON.stringify(data)); }
                showResult('health-result', data);
            } catch (error) {
                showResult('health-result', { error: error.message }, true);
            }
        }

        async function testPredict() {
            showLoading('predict-result');
            
            // Dummy 10x4 feature matrix
            const lob_snapshot = Array.from({length: 10}, () => 
                [Math.random() * 0.2 - 0.1, Math.random() * 0.1, 100 + Math.random(), 100 + Math.random()]
            );

            try {
                const response = await fetch(`${API_BASE}/predict`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ lob_snapshot })
                });
                const data = await response.json();
                if (!response.ok) { throw new Error(JSON.stringify(data)); }
                showResult('predict-result', data);
            } catch (error) {
                showResult('predict-result', { error: error.message }, true);
            }
        }

        async function testRegime() {
            showLoading('regime-result');
            
            const vol = parseFloat(document.getElementById('input-vol').value);
            const auto = parseFloat(document.getElementById('input-auto').value);

            try {
                const response = await fetch(`${API_BASE}/regime`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ realized_vol: vol, autocorrelation: auto })
                });
                const data = await response.json();
                if (!response.ok) { throw new Error(JSON.stringify(data)); }
                showResult('regime-result', data);
            } catch (error) {
                showResult('regime-result', { error: error.message }, true);
            }
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
