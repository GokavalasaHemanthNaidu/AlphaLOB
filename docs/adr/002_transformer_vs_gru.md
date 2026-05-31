# ADR 002: Transformer vs. GRU for Limit Order Book (LOB) Modeling

## Status
Accepted

## Context
Predicting short-term price movements from Limit Order Book (LOB) data requires capturing temporal dependencies (how the book evolves over time) and cross-sectional dependencies (the relationship between bids, asks, prices, and volumes at a specific millisecond). The industry standards for sequence modeling in finance have traditionally been Recurrent Neural Networks (LSTMs or GRUs) or Temporal Convolutional Networks (TCNs). We evaluated using a **GRU (Gated Recurrent Unit)** versus a **Transformer** architecture.

## Decision
We elected to use a **Transformer with Cross-Attention** for LOB modeling, specifically exporting the final weights to an ONNX CPU graph for low-latency inference.

## Rationale
1. **Parallelizable Training**: HFT models require training on massive datasets (millions of ticks per day). GRUs process data sequentially, causing a significant bottleneck during backpropagation through time (BPTT). Transformers process the entire sequence window simultaneously, allowing us to fully saturate modern GPUs during training.
2. **Attention to Depth**: LOB data has strict hierarchical importance (Level 1 is usually more predictive than Level 10). By implementing spatial cross-attention across the order book levels, the Transformer can dynamically learn which levels matter during specific market regimes (e.g., ignoring spoofed volume at Level 10).
3. **Vanishing Gradients over Long Horizons**: While predicting 5-second horizons is easy for a GRU, our multi-task head also predicts 5-minute horizons. Transformers handle long-range dependencies significantly better than GRUs without gradient degradation.

## Consequences
- **Pro**: Faster convergence during training on large historical datasets.
- **Pro**: Better handling of simultaneous, multi-horizon predictions.
- **Con**: Transformers require a fixed context window and generally consume more memory than GRUs. To mitigate this for real-time inference, we stripped the PyTorch backend entirely and execute the optimized computational graph via `onnxruntime` on the CPU, achieving sub-millisecond latency.
