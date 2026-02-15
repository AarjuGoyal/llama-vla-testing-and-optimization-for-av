# Llama 3.2-3B Inference Optimization for Autonomous Driving VLA Systems

Optimized LLM inference achieving **2.9x throughput improvement** through profiling-driven batch processing, validated for real-time autonomous vehicle applications.

## Setup
1. Install llama-models:
```bash
   git clone https://github.com/meta-llama/llama-models.git
   cd llama-models
   pip install -e .
```

2. Download Llama 3.2-3B model (requires Meta approval):
   - Visit https://www.llama.com/llama-downloads/
   - Follow download instructions
   - Place in `~/.llama/checkpoints/Llama3.2-3B/`

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Run benchmark:
```bash
   torchrun --nproc_per_node 1 scripts/benchmark.py
```

## Key Results

- **Throughput:** 67 → 194 tokens/sec (2.9x improvement)
- **Latency:** 14.9ms → 5.16ms per token (2.9x faster)
- **GPU Utilization:** 53% → 58% (activated Tensor Cores)
- **Real-time feasibility:** <25ms for typical VLA reasoning chains

## Optimization Summary

| Configuration | Throughput (tok/s) | Latency (ms/tok) | Speedup |
|---------------|-------------------|------------------|---------|
| Baseline (batch=1) | 67 | 14.9 | 1.0x |
| Batched (batch=4) | 194 | 5.16 | 2.9x |

![Performance Comparison](results/comparison_chart.png)

## Technical Approach

1. **Baseline profiling** with NVIDIA Nsight Systems
2. **Identified bottleneck:** GEMV operations on general CUDA cores
3. **Applied optimization:** Batch size increase to trigger WMMA (Tensor Core) kernels
4. **Validated results:** Re-profiled to confirm kernel transition

## Autonomous Driving Context

This work validates computational feasibility of LLM reasoning in Vision-Language-Action (VLA) systems for autonomous vehicles...
