# M4 Pro Inference: SGLang with MLX

High-throughput LLM inference on Apple M4 Pro (64GB unified memory) using SGLang's native MLX backend. Supports **256K context** with FP8 quantized KV cache.

## Quick Start

```bash
# Setup
./scripts/setup.sh

# Run any model
./scripts/launch.sh coder-30b              # Coder-30B MoE — fastest, best for agents
./scripts/launch.sh devstral               # Devstral-24B — best all-round
./scripts/launch.sh gemma4                 # Gemma 4 26B MoE
./scripts/launch.sh qwen35                 # Qwen3.5-27B DeltaNet
./scripts/launch.sh qwen3-32b             # Qwen3-32B Dense
./scripts/launch.sh qwen3-moe             # Qwen3-30B MoE
./scripts/launch.sh coder-next            # Coder-Next 80B — largest

# 256K context (FP8 KV cache)
./scripts/launch.sh coder-30b --context-length 262144 --kv-cache fp8

# TurboQuant for models with large KV heads
./scripts/launch.sh gemma4-31b --context-length 262144 --kv-cache turboquant
```

## Model Support

| Model | Type | Weights | 1-user tok/s | 256K tok/s | Launch |
|-------|------|:-------:|:------------:|:----------:|--------|
| Coder-30B | MoE (3B active) | 16 GB | 68.4 | 3.2 | `coder-30b` |
| Gemma 4 26B | MoE (4B active) | 15 GB | 58.8 | 1.5 | `gemma4` |
| Coder-Next 80B | MoE+DeltaNet | 42 GB | 55.3 | mem limited | `coder-next` |
| Devstral-24B | Dense | 14 GB | 17.0 | 1.8 | `devstral` |
| Qwen3.5-27B | DeltaNet hybrid | 15 GB | 14.5 | pending | `qwen35` |
| Qwen3-32B | Dense | 18 GB | ~12 | pending | `qwen3-32b` |
| Qwen3-30B | MoE (3B active) | 16 GB | ~68 | pending | `qwen3-moe` |
| Gemma 4 31B | Dense | 17 GB | 12.5 | 16K max | `gemma4-31b` |

All models 4-bit MLX quantized from `mlx-community/` on HuggingFace. MoE models recommended for agentic workloads.

### KV Cache Modes

| Mode | Memory | Use Case |
|------|:------:|----------|
| `auto` (fp16) | 1x | Default, short context |
| `fp8` | ~0.5x | Long context (256K), most models |
| `turboquant` | ~0.28x | Models with large KV heads (Gemma 4 31B) |

```bash
./scripts/launch.sh <model> --kv-cache fp8          # MXFP8, ~2x savings
./scripts/launch.sh <model> --kv-cache turboquant    # Affine 4-bit, ~3.5x savings
```

## Performance

> M4 Pro 64GB, SGLang + MLX, `sglang.bench_serving`, FP8 KV cache

### Context Length Scaling

![Context vs Decode Speed](benchmarks/all_models_context.png)

### Devstral-24B (Dense, FP8 KV)

| Context | TPOT (ms) | tok/s | TTFT |
|:-------:|:---------:|:-----:|:----:|
| 128 | 58.7 | 17.0 | 0.5s |
| 4K | 62.1 | 16.1 | 8s |
| 16K | 108.8 | 9.2 | 1.8m |
| 32K | 169.2 | 5.9 | 4.3m |
| 64K | 295.7 | 3.4 | 10.9m |
| **256K** | **540.9** | **1.8** | **29.3m** |

30% KV pool usage at 256K. Peak concurrent: 39.2 tok/s @8.

### Coder-30B MoE (FP8 KV)

| Context | TPOT (ms) | tok/s | TTFT |
|:-------:|:---------:|:-----:|:----:|
| 128 | 14.6 | 68.4 | 0.2s |
| 4K | 18.1 | 55.2 | 1.3s |
| 16K | 46.4 | 21.6 | 26s |
| 32K | 84.7 | 11.8 | 1.4m |
| 64K | 158.3 | 6.3 | 5.0m |
| 128K | 307.6 | 3.3 | 19.5m |
| **256K** | **309.2** | **3.2** | **19.5m** |

20% KV pool at 256K. Peak concurrent: 107.4 tok/s @8. Best model for 256K agentic workloads.

### Gemma 4 26B MoE (FP8 KV)

| Context | TPOT (ms) | tok/s | TTFT |
|:-------:|:---------:|:-----:|:----:|
| 128 | 17.0 | 58.8 | 0.2s |
| 4K | 22.1 | 45.4 | 1.6s |
| 16K | 82.8 | 12.1 | 30s |
| 32K | 165.6 | 6.0 | 1.4m |
| 64K | 333.1 | 3.0 | 4.8m |
| 128K | 672.6 | 1.5 | 17.8m |
| **256K** | **674.9** | **1.5** | **17.7m** |

48% KV pool at 256K — tightest fit.

### Throughput Scaling

![Throughput Scaling](benchmarks/all_models_concurrency.png)

| Model | Conc=1 | Conc=4 | Conc=8 |
|-------|:------:|:------:|:------:|
| Coder-30B | 82.6 | 97.1 | **107.4** |
| Devstral-24B | 27.0 | 30.4 | 39.2 |

## 256K Context: How It Works

Three changes enable 256K context on 64GB Apple Silicon:

1. **FP8 KV cache** (`--kv-cache fp8`) — MXFP8 quantization stores KV in 8-bit, halving memory vs float16. Implemented in `kv_cache/kv_quant.py`.

2. **Health check timeout** (`SGLANG_HEALTH_CHECK_TIMEOUT=120`) — SGLang's default 20s health check timeout is too short for chunked prefill at 64K+ context (each 4K chunk takes 50-80s). Set automatically in `common.sh`.

3. **DeltaNet hybrid support** — Models with mixed attention types (standard + linear/DeltaNet) use per-layer cache detection. Standard attention layers get FP8 KV cache; DeltaNet layers use native `ArraysCache`.

### Memory at 256K (FP8 KV)

| Model | Weights | KV Pool | Pool Usage | Fits 64GB |
|-------|:-------:|:-------:|:----------:|:---------:|
| Coder-30B (MoE) | 16 GB | 627K slots | 20% | Yes |
| Devstral-24B | 14 GB | 423K slots | 30% | Yes |
| Gemma 4 26B (MoE) | 15 GB | 268K slots | 48% | Yes |
| Coder-Next 80B | 42 GB | 82K slots | — | No (weights) |

## Known Issues

- **Greedy sampling only** — MLX backend uses `mx.argmax`; temperature/top-p not yet supported
- **VLM warmup crash** — Devstral detected as VLM; use `--skip-server-warmup` (set automatically)
- **HDMI display blackout** — Brief screen blank when server starts heavy GPU load (M4 Pro HDMI issue, not a crash)

## Patches

5 patches on top of SGLang `main` at commit `1f8df9705`:

| Patch | Purpose |
|-------|---------|
| **001-mlx-radix-cache** | Radix cache + KV pool for MLX |
| **002-mps-backend-defaults** | Disable CUDA graph, force torch_native attention on MPS |
| **003-mlx-skip-quantization-check** | Skip quant verification for MLX models |
| **004-mlx-lifecycle-and-hybrid-fixes** | Request lifecycle + DeltaNet/Mamba support |
| **005-mlx-attn-wrapper-varargs** | Attention wrapper fix for Devstral |

## Setup

```bash
./scripts/setup.sh
```

| Component | Version |
|-----------|---------|
| SGLang | main + 5 patches |
| MLX | 0.31.1 |
| mlx-lm | 0.31.2 |
| PyTorch | 2.9.1 (MPS) |
| Python | 3.12 |

### Quantization

MLX uses its own quantization format. Pre-quantized models available from `mlx-community/` on HuggingFace, or convert yourself:

```bash
python -m mlx_lm.convert --hf-path <model> --mlx-path <output> -q --q-bits 4
```

AWQ/GPTQ models from CUDA/ROCm are **not compatible** — MLX requires its own format.

## Test System

```
Mac mini (Mac16,11)
Apple M4 Pro — 14-core CPU, 20-core GPU
64 GB unified memory (LPDDR5, ~273 GB/s)
macOS 26.2
```
