# M4 Pro Inference: SGLang with MLX

High-throughput LLM inference on Apple M4 Pro (64GB unified memory) using SGLang's native MLX backend.

## Known Issues

- **Greedy sampling only** — MLX backend uses `mx.argmax`; temperature, top-p, top-k not yet supported
- **Variable-length prefill** — Falls back to serial processing when sequences have different lengths
- **MPS 4GB tensor limit** — Tensors >1 GB kept on CPU to avoid `MPSTemporaryNDArray` crashes
- **Request cleanup bug** — Concurrent requests can crash with `KeyError` (fixed by patch 001, see [Patches](#patches))

## Quick Start

```bash
# 1. Setup: create venv, install SGLang + MLX deps, apply patches
./scripts/setup.sh

# 2. Run any model:
./scripts/launch.sh devstral            # Devstral-24B 4-bit — best all-round
./scripts/launch.sh coder-30b           # Coder-30B MoE 4-bit — best throughput
./scripts/launch.sh coder-next          # Coder-Next 80B 4-bit — largest model
./scripts/launch.sh gemma4              # Gemma 4 26B MoE 4-bit
./scripts/launch.sh qwen35              # Qwen3.5-27B 4-bit

# 3. Test quality
python scripts/eval/eval_comprehensive.py --port 23334 --parallel 4

# 4. Benchmark
python scripts/bench/bench_all_unified.py --name "Model Name" --port 23334
```

## Prerequisites

- Apple Silicon Mac (M1 or later) — tested on M4 Pro
- macOS with Xcode Command Line Tools (`xcode-select --install`)
- Python 3.12+ (system or Homebrew)
- ~100GB disk for models

## Model Support

All models run on SGLang with the MLX backend (`SGLANG_USE_MLX=1`). Models must be in MLX format — download from `mlx-community/` on HuggingFace or convert with `mlx_lm.convert`.

### Agent / coding workloads (single-user, max context)

| Model | Type | 4-bit size | Max context | Launch | Status |
|-------|------|:----------:|:----------:|:------:|:------:|
| Devstral-24B | Dense | ~14 GB | 32K | `launch.sh devstral` | Testing |
| Coder-30B | MoE (128 experts) | ~16 GB | 32K | `launch.sh coder-30b` | Testing |
| Gemma 4 26B | MoE (128 experts) | ~15 GB | 4K | `launch.sh gemma4` | Testing |
| Qwen3.5-27B | DeltaNet hybrid | ~15 GB | 32K | `launch.sh qwen35` | Testing |
| Coder-Next 80B | MoE+DeltaNet (512 experts) | ~42 GB | 8K | `launch.sh coder-next` | Testing |

All models are served as 4-bit MLX quantized. Benchmarks pending.

### Memory budget (64GB unified)

| Model | Weights | KV cache headroom | Concurrent users |
|-------|:-------:|:-----------------:|:----------------:|
| Devstral-24B | ~14 GB | ~41 GB | Many |
| Coder-30B MoE | ~16 GB | ~39 GB | Many |
| Qwen3.5-27B | ~15 GB | ~40 GB | Many |
| Gemma 4 26B MoE | ~15 GB | ~40 GB | Many |
| Coder-Next-80B | ~42 GB | ~13 GB | 1-2 |

~9 GB reserved for macOS + system overhead.

### Expected performance

The M4 Pro has ~273 GB/s memory bandwidth. Autoregressive decode is memory-bandwidth-bound — each token requires reading all model weights once.

| Model | Estimated tok/s (single user) |
|-------|:----------------------------:|
| Devstral-24B 4-bit | 40-60 |
| Coder-30B 4-bit | 30-50 |
| Qwen3.5-27B 4-bit | 30-50 |
| Gemma 4 26B 4-bit | 30-50 |
| Coder-Next 80B 4-bit | 10-15 |

These are theoretical estimates based on bandwidth. Actual numbers depend on MLX kernel efficiency and will be measured with `bench_all_unified.py`.

### Models that need special handling

- **Qwen3.5-27B** — DeltaNet layers may degrade at 4-bit. Try 8-bit if quality is poor.
- **Coder-Next 80B** — Tight fit in 64GB. Monitor `memory_pressure` for swapping.
- **Gemma 4 26B** — MoE expert calibration may be uneven at 4-bit.

## Performance

Benchmarks not yet collected. Run:

```bash
./scripts/launch.sh devstral
# In another terminal:
python scripts/bench/bench_all_unified.py --name "Devstral-24B 4bit" --port 23334
```

## Patches

1 patch on top of SGLang `main`:

### 001-mlx-request-cleanup (PR #22632)

Fixes `KeyError` crash during concurrent multi-request decoding. The old MLX worker
deleted request state when a request was absent from an intermediate batch, but the
scheduler can legitimately skip a live request and include it later.

**Fix:** Explicit lifecycle-based cleanup via `cleanup_requests()` hooks instead of
implicit batch-membership inference.

See [patches/README.md](patches/README.md) for details.

## Setup

```bash
./scripts/setup.sh
```

Or manually:
```bash
# Create venv
python3 -m venv .venv && source .venv/bin/activate

# Install MLX
pip install mlx mlx-lm

# Clone and install SGLang
git clone https://github.com/sgl-project/sglang.git components/sglang
cd components/sglang
git apply ../../patches/001-mlx-request-cleanup.patch
cd python && pip install -e ".[srt]"
```

| Component | Version | Notes |
|-----------|---------|-------|
| SGLang | main + 1 patch | MLX backend (merged PR #20342) |
| MLX | latest | Apple's ML framework |
| mlx-lm | latest | Model loading + quantization |
| Python | 3.12+ | macOS native |

## Quantization

MLX uses its own quantization format. Convert HuggingFace models:

```bash
# 4-bit (default, best speed/quality)
python -m mlx_lm.convert --hf-path <hf-model-id> --mlx-path ~/AI/models/<name>-4bit -q --q-bits 4

# 8-bit (better quality, ~2x memory)
python -m mlx_lm.convert --hf-path <hf-model-id> --mlx-path ~/AI/models/<name>-8bit -q --q-bits 8
```

Or download pre-quantized models from `mlx-community/` on HuggingFace.

**AWQ/GPTQ models from CUDA/ROCm projects are NOT compatible** — MLX requires its own format.

See [rules-for-agents.md](rules-for-agents.md) for quantization rules and MoE/DeltaNet caveats.

## Test System

```
Model:   Mac mini (Mac16,11)
Chip:    Apple M4 Pro (14-core CPU, 20-core GPU)
Memory:  64 GB unified (LPDDR5, ~273 GB/s)
OS:      macOS (Darwin 25.2.0)
Python:  3.12+
```

## Structure

```
patches/                           # SGLang patches for MLX fixes
  001-mlx-request-cleanup.patch   #   Fix concurrent request KeyError
benchmarks/                        # Benchmark results (per-model directories)
  {model}/results.json            #   Structured data from bench_all_unified.py
  baselines.json                  #   Regression test baselines
scripts/
  launch.sh                       #   Unified model launcher (launch.sh <model>)
  common.sh                       #   Shared MLX environment setup
  setup.sh                        #   Full setup (venv, SGLang, MLX deps)
  devstral_chat_template.jinja    #   Devstral BOS-fix chat template
  bench/                          #   Benchmark scripts
  eval/                           #   Quality evaluation + warmup
components/sglang/                 # SGLang main + patches (cloned by setup.sh)
```
