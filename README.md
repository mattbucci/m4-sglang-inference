# M4 Pro Inference: SGLang with MLX

High-throughput LLM inference on Apple M4 Pro (64GB unified memory) using SGLang's native MLX backend.

## Known Issues

- **Greedy sampling only** — MLX backend uses `mx.argmax`; temperature, top-p, top-k not yet supported
- **Batched decode quality** — Concurrent requests may produce garbage output due to KV cache corruption in the merge/extract path. Single-user inference is reliable. This is a limitation of the upstream MLX backend's batched decode implementation.
- **VLM warmup crash** — Devstral (Mistral3) detected as VLM; image processor triggers CUDA assertion on MPS. Workaround: `--skip-server-warmup` (set automatically in launch presets).
- **HDMI display blackout** — Screen may go black briefly when server starts heavy GPU load. This is a known M4 Pro HDMI issue under sustained Metal compute, not a crash. Display recovers within seconds.
- **Variable-length prefill** — Falls back to serial processing when sequences have different lengths
- **MPS 4GB tensor limit** — Tensors >1 GB kept on CPU to avoid `MPSTemporaryNDArray` crashes

## Quick Start

```bash
# 1. Setup: create venv, install SGLang + MLX deps, apply patches
./scripts/setup.sh

# 2. Run any model:
./scripts/launch.sh devstral            # Devstral-24B 4-bit — best all-round
./scripts/launch.sh coder-30b           # Coder-30B MoE 4-bit
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

All models run on SGLang with the MLX backend (`SGLANG_USE_MLX=1`). Models are downloaded automatically from HuggingFace on first launch. The `mlx-community/` namespace has pre-quantized MLX versions of most popular models.

### Agent / coding workloads (single-user)

| Model | Type | 4-bit size | 1-user tok/s | Launch | Status |
|-------|------|:----------:|:------------:|:------:|:------:|
| Devstral-24B | Dense | ~14 GB | 16 | `launch.sh devstral` | Working |
| Coder-30B | MoE (128 experts) | ~16 GB | 68 | `launch.sh coder-30b` | Working |
| Gemma 4 26B | MoE (128 experts) | ~15 GB | — | `launch.sh gemma4` | Testing |
| Qwen3.5-27B | DeltaNet hybrid | ~15 GB | — | `launch.sh qwen35` | Testing |
| Coder-Next 80B | MoE+DeltaNet (512 experts) | ~42 GB | — | `launch.sh coder-next` | Testing |

All models served as 4-bit MLX quantized from `mlx-community/` on HuggingFace. Max context is limited by available memory, not a fixed cap — 64GB unified memory supports long context for most models.

### Memory budget (64GB unified)

| Model | Weights | KV cache headroom | Concurrent users |
|-------|:-------:|:-----------------:|:----------------:|
| Devstral-24B | ~14 GB | ~41 GB | Many |
| Coder-30B MoE | ~16 GB | ~39 GB | Many |
| Qwen3.5-27B | ~15 GB | ~40 GB | Many |
| Gemma 4 26B MoE | ~15 GB | ~40 GB | Many |
| Coder-Next-80B | ~42 GB | ~13 GB | 1-2 |

~9 GB reserved for macOS + system overhead.

### Models that need special handling

- **Devstral-24B** — VLM (Mistral3 architecture) requires `--skip-server-warmup` to avoid image processor CUDA assertion. Vision not supported on MLX.
- **Qwen3.5-27B** — DeltaNet layers may degrade at 4-bit. Try 8-bit if quality is poor.
- **Coder-Next 80B** — Tight fit in 64GB. Monitor `memory_pressure` for swapping.
- **Gemma 4 26B** — MoE expert calibration may be uneven at 4-bit.

## Performance (M4 Pro 64GB, SGLang + MLX, updated 2026-04-12)

**Methodology:** All numbers use `bench_all_unified.py` which measures completion tokens / elapsed time for single-user context sweeps and concurrent throughput. Quality validated with `eval_comprehensive.py` (29/36 tests passed for Devstral, 8/8 code generation).

### Devstral-24B 4-bit

24B dense transformer. ~14 GB 4-bit weights. Best all-round model.

**Quality:** 29/36 eval tests passed (8/8 code, 7/7 knowledge, 6/8 math, 4/5 edge cases). Parallel stress test: 4/8 (batched decode quality issues — see Known Issues).

![Devstral context scaling](benchmarks/devstral-24b-4bit/context_vs_toks.png)

| Context Length | tok/s |
|:--------------:|:-----:|
| 128 | 16.0 |
| 256 | 15.5 |
| 512 | 14.5 |
| 1K | 12.6 |
| 2K | 9.7 |
| 4K | 6.7 |
| 8K | 4.2 |
| 16K | 2.5 |
| **32K** | **1.4** |

![Devstral concurrency](benchmarks/devstral-24b-4bit/concurrency_vs_toks.png)

| Concurrency | Total tok/s |
|:-----------:|:-----------:|
| 1 | 17 |
| 2 | 27 |
| 4 | 29 |
| 8 | 30 |
| **16** | **45** |

**Notes:**
- Single-user decode: ~16 tok/s at short context, dropping to 1.4 tok/s at 32K
- The M4 Pro's ~273 GB/s bandwidth reading 14 GB of weights gives a theoretical ceiling of ~19 tok/s — we're at 84% of theoretical at short context
- MLX kernels compile on first use: first request after cold start takes 20-30s. Subsequent requests are fast.
- Concurrency throughput scales to 45 tok/s @16 concurrent, but output quality degrades with batched decode (KV cache merge/extract limitation)

### Coder-30B MoE 4-bit (128 experts)

30B total / 3B active MoE. ~16 GB 4-bit weights. Fastest model — MoE only reads active expert weights per token.

![Coder-30B context scaling](benchmarks/coder-30b-4bit/context_vs_toks.png)

| Context Length | tok/s |
|:--------------:|:-----:|
| 128 | 61.8 |
| 512 | 61.4 |
| 1K | 55.7 |
| 4K | 36.1 |
| 8K | 24.2 |
| 16K | 14.0 |
| **32K** | **14.0** |

![Coder-30B concurrency](benchmarks/coder-30b-4bit/concurrency_vs_toks.png)

| Concurrency | Total tok/s |
|:-----------:|:-----------:|
| 1 | 68 |
| 2 | 60 |
| 4 | 70 |
| 8 | 82 |
| **16** | **94** |

**Notes:**
- 68 tok/s single-user — 4x faster than Devstral because MoE only activates 3B of 30B params per token
- Throughput scales to 94 tok/s @16 concurrent
- Context length limited by KV cache memory, not model weights — should support much longer context with 64GB

## Patches

2 patches on top of SGLang `main`:

### 001-mlx-request-cleanup (PR #22632)

Fixes `KeyError` crash during concurrent multi-request decoding. The old MLX worker
deleted request state when a request was absent from an intermediate batch, but the
scheduler can legitimately skip a live request and include it later.

**Fix:** Explicit lifecycle-based cleanup via `cleanup_requests()` hooks instead of
implicit batch-membership inference.

### 002-mlx-quantization-skip

Fixes startup crash on MLX models whose `quantization_config` lacks a `quant_method`
field. Also disables CUDA graph and piecewise CUDA graph for MPS device to prevent
unnecessary model config loads that trigger the same error.

See [patches/README.md](patches/README.md) for details.

## Setup

```bash
./scripts/setup.sh
```

Or manually:
```bash
# Create venv
python3 -m venv .venv && source .venv/bin/activate

# Clone and install SGLang with MPS/MLX extras
git clone https://github.com/sgl-project/sglang.git components/sglang
cd components/sglang
git apply ../../patches/001-mlx-request-cleanup.patch
git apply ../../patches/002-mlx-quantization-skip.patch
cd python
cp pyproject_other.toml pyproject.toml
pip install -e ".[srt_mps]"
```

| Component | Version | Notes |
|-----------|---------|-------|
| SGLang | main + 2 patches | MLX backend (merged PR #20342) |
| MLX | 0.31.1 | Apple's ML framework |
| mlx-lm | 0.31.2 | Model loading + quantization |
| PyTorch | 2.9.1 | MPS backend for tensor bridge |
| Python | 3.12 | macOS native |

## Quantization

MLX uses its own quantization format. Most models are pre-quantized on HuggingFace:

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit` (180K downloads)
- `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`
- `mlx-community/gemma-4-26b-a4b-it-4bit` (77K downloads)
- `mlx-community/Qwen3.5-27B-4bit` (32K downloads)
- `mlx-community/Qwen3-Coder-Next-4bit`

Or convert HuggingFace models yourself:

```bash
# 4-bit (default, best speed/quality)
python -m mlx_lm.convert --hf-path <hf-model-id> --mlx-path ~/AI/models/<name>-4bit -q --q-bits 4

# 8-bit (better quality, ~2x memory)
python -m mlx_lm.convert --hf-path <hf-model-id> --mlx-path ~/AI/models/<name>-8bit -q --q-bits 8
```

**AWQ/GPTQ models from CUDA/ROCm projects are NOT compatible** — MLX requires its own format.

See [rules-for-agents.md](rules-for-agents.md) for quantization rules and MoE/DeltaNet caveats.

## Test System

```
Model:   Mac mini (Mac16,11)
Chip:    Apple M4 Pro (14-core CPU, 20-core GPU)
Memory:  64 GB unified (LPDDR5, ~273 GB/s)
OS:      macOS 26.2 (Darwin 25.2.0)
Python:  3.12
MLX:     0.31.1
SGLang:  main (2026-04-12) + 2 patches
```

## Structure

```
patches/                           # SGLang patches for MLX fixes
  001-mlx-request-cleanup.patch   #   Fix concurrent request KeyError (PR #22632)
  002-mlx-quantization-skip.patch #   Skip quant verification + CUDA graph on MPS
benchmarks/                        # Benchmark results (per-model directories)
  devstral-24b-4bit/              #   Devstral results + charts
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
