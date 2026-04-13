# M4 Pro Inference: SGLang with MLX

High-throughput LLM inference on Apple M4 Pro (64GB unified memory) using SGLang's native MLX backend.

## Known Issues

- **Greedy sampling only** — MLX backend uses `mx.argmax`; temperature, top-p, top-k not yet supported
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
| Devstral-24B | Dense | ~14 GB | 17.2 | `launch.sh devstral` | Working |
| Coder-30B | MoE (128 experts) | ~16 GB | 71.7 | `launch.sh coder-30b` | Working |
| Gemma 4 26B | MoE (128 experts) | ~15 GB | 60.6 | `launch.sh gemma4` | Working |
| Qwen3.5-27B | DeltaNet hybrid | ~15 GB | 14.5 | `launch.sh qwen35` | Working |
| Coder-Next 80B | MoE+DeltaNet (512 experts) | ~42 GB | 55.3 | `launch.sh coder-next` | Working |

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
- **Gemma 4 26B** — Working. Our patch disables multimodal in `_handle_mps_backends` (before TokenizerManager forks) and forces `torch_native` attention (Triton not available on macOS).
- **Qwen3.5-27B** — Working including concurrent requests. Our patch overrides `hybrid_gdn_config` to prevent MambaRadixCache creation and adds `ArraysCache` support to the batched decode merge path.
- **Coder-Next 80B** — Working including concurrent (with ArraysCache fix). ~42 GB weights leaves ~12 GB for KV cache + OS.

## Performance (M4 Pro 64GB, SGLang + MLX, updated 2026-04-12)

**Methodology:** TPOT (Time Per Output Token) and TTFT (Time To First Token) measured with `sglang.bench_serving` which separates prefill from decode. Throughput measured with `bench_comprehensive.sh`. Quality validated with `eval_comprehensive.py`.

### Devstral-24B 4-bit

24B dense transformer. ~14 GB 4-bit weights. Max tested context: 32K.

**Quality:** 29/36 eval tests passed (8/8 code, 7/7 knowledge, 6/8 math, 4/5 edge cases).

| Context Length | TPOT (ms) | tok/s | TTFT (ms) |
|:--------------:|:---------:|:-----:|:---------:|
| 128 | 57.1 | 17.2 | 69 |
| 512 | 57.2 | 16.8 | 134 |
| 1K | 57.1 | 17.3 | 70 |
| 4K | 57.1 | 17.2 | 70 |
| 8K | 57.4 | 17.1 | 75 |
| 16K | 57.3 | 16.9 | 105 |
| **32K** | **57.5** | **16.9** | **117** |

| Concurrency | TPOT (ms) | Output tok/s |
|:-----------:|:---------:|:------------:|
| 1 | 127 | 24.5 |
| 2 | 238 | 25.1 |
| 4 | 333 | 29.6 |
| 8 | 434 | 39.3 |
| **16** | **913** | **39.5** |

**Notes:**
- **TPOT is flat at 57ms across all context lengths** — decode speed is independent of context on MLX (no attention recomputation)
- The M4 Pro's ~273 GB/s bandwidth reading 14 GB of weights gives a theoretical ceiling of ~19 tok/s — we're at 90% of theoretical
- TTFT (prefill) is fast: 69ms at 128 tokens, 117ms at 32K — limited by Metal compute, not bandwidth
- MLX kernels compile on first use: first request after cold start takes 20-30s
- Concurrency throughput scales to 39.5 tok/s @16 (TPOT degrades under batched decode)

### Coder-30B MoE 4-bit (128 experts)

30B total / 3B active MoE. ~16 GB 4-bit weights. Max tested context: 32K. Fastest model — MoE only reads active expert weights per token.

| Context Length | TPOT (ms) | tok/s | TTFT (ms) |
|:--------------:|:---------:|:-----:|:---------:|
| 128 | 14.0 | 33.9 | 27 |
| 512 | 14.1 | 33.8 | 194 |
| 1K | 13.9 | 33.9 | 28 |
| 4K | 13.9 | 33.9 | 27 |
| 8K | 13.9 | 33.8 | 34 |
| 16K | 14.3 | 33.8 | 153 |
| **32K** | **13.9** | **33.9** | **56** |

| Concurrency | TPOT (ms) | Output tok/s |
|:-----------:|:---------:|:------------:|
| 1 | 53.6 | 62.6 |
| 2 | 86.4 | 69.4 |
| 4 | 151.6 | 73.7 |
| 8 | 264.5 | 81.2 |
| **16** | **517.2** | **82.1** |

**Notes:**
- **TPOT: 14ms flat (71 tok/s decode)** — 4x faster than Devstral because MoE activates only 3B of 30B params per token
- The `sglang.bench_serving` reports 33.9 tok/s throughput at concurrency=1 because it includes scheduling overhead; raw single-user TPOT of 14ms = 71 tok/s decode
- Single-user with 256 in/256 out: TPOT 53.6ms, throughput 62.6 tok/s
- Peak concurrent throughput: 82.1 tok/s @16
- TPOT flat across all context lengths (14ms at 128 and 32K)

### Qwen3.5-27B DeltaNet 4-bit

27B DeltaNet hybrid (linear attention + standard attention). ~15 GB 4-bit weights. Max tested context: 32K. Single-user only (concurrent crashes on batched decode).

| Context Length | TPOT (ms) | tok/s | TTFT (ms) |
|:--------------:|:---------:|:-----:|:---------:|
| 128 | 68.5 | 14.5 | 81 |
| 512 | 68.6 | 14.5 | 87 |
| 1K | 68.8 | 14.5 | 86 |
| 4K | 68.6 | 14.5 | 92 |
| 8K | 68.9 | 14.4 | 83 |
| 16K | 68.6 | 14.4 | 105 |
| **32K** | **68.5** | **14.3** | **129** |

**Notes:**
- **TPOT: 68.5ms flat (14.5 tok/s)** — similar to Devstral (dense 24B), expected since both read ~14-15 GB of weights per token
- Concurrent requests now work — our patch adds `ArraysCache` support to `_merge_kv_caches`. 4 concurrent: 21.6 tok/s throughput, TPOT 142ms
- Our patch overrides `hybrid_gdn_config`/`mamba2_config` to prevent the scheduler from creating MambaRadixCache, which fixed the startup crash
- Thinking model: generates `<think>` tokens before response content. Use without `--reasoning-parser` for benchmarking.

### Gemma 4 26B MoE 4-bit (128 experts)

26B total / 4B active MoE. ~15 GB 4-bit weights. Max tested context: 4K.

| Context Length | TPOT (ms) | tok/s | TTFT (ms) |
|:--------------:|:---------:|:-----:|:---------:|
| 128 | 16.8 | 34.4 | 58 |
| 512 | 16.1 | 34.4 | 31 |
| 1K | 16.6 | 34.4 | 30 |
| 2K | 16.5 | 34.4 | 29 |
| **4K** | **16.8** | **34.4** | **63** |

**Notes:**
- **TPOT: 16.5ms flat (60.6 tok/s decode)** — fastest model, MoE activates only ~4B params per token
- Required three patches to work: disable multimodal in `_handle_mps_backends` (before TokenizerManager forks), force `torch_native` attention (Triton unavailable on macOS), skip server warmup
- `sglang.bench_serving` reports 34.4 tok/s throughput (includes scheduling overhead); raw TPOT of 16.5ms = 60.6 tok/s decode
- Context limited to 4K in current preset — needs testing at higher context

### Coder-Next 80B MoE+DeltaNet 4-bit (512 experts)

80B total / 3B active MoE + DeltaNet hybrid. ~42 GB 4-bit weights. Max tested context: 8K. Single-user only.

| Context Length | TPOT (ms) | tok/s | TTFT (ms) |
|:--------------:|:---------:|:-----:|:---------:|
| 128 | 18.0 | 34.4 | 29 |
| 512 | 18.1 | 34.4 | 50 |
| 1K | 18.1 | 34.4 | 33 |
| 4K | 18.1 | 34.4 | 65 |
| **8K** | **18.1** | **34.3** | **42** |

**Notes:**
- **TPOT: 18ms flat (55.3 tok/s decode)** — 80B model running at 55 tok/s on a Mac mini
- Only ~12 GB free after model weights — tight for KV cache at long context
- MoE activates only ~3B of 80B params per token, making decode speed comparable to much smaller models
- DeltaNet hybrid — concurrent requests now work with our `ArraysCache` merge patch
- `sglang.bench_serving` reports 34.4 tok/s throughput (includes overhead); raw TPOT 18ms = 55.3 tok/s

## Patches

5 patches on top of SGLang `main` (apply in order):

| Patch | Purpose |
|-------|---------|
| **001-mlx-request-cleanup** | Fix `KeyError` on concurrent decode (PR #22632) |
| **002-mps-backend-defaults** | Disable CUDA graph, force torch_native attention, disable multimodal on MPS |
| **003-mlx-skip-quantization-check** | Skip quantization verification for MLX models (no `quant_method` field) |
| **004-mlx-stub-hybrid-ssm-fixes** | Override hybrid SSM detection, add DummyMambaPool, disable multimodal in subprocess |
| **005-mlx-arrayscache-batched-decode** | Add `ArraysCache` support for concurrent DeltaNet model decode |

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
