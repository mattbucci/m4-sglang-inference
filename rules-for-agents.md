# Rules for AI Agents

## Inference Engine
**All inference MUST use SGLang with the native MLX backend** (`SGLANG_USE_MLX=1`).
No vLLM, no llama.cpp, no direct mlx_lm serving — SGLang is the serving layer.

## Hardware
- Apple M4 Pro (Mac mini, Mac16,11)
- 64 GB unified memory (shared CPU/GPU)
- 20-core GPU, ~273 GB/s memory bandwidth
- macOS (Darwin)

## Apple Silicon / MLX Constraints

### MLX Backend
- Activated via `SGLANG_USE_MLX=1` environment variable
- Models loaded via `mlx_lm.load()` — requires MLX-format weights
- **Greedy sampling only** — `mx.argmax` for token selection; no temperature, top-p, top-k
- No tensor parallelism — single unified memory device
- No CUDA graphs — not applicable on Metal
- No custom all-reduce — single device
- KV cache managed by MLX internally, not by SGLang's memory pool
- Variable-length prefill falls back to serial processing

### Model Format
- **MLX-format models required** — standard HuggingFace (BF16/FP16) or MLX quantized
- AWQ/GPTQ/Marlin models from CUDA/ROCm repos are NOT compatible
- Pre-quantized models available at `mlx-community/` on HuggingFace
- Convert BF16 models to 4-bit MLX: `mlx_lm.convert --hf-path <model> --mlx-path <output> -q --q-bits 4`
- Convert to 8-bit MLX: `mlx_lm.convert --hf-path <model> --mlx-path <output> -q --q-bits 8`

### Memory Budget
64 GB unified memory, shared between OS, CPU, and GPU. Practical GPU budget ~55 GB.

| Model | 4-bit size (approx) | Fits? |
|-------|:-------------------:|:-----:|
| Devstral-24B | ~14 GB | Yes |
| Coder-30B MoE | ~16 GB | Yes |
| Gemma 4 26B MoE | ~15 GB | Yes |
| Qwen3.5-27B | ~15 GB | Yes |
| Coder-Next-80B MoE | ~42 GB | Tight — leaves ~13 GB for KV cache + OS |

### Memory Bandwidth
The M4 Pro has ~273 GB/s memory bandwidth. For autoregressive decode, throughput is
memory-bandwidth-bound: each token requires reading all model weights once.

Expected single-user decode speed for 4-bit models:
- 24B model: ~40-60 tok/s (reading ~14 GB per token)
- 30B model: ~30-50 tok/s (reading ~16 GB per token)
- 80B model: ~10-15 tok/s (reading ~42 GB per token)

These are rough estimates — actual speed depends on MLX kernel efficiency and KV cache overhead.

### Tensor Bridge
The SGLang MLX backend converts tensors between PyTorch and MLX:
- Zero-copy on unified memory via `memoryview` (when possible)
- Special handling for bfloat16 (numpy doesn't support it)
- MPS tensors >1 GB kept on CPU to avoid `MPSTemporaryNDArray` allocation crashes

## Server Launch

```bash
source scripts/common.sh
setup_mlx_env
./scripts/launch.sh <model>
```

Always source `common.sh` and call `setup_mlx_env` before launching.

### Launch flags
- `SGLANG_USE_MLX=1` — required, activates MLX backend
- `--context-length` — max context; limited by available memory
- `--max-running-requests` — concurrent request limit; keep low for large models
- No `--tensor-parallel-size` (always 1)
- No `--quantization` flag needed (MLX handles quantization internally)
- No `--attention-backend` flag (MLX handles attention)
- No `--kv-cache-dtype` flag (MLX manages KV cache format)

## Quantization

### MLX native quantization (preferred)
MLX uses its own quantization format. Convert BF16 HuggingFace models:

```bash
# 4-bit quantization (best speed/quality tradeoff)
python -m mlx_lm.convert --hf-path <hf-model-id> --mlx-path <output-dir> -q --q-bits 4

# 8-bit quantization (better quality, ~2x memory)
python -m mlx_lm.convert --hf-path <hf-model-id> --mlx-path <output-dir> -q --q-bits 8
```

### Pre-quantized models
Many models are pre-quantized on HuggingFace under the `mlx-community/` namespace:
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit`
- `mlx-community/Qwen3-Coder-30B-A3B-4bit`
- etc.

Check `https://huggingface.co/mlx-community` for available models.

### MoE model quantization — CRITICAL
The same MoE quantization challenges apply to MLX:
- Standard quantization under-calibrates rare experts (inter-expert imbalance)
- Verify model quality after quantization with `scripts/eval/eval_comprehensive.py`
- If quality is poor, try 8-bit quantization or use a model with proper calibration

### DeltaNet/Mamba/SSM layers — caution
Models with recurrent state (DeltaNet, Mamba, SSM) accumulate quantization error:
`S(t) = gating * S(t-1) + delta`
- INT4 quantization may destroy output quality on these layers
- Qwen3.5-27B and Coder-Next-80B have DeltaNet layers
- Consider 8-bit or BF16 for these models if quality is poor at 4-bit

## Benchmarking

### Methodology
- Use `sglang.bench_serving` for accurate TPOT measurement when possible
- For models with greedy-only sampling, `bench_all_unified.py` measures tok/s via the chat API
- Concurrency sweep: 1, 2, 4, 8, 16
- Context sweep: powers of 2 from 128 to model max
- Save to `benchmarks/{model}/results.json`
- Regenerate charts: `python scripts/bench/generate_charts.py`
- Run regression test before committing: `./scripts/bench/bench_regression.sh <model>`
- Regression threshold: >10% deviation triggers alert

### Important notes
- MLX kernels compile on first use — first requests will be slow (warmup required)
- Unified memory means no OOM in the traditional sense — system will swap, degrading performance
- Watch `memory_pressure` to detect when models are too large for available memory
- Single-device means no TP overhead, but also no parallelism benefit

## Model Status

See [README.md](README.md) for current model status, benchmarks, and known issues.
