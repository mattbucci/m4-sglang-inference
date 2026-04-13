# Patches

Patches applied on top of SGLang `main` branch for MLX backend on Apple Silicon.
Apply in order with `scripts/setup.sh` or manually:

```bash
cd components/sglang
git apply ../../patches/001-mlx-radix-cache.patch
git apply ../../patches/002-mps-backend-defaults.patch
git apply ../../patches/003-mlx-skip-quantization-check.patch
git apply ../../patches/004-mlx-lifecycle-and-hybrid-fixes.patch
```

## 001-mlx-radix-cache (PR #21509)

**Source:** https://github.com/sgl-project/sglang/pull/21509

Adds radix cache support to the MLX backend. Introduces a Metal-native KV pool
(`MlxKVPool`) that auto-sizes from available memory, zero-copy gather for
prefix-cached prefills, and batched attention for BS>1 decode. ~102x prefill
throughput improvement on cache hits.

New files: `kv_cache/` sub-package (kv_pool, contiguous_cache, attention_wrapper, model_patching).

## 002-mps-backend-defaults

Sets safe defaults for MPS/MLX in `_handle_mps_backends()` (runs before process fork):
- Disable CUDA graph and piecewise CUDA graph (not applicable on Metal)
- Force `torch_native` attention backend (Triton unavailable on macOS)
- Disable multimodal by default (VLM image processors crash on MPS)
- Skip piecewise CUDA graph model config load when already disabled

## 003-mlx-skip-quantization-check

Skips `_verify_quantization()` when MLX backend is active. MLX models have
`quantization_config` without a `quant_method` field, causing SGLang's
verification to fail with `Unknown quantization method: ""`.

## 004-mlx-lifecycle-and-hybrid-fixes

Combined fix for request lifecycle management and hybrid DeltaNet/Mamba models:

**Lifecycle hooks** (from PR #22632 intent, adapted for radix cache rewrite):
- Add `cleanup_requests()` and `clear_runtime_state()` stubs to base `TpModelWorker`
- Implement them in `MlxTpModelWorker` to call `remove_request()`/`clear()`
- Wire scheduler `flush_cache` → `clear_runtime_state()`
- Wire `_handle_finished_req` → `cleanup_requests()`
- Remove batch-membership-based auto-cleanup from `_forward_batch_generation_mlx`

**Hybrid SSM model support** (Qwen3.5, Coder-Next):
- Override `hybrid_gdn_config`/`mamba2_config`/`linear_attn_model_spec` → None
  in `MlxModelRunnerStub` to prevent scheduler from creating `MambaRadixCache`
- Add `_DummyMambaPool` for scheduler runtime checker compatibility
- Force `is_multimodal=False` in MLX tp_worker subprocess
