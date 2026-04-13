# Patches

Patches applied on top of SGLang `main` branch for MLX backend on Apple Silicon.
Apply in order with `scripts/setup.sh` or manually:

```bash
cd components/sglang
git apply --exclude='test/*' ../../patches/001-mlx-request-cleanup.patch
git apply ../../patches/002-mps-backend-defaults.patch
git apply ../../patches/003-mlx-skip-quantization-check.patch
git apply ../../patches/004-mlx-stub-hybrid-ssm-fixes.patch
git apply ../../patches/005-mlx-arrayscache-batched-decode.patch
```

## 001-mlx-request-cleanup (PR #22632)

**Source:** https://github.com/sgl-project/sglang/pull/22632

Fixes premature request-state cleanup in the MLX backend that causes `KeyError`
during concurrent multi-request decoding. Replaces implicit batch-membership-based
cleanup with explicit lifecycle hooks (`cleanup_requests`, `clear_runtime_state`).

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

## 004-mlx-stub-hybrid-ssm-fixes

Fixes for hybrid DeltaNet/Mamba models (Qwen3.5, Coder-Next) on MLX:
- Override `hybrid_gdn_config`/`mamba2_config`/`linear_attn_model_spec` → None
  in `MlxModelRunnerStub` to prevent scheduler from creating `MambaRadixCache`
- Add `_DummyMambaPool` for scheduler runtime checker compatibility
- Force `is_multimodal=False` in MLX tp_worker subprocess

## 005-mlx-arrayscache-batched-decode

Adds `ArraysCache` support to `_merge_kv_caches()` for batched decode.
DeltaNet/linear attention layers use `ArraysCache` (not `KVCache`) for state.
Without this, concurrent requests crash with `TypeError: Unsupported cache type: ArraysCache`.
Also adds a generic fallback for any cache type with a `.merge()` classmethod.
