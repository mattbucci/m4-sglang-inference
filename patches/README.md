# Patches — SGLang v0.5.11 on Apple Silicon

7 patches on top of SGLang `v0.5.11` (commit `612785ffd`, 2026-05-04) for the MLX backend on Apple M4 Pro. Applied in order by `scripts/setup.sh`:

```bash
cd components/sglang
git checkout v0.5.11
for p in ../../patches/00[2-9]-*.patch; do git apply "$p"; done
```

| # | Patch | Files | Why |
|:-:|-------|-------|-----|
| 002 | mps-backend-defaults | `server_args.py` | Disable CUDA graph & piecewise CUDA on MPS, force `torch_native` attention, default multimodal off (Devstral re-enables via patch 007). |
| 003 | mlx-skip-quantization-check | `configs/model_config.py` | MLX checkpoints have `quantization_config` without `quant_method`; SGLang's verify-quant raises. Skip when MLX backend is active. |
| 004 | mlx-lifecycle-and-hybrid-fixes | `model_runner.py`, `model_runner_stub.py`, `tp_worker.py` (MLX + base), `scheduler.py`, `scheduler_output_processor_mixin.py` | Lifecycle (clear-on-idle, drop-on-finish) + hybrid-model bookkeeping (`hybrid_gdn_config`/`mamba2_config`/`linear_attn_model_spec` properties, `_DummyMambaPool`) + the load-bearing **patch 013** that routes hybrid cache via `model.language_model.make_cache()` for VLM-wrapped DeltaNet (Qwen3.5/3.6 MMLU 16.7% → 93.0%) + **patch 015** keep `RotatingKVCache` native for Gemma 4 sliding layers + full cache reset on pool reuse + VLM-detect-first `_load_model` with image-aware `_TextOnlyVLMShim` + RoPE auto-scaling for 256K-on-40K-native models. |
| 005 | mlx-attn-wrapper-varargs | `kv_cache/attention_wrapper.py` | Devstral / Ministral3 pass `attn_scale` as a positional arg between `x` and `mask`. Wrapper now accepts `*args, **kwargs` and applies `attn_scale` if present. |
| 006 | mlx-offsetcache-and-make-mask | `kv_cache/contiguous_cache.py` | `OffsetCache.__getitem__`/`__setitem__`/`__len__`/`lengths`/`advance` stubs so hybrid DeltaNet decode doesn't `AttributeError` (the cache surface, not data). **Patch 014** also lives here: `ContiguousKVCache.make_mask` returns an explicit `(N, offset+N)` causal mask when `offset>0`, unblocking chunked prefill at large context. |
| 007 | mlx-multimodal-and-mps-shim | `_mps_stub.py`, `managers/mm_utils.py`, `managers/schedule_batch.py` | (a) MPS stub redirects `to('cuda')` → `to('cpu')` so transformers code that defaults to CUDA keeps working on Apple Silicon. (b) `ShmPointerMMData` slice fix — macOS rounds shm allocations up to a 16 KB page, so `torch.frombuffer(shm.buf, ...)` returns a larger-than-logical tensor; slice to `nbytes` on write, to `n_elements` on read. (c) Add `Modality.MULTI_IMAGES` enum member that SGLang's `transformers_auto` references but doesn't define. |
| 008 | mlx-kv-quant-module | `kv_cache/kv_quant.py` (new), `kv_cache/__init__.py` | KV cache quantization for MLX backend. New `KVCacheMode` enum + `parse_kv_cache_mode` (accepts `fp8` / `mxfp8` / `turboquant` / `tq` / `4bit` aliases) + `bytes_per_element` for pool sizing + `KVQuantizer` with quantize/dequantize on 3D pool buffers and 4D cache buffers. Wired into `MlxModelRunner.__init__` (accepts `kv_cache_mode` + `context_length` kwargs); turboquant is load-bearing for 256K work on the 64 GB Mac (~3.5× savings vs fp16, ~1.75× vs fp8). |

## What was dropped at v0.5.11

| Old patch | Status |
|-----------|--------|
| 001-mlx-radix-cache | **Upstreamed.** SGLang v0.5.11 ships `python/sglang/srt/hardware_backend/mlx/kv_cache/` natively with `ContiguousKVCache`, `MlxKVPool`, `OffsetCache`, `attention_wrapper.py`, and `model_patching.py`. |

## Apply / sanity check

```bash
# Apply all
cd components/sglang && git checkout v0.5.11
for p in ../../patches/00[2-9]-*.patch; do git apply "$p"; done

# Verify clean apply on a fresh tree
for p in ../../patches/00[2-9]-*.patch; do git apply --check "$p" && echo "OK: $(basename $p)"; done
```

## Open follow-up

Patch 008 ships the `kv_quant.py` module + `MlxModelRunner.__init__` plumbing for `kv_cache_mode` / `context_length`, but `ContiguousKVCache` and `MlxKVPool` still store fp16 KV. To activate turboquant end-to-end on v0.5.11 we need to wire `KVQuantizer` into both buffer classes. Tracked as TODO; the parameter pass-through is in place so when wired, no API surface changes.

## v0.5.11 rebase context

See [REBASE-v0.5.11-NOTES.md](REBASE-v0.5.11-NOTES.md) for the strategy that drove this rebase (drop what's upstream, keep only what's load-bearing for the mlx-community model set that mirrors what sister teams run).
