# Patches

Patches applied on top of SGLang `main` branch for MLX backend on Apple Silicon.
Apply in order with `scripts/setup.sh` or manually:

```bash
cd components/sglang
git apply ../../patches/001-mlx-radix-cache.patch
git apply ../../patches/002-mps-backend-defaults.patch
git apply ../../patches/003-mlx-skip-quantization-check.patch
git apply ../../patches/004-mlx-lifecycle-and-hybrid-fixes.patch
git apply ../../patches/005-mlx-attn-wrapper-varargs.patch
git apply ../../patches/006-mlx-offsetcache-subscript.patch
git apply ../../patches/007-mlx-vlm-fallback.patch
```

## 001-mlx-radix-cache ([PR #21509](https://github.com/sgl-project/sglang/pull/21509))

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

**Lifecycle hooks** (from [PR #22632](https://github.com/sgl-project/sglang/pull/22632) intent, adapted for radix cache rewrite):
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

## 005-mlx-attn-wrapper-varargs

Fix `MLXAttentionWrapper.__call__` to accept variable positional arguments.

Some models (ministral3/Devstral) pass `attn_scale` as an extra positional argument
to the attention forward method: `attn(x, attn_scale, mask, cache)`. The original
wrapper only accepted `(x, mask, cache)`, causing a `TypeError`.

- Change wrapper signature to `(x, *args, **kwargs)` for full pass-through
- Extract `attn_scale` from extra args and apply to queries in batched decode path

## 006-mlx-offsetcache-subscript

Adds `__getitem__` and `__len__` to `OffsetCache` (the data-free cache shim
introduced in patch 001).

Hybrid models like Qwen3.5 / Coder-Next have `linear_attn` / DeltaNet layers
that probe `cache[0] is not None` to detect first-call vs. resumed state.
Without this fix, batched decode on those models crashes:

```
File ".../mlx_lm/models/qwen3_5.py", line 148, in __call__
    if cache is not None and cache[0] is not None:
TypeError: 'OffsetCache' object is not subscriptable
```

The new `__getitem__` returns `None` for any index, signalling "no cached
state" so the layer falls back to a zero-init recurrent state. **This is
a stopgap** — DeltaNet now starts fresh each batched-decode step and loses
its recurrent state. Real fix requires per-request DeltaNet state in
`caches[i]`, passed through to the model instead of the shim. See memory
project_qwen35_deltanet_decode_crash.

## 007-mlx-vlm-fallback (in post_apply.py)

Routes VLM model loads through `mlx_vlm.load` with a `_TextOnlyVLMShim` so
SGLang can load Idefics3/SmolVLM/Qwen2-VL/Mistral3 (Devstral)/Qwen3.5/3.6/Gemma 4
where `mlx_lm.load` fails with `Model type X not supported`. Combined with
config.json detection (`vision_config`, `image_token_id`, `image_token_index`),
forces VLM path even when `mlx_lm` would have technically worked but its
`Model.__call__` only accepts text inputs.

## 008-mlx-hybrid-serial-decode (in post_apply.py)

Stopgap for batched decode on DeltaNet hybrids (Qwen3.5, Coder-Next): if any
layer's cache isn't `ContiguousKVCache` or `OffsetCache`, force serial decode
(one request at a time). Real fix needs per-request DeltaNet `conv_state` /
`ssm_state` plumbed through `caches[i]`. Limits MAX_RUNNING=1 for these models.

## 009-modality-multi-images (in post_apply.py)

Adds `MULTI_IMAGES` member to SGLang's `Modality` enum that
`transformers_auto.py:133` references but doesn't define. Without this,
ANY image-bearing request 500s with `AttributeError: type object 'Modality'
has no attribute 'MULTI_IMAGES'` before reaching the model. Upstream SGLang bug.

## 010-mlx-vlm-pixel-values (in post_apply.py)

Threads `pixel_values` from `req.multimodal_inputs` through
`tp_worker → MlxModelRunner.prefill → TextOnlyVLMShim` so VLM models can
actually do image inference. Also threads `model_specific_data`
(`image_grid_thw`, `video_grid_thw`, `second_per_grid_ts`) via
`mm_extra_kwargs`. Stacks across all `mm_items` for multi-image / video-frame
sequences (Qwen3.5/3.6 see N images instead of just the first).

## 011-mps-stub-cuda-redirect (in post_apply.py)

SGLang's `_mps_stub` patches `torch.Tensor.to` for MPS but doesn't handle
`.to('cuda')` — transformers' image processor unconditionally calls
`.to('cuda')` on Apple, hitting CUDA's `_lazy_init` which crashes
"Torch not compiled with CUDA enabled." Redirect cuda → cpu so the image
tensor lands somewhere torch can handle.

## 012-mm-utils-shm-page-rounding (in post_apply.py)

`SharedMemory` page-rounds size on macOS (16 KB pages on M-series), so
`torch.frombuffer(shm.buf, ...)` produces a tensor LARGER than the logical
nbytes. Slice to logical size on both write and read sides.

## 013-hybrid-cache-via-vlm-language-model (in post_apply.py)

ROOT CAUSE for the "DeltaNet 4-bit quality issue" — was NOT quantization.
When Qwen3.5/3.6 load via `mlx_vlm.load` (because of `vision_config`), the
outer wrapper has no `make_cache` attribute — it lives on `language_model`.
`_acquire_cache` couldn't find it and fell through to building uniform
`ContiguousKVCache` for every layer, giving DeltaNet's hybrid layers the
wrong cache type. Output looked grammatical but was fluent garbage. Fix
routes the cache via `model.language_model.make_cache()` and resets
ArraysCache state on pool reuse (the second part also helps text-only
DeltaNet hybrids like Coder-Next). **Re-measured: Qwen3.5-27B MMLU 16.7%
→ 93.0%, HumanEval 0% → 100%.**

## 014-contiguous-cache-make-mask-handles-offset (in post_apply.py)

`ContiguousKVCache.make_mask` always returned the `"causal"` sentinel,
which `mlx_vlm` expands to a square `(N, N)` `mx.array`. Fine for full
prefill from scratch; broken for chunked-prefill continuation: the keys
have grown to `(offset + N)` length, so a square mask doesn't broadcast
against `(1, n_heads, N, offset+N)` scores → `ValueError(broadcast_shapes)`.
Fix builds an explicit `(N, offset+N)` causal mask when `offset > 0` (with
optional sliding-window support); preserves `"causal"` sentinel when
`offset == 0` so `mx.fast.SDPA`'s optimized path stays on the hot first chunk.

Caveat: Gemma 4 31B-it has KV-shared layers (`layer_idx_to_cache_idx`)
where mask creation reads from one cache slot but updates land in another;
patch 014 alone doesn't unblock its long-context. Tracked in
`benchmarks/gemma4-31b-it-mxfp4/README.md`.
