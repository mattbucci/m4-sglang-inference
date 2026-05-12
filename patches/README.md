# Patches — SGLang v0.5.11 on Apple Silicon

10 patches on top of SGLang `v0.5.11` (commit `612785ffd`, 2026-05-04) for the MLX backend on Apple M4 Pro. Applied in order by `scripts/setup.sh`:

```bash
cd components/sglang
git checkout v0.5.11
for p in ../../patches/0[01][0-9]-*.patch; do git apply "$p"; done
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
| 009 | mlx-nemotron-h-support | `kv_cache/attention_wrapper.py`, `kv_cache/model_patching.py`, `model_runner.py` | NemotronH-style hybrids (Mamba2 + Attention + MoE) expose `mixer` on every layer with the type alternating per-layer. `find_attention_layers` now probes `self_attn`/`attention`/`mixer` and accepts `mixer` only when at least one layer's mixer is real attention (q/k/v/o_proj present); `patch_model_attention` skips non-attention mixers; `_get_attn_config` samples the first attention layer instead of `layer_list[0]`. The attention wrapper accepts either `n_heads`/`n_kv_heads` (mlx_lm) or `num_attention_heads`/`num_key_value_heads` (mlx_vlm + NemotronH) and skips the RoPE call when `inner.rope` is absent (NemotronH's attention is position-free — RoPE lives in the interleaved Mamba layers). |
| 010 | mlx-vlm-position-cache-reset | `model_runner.py` | mlx_vlm's `LanguageModel` for qwen3_5 (and 14 other VLM families that share the same caching pattern: qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_vl_moe, qwen3_5_moe, qwen3_omni_moe, glm4v, glm4v_moe, glm_ocr, ernie4_5_moe_vl, hunyuan_vl, paddleocr_vl, falcon_ocr, falcon_perception) memoizes `_position_ids` and `_rope_deltas` on the instance and only invalidates them when a new `pixel_values` arrives. Under `MAX_RUNNING>1` with text-only requests, request A's cached `(3, 1, L_A)` position tensor is still on the model when request B's prefill arrives — `apply_multimodal_rotary_pos_emb` then broadcasts `(1, 1, L_A, 64)` cos/sin against `(1, 24, L_B, 64)` queries and crashes. `prefill_start` now nulls these attributes at every new-request boundary; chunked-prefill `extend_start` deliberately leaves them intact (the cache is valid for chunk 2+ of the same request). Unblocks `MAX_RUNNING=4` on Qwen3.5-27B end-to-end (verified 2026-05-12 with 16-prompt mixed-length bench: 16/16 successful, zero broadcast errors). Resolves the long-standing crash documented in `patches/HYBRID_CONCURRENT_TRACE_PLAN.md`. |
| 011 | mlx-hybrid-batched-decode | `model_runner.py` | Replaces the serial-per-request hybrid decode fallback (`decode_batch_start` lines 1000-1020 pre-patch) with true batched decode for DeltaNet/GDN/Mamba2 hybrids. Layer-by-layer classification: `ContiguousKVCache` slots are full-attention layers routing through the existing `BatchedDecodeContext` + `MLXAttentionWrapper` path; non-`ContiguousKVCache` slots are linear-attention layers whose per-request `conv_state` and `ssm_state` are stacked along axis 0 into `(B, ...)` tensors for one batched call to `gated_delta_update` (which is already fully batched per `mlx_lm/models/gated_delta.py:262-283`). After the forward, each layer's updated batched state is split back into the per-request `ArraysCache` slots. The position-id mismatch in `mlx_vlm.LanguageModel.__call__` is harmless: full-attention layers ignore the model-level `position_ids` (MLXAttentionWrapper uses `ctx.offsets` directly), and linear layers don't use position encoding. Unblocks proper batched throughput on qwen35/qwen36/coder-next at `MAX_RUNNING>1` — without this patch, multi-user serving still works (correctness from patch 010) but does B forwards per decode step. |

## What was dropped at v0.5.11

| Old patch | Status |
|-----------|--------|
| 001-mlx-radix-cache | **Upstreamed.** SGLang v0.5.11 ships `python/sglang/srt/hardware_backend/mlx/kv_cache/` natively with `ContiguousKVCache`, `MlxKVPool`, `OffsetCache`, `attention_wrapper.py`, and `model_patching.py`. |

## Apply / sanity check

```bash
# Apply all
cd components/sglang && git checkout v0.5.11
for p in ../../patches/0[01][0-9]-*.patch; do git apply "$p"; done

# Verify clean apply on a fresh tree
for p in ../../patches/0[01][0-9]-*.patch; do git apply --check "$p" && echo "OK: $(basename $p)"; done
```

## Open follow-up

Patch 008 ships the `kv_quant.py` module + `MlxModelRunner.__init__` plumbing for `kv_cache_mode` / `context_length`, but `ContiguousKVCache` and `MlxKVPool` still store fp16 KV. To activate turboquant end-to-end on v0.5.11 we need to wire `KVQuantizer` into both buffer classes. Tracked as TODO; the parameter pass-through is in place so when wired, no API surface changes.

## v0.5.11 rebase context

See [REBASE-v0.5.11-NOTES.md](REBASE-v0.5.11-NOTES.md) for the strategy that drove this rebase (drop what's upstream, keep only what's load-bearing for the mlx-community model set that mirrors what sister teams run).
