# Patches — SGLang v0.5.11 on Apple Silicon

13 patches on top of SGLang `v0.5.11` (commit `612785ffd`, 2026-05-04) for the MLX backend on Apple M4 Pro. Applied in order by `scripts/setup.sh`:

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
| 011 | mlx-hybrid-batched-decode-gated-attn | `model_runner.py`, `kv_cache/attention_wrapper.py` | Two co-dependent changes that together unblock true batched decode for DeltaNet hybrids (Qwen3.5/3.6 family). **(a) `model_runner.decode_batch_start`**: replaces the serial-per-request hybrid fallback with a mixed per-layer cache — full-attention layers (ContiguousKVCache) route through the existing `BatchedDecodeContext`+`MLXAttentionWrapper`, linear-attention layers (ArraysCache) stack per-request `conv_state`/`ssm_state` along axis 0 into `(B, ...)` tensors so the existing `gated_delta_update` kernel (already fully batched per `mlx_lm/models/gated_delta.py:262-283`) processes all B requests in one forward; after the call each layer's updated batched state is split back into per-request ArraysCache. **(b) `MLXAttentionWrapper._batched_decode`**: extended to handle mlx_vlm's `Qwen3_5Attention` style — q_proj output is `n_heads*head_dim*2` (queries + gate concatenated), so head_dim must be derived from keys not queries; the gate is split off the queries reshape and `sigmoid(gate)`-multiplied after SDPA before `o_proj`. RoPE dispatches on `hasattr(inner, "rotary_emb")`: if present, build `(3, B, 1)` text-only position_ids from `ctx.offsets`, call `inner.rotary_emb(values, position_ids)`, then `apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)` (imported from `type(inner).__module__` so the same wrapper works for every mlx_vlm family). `args[0]` is also type-discriminated: only treated as `attn_scale` when it's a Python number or 0-d mx.array (mlx_lm Devstral/ministral3 style); for mlx_vlm callers where args[0] is the mask, the existing built-from-ctx padding mask is used instead. **Verified 2026-05-12 08:39**: Qwen3.5-27B `MAX_RUNNING=4` 16-prompt bench reached 16 concurrent decode with gen throughput 41 tok/s before the OOM guard fired on memory pressure (mf=0.4 + 16 sequences is still over the M4's activation budget — that's a separate tuning task, not a code bug). Without this patch the same bench did 1× per-request serial decode at MR>1. |
| 012 | mlx-sync-pool-skip-non-contiguous | `model_runner.py` | `_sync_new_kv_to_pool` now filters to `ContiguousKVCache` layers only and writes per-layer via `set_kv`, instead of stacking all layers and calling `set_kv_all_layers`. Hybrid models have heterogeneous per-layer caches (`ContiguousKVCache` for full-attention, `ArraysCache` for DeltaNet/Mamba2 recurrent state, `RotatingKVCache` for Gemma 4 sliding) — only the first kind has the `(1, n_kv_heads, S, head_dim)` shape the pool expects, so the old `mx.stack([... .keys ...])` crashed with `AttributeError: 'ArraysCache' object has no attribute 'keys'` or shape mismatch. With the filter, non-attention layer slots stay zero-initialised in the pool and the write side stops crashing; users must still keep `--disable-radix-cache` for hybrid models for correctness (radix prefix reuse would still produce wrong outputs at the recurrent layers — the pool simply can't represent linear state per-token). This is the defence-in-depth fix for the latent bug that bench runs and ad-hoc launches kept tripping over before commit a8a3ff0 added `--disable-radix-cache` to every hybrid preset. |
| 013 | mlx-vlm-pixel-values | `model_runner.py`, `tp_worker.py` | **Restores the v0.5.10 VLM image-bearing inference path that was silently lost in the v0.5.11 rebase.** The original `Patch 010: pixel_values plumbing tp_worker → prefill → TextOnlyVLMShim` (Apr-18 commit `f20ee6e`) was the load-bearing change that made Devstral describe images correctly on the old stack. It didn't reapply during `ebe23bb` and the slot was silently reused for an unrelated patch (the new `mlx-vlm-position-cache-reset`, same number, different purpose). Between then and the 2026-05-13 probe sweep, every VLM image request silently took the text-only branch of `_TextOnlyVLMShim` — model hallucinated content from training prior, `validate_capabilities.py` keyword grep happened to pass on lucky words like "circle". This patch re-adds: `MlxModelRunner.prefill` / `prefill_start` accept `pixel_values=None, mm_kwargs=None`; build a `_model_kwargs` dict that's threaded to every `self.model(input_ids, cache=cache, **_model_kwargs)` call (both `disable_radix_cache` branch and full-radix branch). `tp_worker._forward_batch_generation_mlx` extracts `pixel_values` from `req.multimodal_inputs.mm_items[0].feature` (torch→mlx via numpy bridge) and forwards every key in `item.model_specific_data` as `mm_kwargs` (Pixtral/Mistral3 require `image_sizes` for the patch merger; Qwen2-VL uses `image_grid_thw`; Qwen3.5/3.6 use `video_grid_thw` + `second_per_grid_ts`). Verified 2026-05-13 by `probe_vision.py`: Devstral STRONG ("The image shows a single red circle with a black outline on a white background.") and Qwen3.5-9B-8bit STRONG (4-step reasoning trace correctly identifies circle/red/white/black-outline) — both match the Apr-18 baseline. |
| 014 | mlx-gemma4-image-only-processor | `utils/hf_transformers/processor.py` | **Unblocks Gemma 4 image+text serving on mlx-community checkpoints.** Upstream `google/gemma-4-26b-a4b-it` and its mlx-community quantizations ship only `processor_config.json` (image processor + token-id metadata). Neither `preprocessor_config.json` (audio feature_extractor) nor `video_preprocessor_config.json` is present. On transformers 5.5.3, `AutoProcessor.from_pretrained` calls `AutoFeatureExtractor.from_pretrained` (404) and `AutoVideoProcessor.from_pretrained` (404) because `Gemma4Processor.__init__` strictly requires all four sub-components. The construction failed with `OSError: Can't load feature extractor for ... preprocessor_config.json` before any image processing path ran, and Gemma 4 was text-only on M4. SGLang's downstream `Gemma4SGLangProcessor` already accesses `feature_extractor` and `video_processor` via `getattr(..., None)` with safe defaults — the strict requirement was only at the transformers-level constructor. Patch adds a `_build_gemma4_image_only_processor` helper that replays `Gemma4Processor.__init__`'s body plus `ProcessorMixin.__init__`'s token-id-list setup, but skips the strict type check on the missing sub-components and leaves `feature_extractor` / `video_processor` as `None`. Dispatch from the `AutoProcessor.from_pretrained` OSError handler when `"feature extractor" in str(e).lower()` and `config.model_type == "gemma4"`. Verified end-to-end 2026-05-15: `get_processor('mlx-community/gemma-4-26b-a4b-it-4bit')` returns a working processor, `proc(text=..., images=[...])` produces `pixel_values` of shape (1, 2520, 768) + `mm_token_type_ids` correctly tagging image tokens. Probe_vision pending live server boot. |

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

## Shipped narrative — resolved post-rebase

Forensics for items that started life on the main README's "Active work" list and have since landed. The patch table above is the canonical reference; this section just keeps the *why* so future debuggers don't have to re-derive it from `git log`.

### Patch 013 — VLM image regression fixed (2026-05-13)

**Symptom:** between the v0.5.11 rebase and the 2026-05-13 probe sweep, every VLM image request silently took the text-only branch of `_TextOnlyVLMShim`. Devstral on a red-circle-on-white prompt returned `"A diagram of a circular flow chart with a central circle labeled '1' surrounded by 12 smaller circles numbered 2 through 13..."` — total fabrication from training prior. `validate_capabilities.py:check_vision` keyword grep happened to pass because the word "circle" appeared. The fabricated-VLM mode escaped detection from Apr 18 → May 13.

**Root cause:** the v0.5.11 rebase silently DROPPED the original Apr-18 `Patch 010: pixel_values plumbing tp_worker → prefill → TextOnlyVLMShim` (commit `f20ee6e`). The patch-010 slot was reused for an unrelated change (the new `mlx-vlm-position-cache-reset`, same number, different purpose). The omission was invisible to `git apply --check` because the new patch 010 applied cleanly — there was no merge conflict to surface.

**Fix:** patch 013 re-threads `pixel_values` + `mm_kwargs` (image_sizes for Mistral3/Pixtral, image_grid_thw for Qwen-VL, video_grid_thw + second_per_grid_ts for Qwen3.5/3.6 video) from `tp_worker._forward_batch_generation_mlx` through `MlxModelRunner.prefill` / `prefill_start` all the way to `self.model(input_ids, cache=cache, **mm_kwargs)`. Devstral + Qwen3.5-9B-8bit both probe_vision STRONG after the patch.

**Lesson:** when a rebase reuses a patch number, re-check the patch *contents*. Trust the probe trio over the keyword-grep validator. Sister teams hit the same lesson on R9700's patch 030 (presharded-w2 detection) — read the number, not just the file count, when validating a rebase.

### Patch 011 — hybrid batched decode + Qwen3.5 gated multimodal attention (2026-05-12)

**Symptom:** at `MAX_RUNNING>1` on any Qwen3.5/3.6 hybrid, decode would either crash or fall back to serial-per-request execution. Pre-patch the 8-prompt random bench gave 1× throughput at MR>1.

**Two co-dependent changes:**
- `model_runner.decode_batch_start` replaces the serial-per-request hybrid fallback with a mixed per-layer cache. Full-attention (ContiguousKVCache) layers route through `BatchedDecodeContext`+`MLXAttentionWrapper`; linear-attention (ArraysCache) layers stack per-request `conv_state`/`ssm_state` along axis 0 into `(B, ...)` tensors so `gated_delta_update` (already fully batched per `mlx_lm/models/gated_delta.py:262-283`) processes B requests in one forward, then splits back per-request.
- `MLXAttentionWrapper._batched_decode` extended for `Qwen3_5Attention`: q_proj output is `n_heads*head_dim*2` (queries + gate concatenated), head_dim derived from keys not queries, gate split off and `sigmoid(gate)`-multiplied after SDPA. RoPE dispatches on `hasattr(inner, "rotary_emb")`; `args[0]` type-discriminated (number/0-d mx.array → `attn_scale`, else mask).

**Verification (2026-05-12):** qwen35 (27B+DeltaNet) peak 34 tok/s at MR=2 (2.3×); qwen36-27b (Dense+DeltaNet) peak 34 tok/s at MR=2; **qwen36 (35B-A3B MoE+DeltaNet) peak 148 tok/s at MR=2** — MoE active-params win compounds with batched decode. Wrapper rework is backward-compatible on non-hybrid mlx_lm: Devstral 24B peak 40 tok/s at MR=4 (8/8 successful); Qwen3-30B-A3B peak **160 tok/s at MR=8** (16-prompt queue, concurrency 15.11, 16/16 successful). Recipe: `MAX_RUNNING={2,4} MEM_FRAC=0.4 EXTRA_ARGS="--disable-radix-cache --chunked-prefill-size 1024 --max-total-tokens 32768"`.

**Open follow-up:** 16-prompt queue at MR=4 still trips the OOM guard. Needs mf=0.3 or smaller chunked-prefill — tuning, not a code bug.

### Reasoning + tool-call parsers wired into launch presets (2026-05-13)

**Gap:** 3090 shipped `--tool-call-parser` on every Qwen3-Coder preset; M4 had it only on `coder-30b` and `coder-next`. R9700 also had `--reasoning-parser gemma4`. M4 was missing the reasoning parser on Gemma 4 and the tool-call parser on Devstral, Qwen3-family, Qwen3.5/3.6 family.

**Audit + fix (commit `01f19d3`):** 11 presets gained `--tool-call-parser` per 3090's chat-template grep mapping:
- Devstral (Mistral arch) → `mistral`
- Gemma 4 26B / 31B → `gemma4`
- Qwen3.5 / 3.6 family (every variant) → `qwen3_coder` (XML `<function=NAME>` format)
- Qwen3 base (qwen3-32b, qwen3-moe) → `qwen25` (JSON-in-tag format)

`gemma4` / `gemma4-31b` also gained `--reasoning-parser gemma4`. `nemotron-30b` gained `--reasoning-parser nemotron_3` to stop verbose thinking traces from consuming the 1024-tok MC eval budget.

**Verification:** `coder-30b-DWQ` tool-call request returns `finish_reason="tool_calls"` with structured `tool_calls[{function:{name,arguments}}]` and `content=None`. probe_codegen STRONG 8/8 with the parser wired in — no regression on plain codegen. `gemma4` reasoning parser splits 365 reasoning_tokens into `reasoning_content` (864 chars, bullet-format Gemma trace), final answer 0.05 in `content`.

### Content-aware probe trio (2026-05-13)

**Adoption:** ported 3090's `probe_thinking` / `probe_vision` / `probe_codegen` (STRONG / DEGRADED / FAIL classification). Replaced the loose keyword-grep validator as the gate for capability claims.

**First sweep findings (since fixed via patches 011/013 + parser wiring):**
- `coder-30b` (DWQ): codegen STRONG 8/8, matches 3090's `coder-reap-25b` baseline.
- `devstral`: vision FAIL (fabrication) → root-caused to patch 010 loss, fixed by patch 013 → now STRONG.
- `qwen35-9b-8bit`: vision FAIL ("pale pink") → same root cause, also STRONG after patch 013.
- `gemma4`: vision FAIL with a third signature ("Please provide the image") — image dropped at SGLang multimodal layer because mlx-community checkpoint ships without `preprocessor_config.json`. **Still blocked upstream**, tracked in main README Active work.
- `nemotron-30b`: reasoning parser worked — 154 reasoning_tokens (`"We need to solve classic puzzle: ball + bat = 1.10..."`) split cleanly out of `content`.

Per-cell receipts at `benchmarks/quality/probe-trio/*.json`. probe_all.sh sweeps all presets in one command.

### Gemma 4 radix-cache root-cause (2026-05-13)

**Symptom:** Gemma 4 first prefill crashed with `ValueError: [broadcast_shapes] Shapes (2,128) and (1,8,64) cannot be broadcast` inside `_sync_new_kv_to_pool`. Memory pressure from prior runs could mask this as a scheduler-init BrokenPipe; clean `pkill` + retry surfaces the real shape error.

**Root cause:** `MlxKVPool` assumes homogeneous per-layer attention shapes. Gemma 4 26B has 25 sliding-attention layers @ (8 KV heads × 256 dim) interleaved with 5 full-attention layers @ (2 KV heads × 512 dim — `global_head_dim` / `num_global_key_value_heads`); 31B same pattern (50 + 10). Pool gets sized for whatever `_get_attn_config` sees in layer 0 (a sliding layer) but only the full-attention layers actually write to the pool — sliding ones use `RotatingKVCache` and stay native. First full-attention prefill broadcasts `(2,128)` packed-KV into `(1,8,64)` pool slots → ValueError.

**Workaround:** `--disable-radix-cache` baked into both `gemma4` presets via `launch.sh` (commit `a8a3ff0`). Bypasses pool construction + sync write path entirely. Loses agentic prefix reuse but evals (one-shot prompts) unaffected. Already-published numbers were valid because `run_all_evals.sh` enforces radix off anyway.

**Fix-A / fix-B / fix-C** options documented in [`RADIX_CACHE_GEMMA4_ROOT_CAUSE.md`](RADIX_CACHE_GEMMA4_ROOT_CAUSE.md). Fix-A (sample a full-attention layer in `_get_attn_config`) is the minimum to restore radix support; fix-B (per-layer pool shapes) is the structural fix. Untaken pending the upstream `preprocessor_config.json` block — Gemma 4 multimodal isn't useful for agentic prefix reuse on M4 yet.
