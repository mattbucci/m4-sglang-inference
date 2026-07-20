# Patches — SGLang v0.5.15.post1 on Apple Silicon

Seven patches on top of the `v0.5.15.post1` pin (commit `0b3bb0c`), applied in
numeric order by `scripts/setup.sh`. Numbering is sparse; retired numbers are
not reused. 002–015 minus 008 form the text stack; 008 adds the VLM / hybrid
(DeltaNet/Mamba2) path; 015 removes the deep-prefill growth ladder.

## Applied patches

| # | Patch | Files | Why |
|:-:|-------|-------|-----|
| 002 | mps-backend-defaults | `server_args.py` | Default `enable_multimodal=False` on MPS (launch presets depend on it) + MLX `--kv-cache-dtype` aliases (`fp8`/`mxfp8`/`turboquant`/`tq`/`4bit`). The aliases are **boot-critical**: the launch default `KV_CACHE=fp8` is not in upstream's `choices`, so without them every preset is rejected by argparse. |
| 003 | mlx-skip-quantization-check | `configs/model_config.py` | Wrap `self._verify_quantization()` in `if not use_mlx():` — MLX checkpoints carry `quantization_config` without `quant_method`, and the upstream raise rejects them. |
| 005 | mlx-attn-wrapper-varargs | `kv_cache/attention_wrapper.py` | `__call__(self, x, *args, **kwargs)` + transparent delegate. Devstral/ministral3 (and mlx_vlm attention) pass extra positional/keyword args (`attn_scale`, `position_ids`, …); a fixed 3-arg signature would `TypeError`. Batched decode uses `inner.scale`. Also delegates attribute **reads** to the wrapped module via `__getattr__` — model forwards inspect attention attrs directly (mlx_vlm qwen3_5 reads `self_attn.rotary_emb.fused_apply`). |
| 007 | mlx-multimodal-and-mps-shim | `_mps_stub.py`, `managers/mm_utils.py`, `managers/schedule_batch.py`, `multimodal/processors/pixtral.py` | (a) `.to('cuda')`→`.to('cpu')` redirect; (b) `ShmPointerMMData` slice to logical byte/element count (macOS 16 KB shm page rounding); (c) `Modality.MULTI_IMAGES` enum member + pixtral multi-image splitter matches it (StopIteration → HTTP 500 otherwise). |
| 008 | mlx-vlm-hybrid-integration | `mlx/model_runner.py`, `mlx/tp_worker.py`, `kv_cache/attention_contract.py`, `kv_cache/auxiliary_state.py`, `model_executor/model_runner.py`, `arg_groups/overrides.py`, `managers/overlap_utils.py`, `utils/hf_transformers/tokenizer.py` | The VLM / hybrid path — see the section below. `mlx_vlm` loading, attention detection for `rotary_emb`/rope-less archs, the scheduler's mamba-allocator contract on the MLX aux pool, radix cache for hybrid models (`no_buffer` strategy), vision (`pixel_values`) plumbing, NemotronH support, and the **MLX buffer-cache cap** (`SGLANG_MLX_CACHE_LIMIT_GB`, default 4 GB — uncapped, chunked prefill retains ~0.6 MB/token of shape-shifting transient buffers and long prefills jetsam around 30K; capped, 128K completes — `benchmarks/longctx-bisect/ATTRIBUTION.md`). |
| 014 | mlx-hf-processor-fixes | `utils/hf_transformers/processor.py`, `utils/hf_transformers/mistral_utils.py` | (a) Gemma 4 image-only processor when the checkpoint ships only `processor_config.json`. (b) Replace a `MistralCommonBackend` processor tokenizer when the checkpoint explicitly declares `tokenizer_class=TokenizersBackend` — mistral-common ignores jinja chat templates and 400s on image chat messages; mlx-community Devstral ships a full HF `tokenizer.json`. Pairs with 008's same rule in `get_tokenizer`. (c) `wrap_as_pixtral` covers `model_type=mistral3` (transformers 5.12 AutoProcessor returns a bare tokenizer for Devstral — no image processor at all) and reads `spatial_merge_size` from the top-level config where Mistral3 keeps it (missing it halves the patch grid: placeholder count ≠ merged feature count → silent single-feature merge → hallucinated image answers). |
| 015 | mlx-presize-attention-cache | `mlx/model_runner.py`, `kv_cache/attention_kv_cache.py` | `ContiguousAttentionKVCache.reserve()` + per-request pre-sizing in `_acquire_cache`/`prefill_start` (target: `origin_input_ids` + `max_new_tokens`; continuation chunks reuse the cache via `_req_caches`, so first acquisition covers the whole chunked prefill; `_release_cache` trims back to default so the reuse pool never hoards deep-request buffers). Replaces the doubling ladder for known-length requests: the ladder both spikes at each grow (old+new+copy per attention layer) and overshoots to the next power of two — at 160K it holds 262,144-token bf16 buffers and dies at ~156K, while exact pre-size (163,872) completes 157,287 server-verified tokens (A/B receipts: `benchmarks/longctx-bisect/{ladder,presize}-160k-chunked2048*`). Also halves the cache at 32K-class depths (exact size vs next-power-of-two). |

## VLM / hybrid path (patch 008)

The upstream MLX backend is text-only via `mlx_lm`: it cannot load
`*ForConditionalGeneration` checkpoints (Qwen3.5/3.6, Devstral, Gemma 4), its
attention contract requires an attribute literally named `rope` (mlx_vlm names
it `rotary_emb`; NemotronH attention is position-free), and its hybrid-SSM
scheduler machinery assumes CUDA-side pools. Patch 008 closes all of that:

1. **mlx_vlm loader + `_TextOnlyVLMShim`** (`mlx/model_runner.py`) — VLM-detect
   `_load_model` (model-only load; the mlx_vlm processor is never used —
   SGLang's mm pipeline supplies features), wrap so text requests route to
   `language_model` and the native cache machinery unwraps via `__getattr__`.
   Needs `pip install mlx-vlm --no-deps` (see setup.sh). Image-bearing calls
   (pixel_values) go through the full VLM forward.
2. **Attention detection** (`kv_cache/attention_contract.py`) — require
   `scale` (the softmax marker that excludes DeltaNet), make `rope` optional so
   `rotary_emb` (mlx_vlm) and position-free (NemotronH) attention are detected.
3. **Serial decode routing** (`mlx/model_runner.py`) — a
   `_attention_wrapper_batchable` flag (inner has `.rope`); route
   `rotary_emb`/rope-less attention to `_decode_with_native_cache` (delegates to
   the model's own forward). Batched decode is an MR>1 throughput optimization
   only; correctness holds either way.
4. **Skip general attn-backend init on MLX** (`model_executor/model_runner.py`) —
   `init_attention_backends` no-ops on `use_mlx()`. The MLX tp worker overrides
   `forward_batch_generation`, so the general backends are unused; skipping also
   avoids `GDNAttnBackend.__init__` crashing on the MLX aux-state pool.
5. **Mamba-allocator contract on `MlxAuxiliaryStatePool`**
   (`kv_cache/auxiliary_state.py`) — the scheduler drives hybrid-SSM slot
   allocation through `mamba_allocator.alloc_group_begin/end` in the
   batch-admission path. The MLX pool implements the full allocator surface
   (`alloc_group_begin/end`, `alloc`/`_do_alloc` iterator batching,
   `schedulable_available_size`, `mamba_allocator` alias), mirroring upstream
   `MambaSlotAllocator` semantics.
6. **Eager copy-on-write on prefix match** (`kv_cache/auxiliary_state.py`) —
   the base `MambaComponent.finalize_match_result` stages a deferred COW that
   the general model runner applies on-device inside the forward pass, which
   never runs on MLX. `MlxAuxiliaryStateComponent` overrides it to apply the
   snapshot-dict copy eagerly (host-side, stream-free).
7. **`no_buffer` mamba-radix strategy on MLX** (`arg_groups/overrides.py`) —
   the `auto` strategy resolves to `extra_buffer`, which asserts CUDA/MUSA/NPU
   (FLA kernels). On MLX, `auto` resolves to `no_buffer` +
   `disable_overlap_schedule=True` — hybrid models run the normal event loop
   instead of the MLX overlap loop when the radix cache is on.
8. **Device-aware FutureMap stash/publish** (`managers/overlap_utils.py`) —
   the normal event loop relays MLX host tensors into device-allocated buffers;
   `.to()` pins dtype **and** device (no-op on CUDA).
9. **Vision (`pixel_values`) plumbing** — `tp_worker._extract_mm_inputs`
   (multi-item feature concat + bf16→float32 numpy bridge +
   `model_specific_data` stacking), `prefill_start(pixel_values, mm_kwargs)`
   with image-aware rules (image-bearing prefills take the full-prompt path on
   any prefix hit and skip the mid-prompt tracked-state split — the VLM
   feature merge only sees the input_ids it is called with), and the qwen3_5
   `_position_ids`/`_rope_deltas` reset at every new prefill.
10. **NemotronH support** — `_extract_model_components` understands
   `backbone.embeddings`/`norm_f`; `_CompactCacheModel` adapts NemotronH's
   compact `make_cache()` (one entry per M/* layer) to the runner's
   full-length `cache[layer_idx]` contract with null placeholders at
   cache-less MoE/MLP layers, compacted back on every forward.
11. **Declared-tokenizer-class guard** (`utils/hf_transformers/tokenizer.py`)
   — when `tokenizer_config.json` explicitly declares
   `tokenizer_class=TokenizersBackend`, skip the use_fast=False re-resolution
   that would downgrade to `MistralCommonBackend` (breaks custom jinja
   templates, image chat, and XGrammar). Pairs with patch 014's processor-side
   rule.

Hybrid presets (qwen35 / qwen35-9b-8bit / qwen36 / qwen36-27b / nemotron-30b)
run with the radix cache. Greedy-determinism holds: a prefix-cache hit
produces token-for-token identical output to a cold run, and branched suffixes
(eager-COW path) answer correctly.

**Probe matrix (radix on):**

| preset | codegen | vision | video | thinking |
|--------|---------|--------|-------|----------|
| qwen36 | STRONG | STRONG | STRONG | VERIFIED |
| qwen35 | STRONG | STRONG | PARTIAL | skipped (known greedy loop) |
| devstral | STRONG | STRONG | PARTIAL | — |
| nemotron-30b | STRONG | — | — | VERIFIED |

**Trade-off (measured, `benchmarks/radix-ab/VERDICT.md`):** radix-on-hybrid
forces `disable_overlap_schedule` (upstream `no_buffer` constraint), i.e. the
normal event loop — a 15–21% MR=1 decode cost on qwen36 (48.6 vs 57.3 tok/s
at 32K). The prefix-cache win dominates for the agentic workload: a full
prefix hit answers in 204 ms vs an 11.2 s cold 8K prefill (55×). Use
`--disable-radix-cache` for single-shot deep-prefill work — faster decode and
a ~4.5 GB safer memory envelope (radix-on retains ~0.4 GB/request outside the
static pool).

## Known gaps

- **gemma4 / gemma4-31b** — upstream `_attention_kv_config_for_layer` raises
  `NotImplementedError("...does not support sliding-window attention yet...")`
  at runner construction, so these raise at boot. Sliding-window support is an
  upstream feature gap; needs per-layer/window-aware pools.

## Apply / sanity check

```bash
cd components/sglang && git checkout v0.5.15.post1
for p in ../../patches/0[01][0-9]-*.patch; do git apply "$p"; done   # 7 patches, all clean
```
