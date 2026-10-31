# Patches — SGLang v0.5.23 on Apple Silicon

Eight patches on top of the `v0.5.23` pin (commit `297e54f`), applied in
numeric order by `scripts/setup.sh`. Numbering is sparse; retired numbers are
not reused. 002–007 and 014 form the text/processor stack; 008 adds the VLM /
hybrid (DeltaNet/Mamba2) path; 015 removes the deep-prefill growth ladder; 018
keeps the MLX auxiliary-state pool from exhausting under the radix cache.

Sampling is upstream: `--mlx-enable-sampling` (set by every `launch.sh`
preset) turns on per-request temperature/top-p/top-k/min-p, logit_bias,
grammar masks and output logprobs on the MLX path. Without the flag every
request is greedy argmax. Greedy requests keep exact argmax either way;
`top_k==1` is served greedy; repetition/frequency/presence penalties are not
applied.

## Applied patches

| # | Patch | Files | Why |
|:-:|-------|-------|-----|
| 002 | mps-backend-defaults | `arg_groups/fields/model.py`, `arg_groups/fields/mm.py`, `arg_groups/platform_hook.py`, `arg_groups/kv_cache_hook.py` | MLX `--kv-cache-dtype` aliases (`fp16`/`fp8`/`turboquant`/`turbo`/`tq`/`4bit`) — **boot-critical**: the launch default `KV_CACHE=fp8` is not in upstream's `choices`, so without them every preset is rejected by argparse. `mxfp8` is an upstream choice gated to SM100+/Blackwell; `handle_mxfp8_kv_cache_compatibility` skips that check under `use_mlx()` so the alias keeps its MLX meaning. `handle_mps_backends` declares `enable_multimodal=False` on MPS when unset (the launch presets rely on it; VLM presets pass `--enable-multimodal`), which requires `enable_multimodal` to be a `resolvable` field. |
| 003 | mlx-skip-quantization-check | `configs/model_config.py` | Wrap `self._verify_quantization()` in `if not use_mlx():` — MLX checkpoints carry `quantization_config` without `quant_method`, and the upstream raise rejects them. |
| 005 | mlx-attn-wrapper-varargs | `kv_cache/attention_wrapper.py` | `__call__(self, x, *args, **kwargs)` + transparent delegate. Devstral/ministral3 (and mlx_vlm attention) pass extra positional/keyword args (`attn_scale`, `position_ids`, …); a fixed 3-arg signature would `TypeError`. Also delegates attribute **reads** to the wrapped module via `__getattr__` — model forwards inspect attention attrs directly (mlx_vlm qwen3_5 reads `self_attn.rotary_emb.fused_apply`). |
| 007 | mlx-multimodal-and-mps-shim | `_platform_stubs.py`, `managers/mm_utils.py` | (a) `.to('cuda')`→`.to('cpu')` redirect in the MPS `Tensor.to` shim (transformers image processors default to cuda); (b) `ShmPointerMMData` slices to the logical byte/element count on both write and read — macOS rounds shm segments up to a 16 KB page, so `torch.frombuffer` returns an oversized view. |
| 008 | mlx-vlm-hybrid-integration | `mlx/model_runner.py`, `mlx/tp_worker.py`, `kv_cache/attention_contract.py`, `kv_cache/auxiliary_state.py`, `arg_groups/overrides.py`, `managers/overlap_utils.py`, `utils/hf_transformers/tokenizer.py` | The VLM / hybrid path — see the section below. |
| 014 | mlx-hf-processor-fixes | `utils/hf_transformers/processor.py`, `utils/hf_transformers/mistral_utils.py` | (a) Gemma 4 image-only processor when the checkpoint ships only `processor_config.json`. (b) Replace a `MistralCommonBackend` processor tokenizer when the checkpoint explicitly declares `tokenizer_class=TokenizersBackend` — mistral-common ignores jinja chat templates and 400s on image chat messages; mlx-community Devstral ships a full HF `tokenizer.json`. Pairs with 008's same rule in `get_tokenizer`. (c) `wrap_as_pixtral` covers `model_type=mistral3` (transformers 5.12 AutoProcessor returns a bare tokenizer for Devstral — no image processor at all) and reads `spatial_merge_size` from the top-level config where Mistral3 keeps it (missing it halves the patch grid: placeholder count ≠ merged feature count → silent single-feature merge → hallucinated image answers). |
| 015 | mlx-presize-attention-cache | `mlx/model_runner.py`, `kv_cache/attention_kv_cache.py` | `ContiguousAttentionKVCache.reserve()` + per-request pre-sizing in `_acquire_cache`/`prefill_start` (target: `origin_input_ids` + `max_new_tokens`; continuation chunks reuse the cache via `_req_caches`, so first acquisition covers the whole chunked prefill; `_release_cache` trims back to default so the reuse pool never hoards deep-request buffers). Only full-attention layers are sized — sliding-window caches are bounded by their window. Replaces the doubling ladder for known-length requests: the ladder both spikes at each grow (old+new+copy per attention layer) and overshoots to the next power of two — at 160K it holds 262,144-token bf16 buffers and dies at ~156K, while exact pre-size (163,872) completes 157,287 server-verified tokens (A/B receipts: `benchmarks/longctx-bisect/{ladder,presize}-160k-chunked2048*`). |
| 018 | mlx-aux-slot-admission-eviction | `mem_cache/allocation.py` | Admission-time auxiliary-slot eviction for the MLX pool. Every finished request donates its live snapshot slot to the radix tree, and upstream's `ensure_mamba_capacity` returns early for anything that is not a `HybridReqToTokenPool`, which the MLX pool is not — without an MLX branch the tree absorbs the entire pool and per-request allocation asserts (`Not enough MLX auxiliary state slots`) after ~pool-size requests. `_ensure_mlx_auxiliary_capacity` mirrors the Hybrid branch: `evict_for_alloc` of `3×num_reqs − free` snapshots (`MAMBA_STATE_PER_REQ_PREFIX_CACHE`). |

The MLX buffer-cache cap is upstream (`SGLANG_MLX_CACHE_LIMIT_GB`, applied
before model load) but off unless set; `scripts/common.sh` exports it at 4 GB.
Uncapped, chunked prefill retains ~0.6 MB/token of shape-shifting transient
buffers and long prefills jetsam around 30K; capped, 128K+ completes
(`benchmarks/longctx-bisect/ATTRIBUTION.md`).

## VLM / hybrid path (patch 008)

The upstream MLX backend is text-only via `mlx_lm`: it cannot load
`*ForConditionalGeneration` checkpoints (Qwen3.5/3.6/3.8, Devstral, Gemma 4),
its attention contract requires an attribute named `rope` (mlx_vlm names it
`rotary_emb`; NemotronH attention is position-free), and its hybrid-SSM radix
machinery assumes the CUDA model runner. Patch 008 closes that:

1. **mlx_vlm loader + `_TextOnlyVLMShim`** (`mlx/model_runner.py`) — VLM
   detection from the resolved checkpoint's `config.json`, after the
   remote-code gate. Model-only load (the mlx_vlm processor is never used —
   SGLang's mm pipeline supplies features); text requests route to
   `language_model` and the native cache machinery unwraps via `__getattr__`.
   Needs `pip install mlx-vlm --no-deps` (see setup.sh). Image-bearing calls
   (pixel_values) go through the full VLM forward. VLMs run without a headless
   trunk (`_trunk = None`): mlx_vlm's inner text model skips the language
   model's mRoPE position-id computation, so non-final chunks run the full
   forward.
2. **Attention detection** (`kv_cache/attention_contract.py`) — a softmax
   `scale` is the attention marker (excludes DeltaNet); `rope` is not required,
   so `rotary_emb` (mlx_vlm) and position-free (NemotronH) attention are
   detected.
3. **Serial decode routing** (`mlx/model_runner.py`) — an
   `_attention_wrapper_batchable` flag (inner has `.rope`); rotary_emb /
   rope-less attention decodes through `_decode_with_native_cache` (the
   model's own forward), since the batched wrapper path calls `inner.rope`.
   Batched decode is an MR>1 throughput optimization only.
4. **MLX auxiliary-state radix component** (`kv_cache/auxiliary_state.py`) —
   `MlxAuxiliaryStateComponent` skips `MambaComponent.__init__` (it asserts a
   `HybridReqToTokenPool`) and mirrors every attribute it sets
   (`mamba_cache_chunk_size`, `mamba_checkpoint_grid`,
   `mamba_max_states_per_path`); the base insert path reads them on the first
   finished request. It also overrides `finalize_match_result_in_cache` to
   apply the prefix-match copy-on-write eagerly: the base component stages a
   deferred COW that the general model runner applies on-device inside the
   forward pass, which never runs on MLX (host-side snapshot dict,
   stream-free).
5. **`no_buffer` mamba-radix strategy on MLX** (`arg_groups/overrides.py`) —
   the `auto` strategy resolves to `extra_buffer`, which needs CUDA/MUSA/NPU
   FLA kernels. On MLX, `auto` resolves to `no_buffer` +
   `disable_overlap_schedule=True` — hybrid models run the normal event loop
   when the radix cache is on.
6. **Device-aware FutureMap stash/publish** (`managers/overlap_utils.py`) —
   the normal event loop relays MLX host tensors into device-allocated buffers;
   `.to()` pins dtype **and** device (no-op on CUDA).
7. **Vision (`pixel_values`) plumbing** — `tp_worker._extract_mm_inputs`
   (multi-item feature concat + bf16→float32 numpy bridge +
   `model_specific_data` stacking) at the `prefill_start` call site;
   `prefill_start(pixel_values, mm_kwargs)` threads them into
   `_forward_lazy_token`, which forces the full model whenever image kwargs are
   present. Image-bearing prefills take the full-prompt path on any prefix hit
   and skip the mid-prompt tracked-state split — the VLM feature merge only
   sees the input_ids it is called with. The qwen3_5
   `_position_ids`/`_rope_deltas` caches reset at every new prefill.
8. **NemotronH support** — `_extract_model_components` understands
   `backbone.embeddings`/`norm_f`; `_CompactCacheModel` adapts NemotronH's
   compact `make_cache()` (one entry per M/* layer) to the runner's
   full-length `cache[layer_idx]` contract with null placeholders at
   cache-less MoE/MLP layers, compacted back on every forward.
9. **Declared-tokenizer-class guard** (`utils/hf_transformers/tokenizer.py`)
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
| qwen35 | STRONG | STRONG | PARTIAL | VERIFIED |
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
cd components/sglang && git checkout v0.5.23
for p in ../../patches/0[01][0-9]-*.patch; do git apply "$p"; done   # 8 patches, all clean
```
