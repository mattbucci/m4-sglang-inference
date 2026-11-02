# Patches — SGLang v0.5.23 on Apple Silicon

Fifteen patches on top of the `v0.5.23` pin (commit `297e54f`), applied in
numeric order by `scripts/setup.sh`. Numbering is sparse; retired numbers are
not reused. 002–007 and 014 form the text/processor stack; 008 adds the VLM /
hybrid (DeltaNet/Mamba2) path; 015 removes the deep-prefill growth ladder; 018
keeps the MLX auxiliary-state pool from exhausting under the radix cache; 019
runs Recursive Flint (sliding-window + attention-sink MoE) on the MLX path; 020
computes prefill logits for the last chunk position only; 021 gives
sliding-window models an exact prefix cache (windowed KV snapshots through the
auxiliary-state pool); 022 makes the pool-backed attention cache append across
the two forwards of a prefix hit's final chunk; 023 releases a finished
request's MLX caches at the scheduler's KV-release hook; 024 makes the
MLX wired-memory limit configurable.

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
| 019 | mlx-flint-support | `configs/flint.py`, `configs/__init__.py`, `utils/hf_transformers/common.py`, `model_loader/utils.py`, `hardware_backend/mlx/text_checkpoint.py`, `hardware_backend/mlx/model_runner.py`, `kv_cache/attention_contract.py`, `kv_cache/attention_wrapper.py`, `parser/reasoning_parser.py`, `parser/reasoning_parser_names.py`, `function_call/flint_detector.py`, `function_call/function_call_parser.py`, `function_call/parser_names.py` | Recursive Flint (`FlintForCausalLM`: 35 sliding-window + 14 full-attention layers, learned attention sinks, NoPE full layers, 256-expert MoE) on the MLX path. (a) `FlintConfig` registered with `AutoConfig` — transformers has no `flint` model type and the checkpoint ships no remote code, so `get_config` would reject it. (b) `load_text_model`: mlx-lm `load_model` through a `sanitize` adapter that strips mlx-vlm's `language_model.` prefix from the weights **and** the per-tensor quantization overrides (oMLX's oQ*e conversions go through mlx-vlm's model class) and drops the all-zero `mlp.expert_bias` tensors mlx-vlm synthesizes and mlx-lm's Flint declares no parameter for; the tokenizer is no longer loaded twice. (c) The per-layer window map falls back to the mlx-lm `ModelArgs` on `container.args` (`layer_types` + `sliding_window`) — Flint declares its 513-token windows only there, and without a map the runner raises `NotImplementedError` at construction. (d) Batched decode: learned sinks stored as `softmax_offset` (gpt_oss: `sinks`) reach SDPA, and NoPE layers (`rope is None`) skip rotation instead of crashing. (e) `--reasoning-parser flint` (`<think>` only where the model writes it; never forced; no template toggle) and `--tool-call-parser flint` (GLM-4.7's XML call format without line breaks). (f) `get_model_architecture` returns a stand-in class on MLX for architectures the Torch registry lacks — the scheduler's KV-cache builder and the tokenizer manager resolve the architecture even though no Torch model is ever built, and `resolve_transformers_arch` otherwise aborts the boot. Also adds `model_loader/utils.py` to the file list. Needs mlx-lm at or past its Flint commit (git pin in `setup.sh`). |
| 020 | mlx-prefill-last-token-logits | `hardware_backend/mlx/model_runner.py` | Final-chunk prefill ran the full model on the whole chunk and materialised `[chunk, vocab]` logits that were sliced to one row — 1.2 GB bf16 at chunk 4096 on a 150K vocabulary, the largest final-chunk transient (Flint oQ8e at 42.3 GB resident has ~9 GB of headroom at 27K tokens). `_last_position_logits` splits the final chunk when a headless trunk exists: trunk on all but the last token (cache side effects only), then the full model on the last token, so every post-head transform (softcapping, scaling) still runs and only `[1, vocab]` is produced. Consecutive forwards on one cache are exactly chunked prefill, so the result is the single-call result; VLM shims (no trunk) and NemotronH are unchanged. |
| 021 | mlx-swa-prefix-snapshots | `hardware_backend/mlx/swa_prefix.py`, `hardware_backend/mlx/model_runner.py`, `hardware_backend/mlx/model_runner_stub.py`, `hardware_backend/mlx/tp_worker.py`, `kv_cache/attention_kv_cache.py`, `kv_cache/auxiliary_state.py`, `mem_cache/kv_cache_builder.py`, `arg_groups/overrides.py` | Exact prefix cache for sliding-window models. Upstream pools full-attention KV only, so a radix hit on a model with sliding-window layers recomputed the whole prefix (a trailing-band rebuild is inexact: each rebuilt position needs its own window of exact hidden states, chaining back through every layer), which left Flint with the radix cache off and every agent turn re-prefilling its context. A sliding layer only ever reads its last `window` tokens, so with the radix cache on the 35 `WindowedAttentionKVCache`s join the auxiliary snapshot set: `snapshot_state()` stores the kept window (1 MB/layer at 513 tokens, 4 KV heads) at the boundaries the tree names — chunk ends and the 64-token mamba-checkpoint grid — and a `state` setter restores it on an exact hit while the 14 full layers gather from the pool, which `init_cache_pools` now allocates for such models. The scheduler side gets the hybrid bookkeeping through one predicate (`mlx_snapshots_sliding_window_prefix`: MLX, radix on, `layer_types` with `sliding_attention`): `uses_ssm_state` turns on the MAMBA tree component and the MLX auxiliary-state req-to-token pool, and `_mamba_radix_cache_resolution` declares `no_buffer` (overlap schedule off, as for DeltaNet/Mamba2 presets). The pool scatters of every chunk (`_sync_new_kv_to_pool`) now go out with the step's `mx.async_eval` (`pending_pool_arrays`): nothing read the pool until a prefix hit, so the scatters chained lazily across chunks and requests, pinning the request KV buffers they read (whose next in-place write then copied the whole buffer) and the pool versions they replaced — a cold 13K-token prefill bottomed 6 GB below the radix-off run and the pinned buffers outlived a cache flush; evaluated per step the same prefill sits 2.6 GB below radix-off, the pool itself. `--disable-radix-cache` restores the previous recompute path with no snapshots. `scripts/eval/test_swa_snapshot_unit.py` is the CPU round-trip test (snapshot at a boundary, keep writing, restore, replay: identical window, offset and mask). |
| 022 | mlx-pool-backed-cache-append | `kv_cache/attention_kv_cache.py`, `hardware_backend/mlx/model_runner.py` | A prefix hit runs its new tokens on `PoolBackedAttentionKVCache`, and 020 computes the final chunk as two consecutive forwards on that cache (trunk on all but the last token, model on the last). The upstream cache re-gathered `slots[:offset]` on the second call — the pool holds nothing for the first call's tokens until the request syncs after the forward — so the last token attended to the prefix plus itself, the contiguous cache materialised from it held prefix + 1 tokens, and every sync after that sliced past the valid rows and wrote stale buffer contents into the pool for the new positions, which every later hit on that prefix gathered. The symptom was agentic: Flint's SWE-bench turns with the radix cache on produced runaway tool calls at 13 in ~254 turns against 0 in ~395 with it off, and a hit with 512 new tokens crashed the server (`broadcast_shapes (192,4,128) vs (512,4,128)` in `_sync_new_kv_to_pool`). The cache now keeps the chunks already fetched in front of the next one, as every other cache does, and `_materialize_pool_backed_attention` pre-sizes the contiguous cache like a cold prefill. Reaches every model with the radix cache on; `scripts/eval/test_pool_backed_cache_unit.py` is the CPU test (11 keys instead of 74 on the unfixed cache). Upstream candidate: the cache class is upstream, the split that exposes it is 020. |
| 023 | mlx-release-finished-request | `hardware_backend/mlx/tp_worker.py`, `managers/scheduler_components/batch_result_processor.py` | The MLX worker drops a request's per-request caches (the contiguous full-attention KV, 28 KB/token on Flint, plus the windowed caches) only in the stale-rid sweep of the next decode batch, and the scheduler calls the worker's KV-release hook only for finished *decoding* requests. A request that finishes in its prefill batch (one-token requests: logprob scoring, the capability probes, the hit-vs-cold probes) is never swept: its KV stays resident for the server's lifetime, 0.28 GB per 10K-token request, through cache flushes, until Metal's 51.8 GB working set trips (`kIOGPUCommandBufferCallbackErrorOutOfMemory` after 66 probe requests on oQ8e). A decode-finished request lingers until the next decode batch, so the last turn's KV (1.8 GB at 64K) sits idle between agent turns. The scheduler calls the hook for every finished request and the hook drops the runner state there; the decode-only work (leaf re-snapshot, tracked-slot clear, row sync) runs only when the request decoded, and the sweep stays as a backstop. Under the overlap schedule (radix off) the next decode step is already launched when a request's finish is processed, so the worker counts in-flight launches per request and drops the state when the last one is finalized (immediate removal there was a `KeyError` in `decode_batch_finalize`). Wired memory holds at 46.5 GB across repeated one-token hit cycles that climb 52 to 54 GB and crash without it. `scripts/eval/test_release_finished_request_unit.py` is the CPU test. Upstream candidate: the sweep and the guarded hook call are both upstream. |
| 024 | mlx-wired-limit-env | `hardware_backend/mlx/model_runner.py`, `environ.py` | `SGLANG_MLX_WIRED_LIMIT_GB`: unset keeps upstream's `mx.set_wired_limit(max_recommended_working_set_size)`; `0` skips the call; a positive value wires at most that many GB. With a nonzero wired limit MLX adds every newly created Metal buffer to a residency set and commits the set on each add and each release (`ResidencySets::insert`/`erase` in `mlx/backend/metal/resident.cpp`); with the limit at 0 `insert` returns before touching the set, so no residency-set traffic reaches the driver. That traffic is the discriminating factor in the IOGPUFamily `completeMemory() prepare count underflow` kernel panic in a third-party single-variable test on an M4 (stock: panic 3/3 at ~100 s; no `set_wired_limit`: 0 panics in 10 runs over 4.2 h; ≤3 % sequential throughput cost; [mlx#3186](https://github.com/ml-explore/mlx/issues/3186)). `scripts/common.sh` defaults it to `0`. Trade-off: weights and KV become pageable, so throughput can drop under external memory pressure. |
| 025 | dllm-structured-reads | `model_executor/forward_batch_info.py`, `dllm/algorithm/base.py`, `dllm/algorithm/gemma4_renoise.py`, `dllm/mixin/scheduler.py` | Structured reads on DiffusionGemma (the shape of vLLM's `structured_diffusion` example): per-request options in `sampling_params.custom_params["diffusion"]` — `seed_canvas` (the answer template as `canvas_length` token ids, replacing the random canvas), `pinned` (positions held at their seed on every denoising step and in the emitted canvas), `max_steps` (per-request cap), `read_only` (emit the argmax canvas after the cap, finish after one block, and return temperature-1 logprobs at every canvas position for the request's `token_ids_logprob`), `constrained` (restrict sampling, argmax and self-conditioning to those ids; logprobs renormalize over the set). Upstream's `Gemma4Renoise` rejects logprobs and all structured-output options, so a read needs this patch; the options ride on the forward batch (`dllm_options`) and the read's logprobs come back through the algorithm state to the diffusion scheduler mixin. Hardware-agnostic Python, but DiffusionGemma itself is CUDA/ROCm-only in SGLang (`configure_server_args` rejects any other device, needs the Triton backend, no quantized loading), so this box cannot run it: validated here by `scripts/eval/test_dllm_structured_reads_unit.py` (synthetic logits through the renoise loop) and the import smoke; end-to-end on the R9700 (`experiments/` next steps in the sister repos). Client: `scripts/eval/structured_reads.py`. |
| 026 | mlx-extend-chunk-mamba-track | `hardware_backend/mlx/model_runner.py`, `hardware_backend/mlx/tp_worker.py` | The MLX runner tracks a Mamba/DeltaNet state at the last 64-token grid boundary of a prefill (`_select_auxiliary_state_track_len`), and the scheduler's end-of-prefill checkpoint publishes it as the deepest tree node — but only `prefill_start` (the first chunk) tracked; `extend_start` (every continuation chunk) ran the chunk in one forward, so a prompt longer than `CHUNKED` kept states only at chunk ends and at the prompt end. The chat template re-renders a history turn without its reasoning, so a follow-up diverges 1 token (Qwen3.8) or 2 tokens (Qwen3.6) before the primed prompt end: the end state never serves an agentic turn and the whole tail of the last prefill chunk is recomputed on the first follow-up (the R9700 found the same under `no_buffer` and moved to `extra_buffer`, which needs the FLA kernels MLX does not have). Continuation chunks now split at the grid boundary and track the state exactly as the first chunk does (`req` is handed to `extend_start`; image-bearing requests are not split), so the first follow-up finds a state within a grid of the prompt end: on Qwen3.8-27B the first follow-up recomputes 55–77 tokens instead of 525–1,347 (TTFT 0.9–1.1 s instead of 5–14 s at 8.7K–52K), on qwen36 21–73 instead of 479–2,889 (0.2–0.5 s instead of 0.8–8.2 s); later turns were already served by the grid state (`benchmarks/session-endurance/VERDICT.md`, Phase C). A hit from a mid-chunk state differs from a cold run the way two cold runs on different chunk sizes do. CPU test: `scripts/eval/test_extend_chunk_track_unit.py`. |

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
and `flint` run with the radix cache. Determinism, measured on 3-turn greedy
conversations at 3K–14K tokens: repeated cold runs are token-identical, and
repeated hits on the same cached prefix are token-identical. A hit and a cold
run of the same prompt differ only as much as two cold runs on different
chunk sizes differ: the new tokens prefill from the cached prefix length on
a chunk boundary the cold run never had, which perturbs bf16 logits at the
noise level. Over 24 positions of a 16.6K-token agent request on Flint the
top token matches at 20–21 of 24 for hits and at 20 of 24 for a cold run at
chunk 2048 versus 1024, median top-1 shift 0.016–0.048 nats in every arm,
with every mismatch at a position where the cold top token sits below 0.75
probability (`benchmarks/flint/2026-10-31/prefix-cache/RESULTS.md`).
On the three-turn conversations Flint's text is identical at 3K and 6.9K
tokens and diverges in reasoning prose at 13.4K; qwen36's is identical at
3.2K and 14.4K and drops a trailing list item at 7.4K — near-ties either
way, the top token's logprob moving ≤ 0.01 nats. Branched suffixes
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
for p in ../../patches/0[012][0-9]-*.patch; do git apply "$p"; done   # 16 patches, all clean
```
