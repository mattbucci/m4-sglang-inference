# Patches — SGLang v0.5.15.post1 on Apple Silicon

**Rebased onto v0.5.15.post1 on 2026-07-19** (commit `0b3bb0c`), from the prior
v0.5.12 pin (`127b9e328`). This was a **large** rebase: v0.5.15.post1 shipped a
major MLX-backend refactor that **upstreamed much of our hybrid/cache work**.

## What v0.5.15.post1 upstreamed (patches DROPPED)

The MLX backend gained native machinery that subsumes several patches:

| new upstream file | what it provides | our dropped patch |
|-------------------|------------------|-------------------|
| `kv_cache/attention_contract.py` | duck-typed attention + multi-name head-count detection (`n_heads`/`num_attention_heads`, `n_kv_heads`/`num_key_value_heads`/`num_k_heads`) + DeltaNet-vs-attention discrimination | **009** (mlx-nemotron-h head-count duck-typing) — *for mlx_lm models only, see caveat* |
| `kv_cache/attention_kv_cache.py` | `AttentionOffsetCache` (has `make_mask`/`state`), `ContiguousAttentionKVCache`, `PoolBackedAttentionKVCache` | **006** (offsetcache + make_mask) |
| `kv_cache/auxiliary_state.py` | `MlxAuxiliaryStatePool` / `MlxAuxiliaryStateComponent(MambaComponent)` — native Mamba/DeltaNet recurrent state in the unified radix cache | hybrid half of **004**, **019** |
| `kv_cache/layout.py` | `MlxModelCacheLayout` attention/auxiliary layer split | **012** (sync-pool skip-non-contiguous), pool-filter half of **020** |
| `kv_cache/model_patching.py` | `find_attention_layers` via the contract + `language_model` (VLM) nesting | routing half of **004**, **009** |
| `kv_cache/attention_wrapper.py` | `MLXAttentionWrapper._batched_decode` natively does gated-attn split+sigmoid **and** an AOT Metal RoPE kernel | **011** (gated-attn batched decode) |

Also new upstream: `aot.py` (compiled Metal kernels), `moe/fused_swiglu.py`,
`profiler.py`, `parent_watchdog.py`. And upstream added the MPS defaults our
patch 002 used to set (torch_native attention resolver, "torch_native ⇒ disable
cuda graph" rule, MPS entry in the piecewise-incompatibility list), so patch 002
collapsed to ~2 lines.

**⚠ CAVEAT — 009/011 were upstreamed for `mlx_lm` text models, NOT for
`mlx_vlm` / NemotronH.** The native contract requires an attribute literally
named `rope`. mlx_vlm attention names it `rotary_emb`; NemotronH attention is
position-free (no rope). So on v0.5.15.post1 the native detection misses those
architectures. See "VLM / hybrid path (WIP)" below.

## Applied patches

Six patches, applied in numeric order by `scripts/setup.sh`. 002–014 minus 008
form the text stack; 008 adds the VLM / hybrid (DeltaNet/Mamba2) path.

| # | Patch | Files | Why (rebased against v0.5.15.post1) |
|:-:|-------|-------|-----|
| 002 | mps-backend-defaults | `server_args.py` | **Collapsed.** Only survivors: default `enable_multimodal=False` on MPS (launch presets depend on it) + MLX `--kv-cache-dtype` aliases (`fp8`/`mxfp8`/`turboquant`/`tq`/`4bit`). The aliases are **boot-critical**: the launch default `KV_CACHE=fp8` is not in upstream's `choices`, so without them every preset is rejected by argparse. (torch_native + cuda-graph disable are now upstream.) |
| 003 | mlx-skip-quantization-check | `configs/model_config.py` | Wrap `self._verify_quantization()` in `if not use_mlx():` — MLX checkpoints carry `quantization_config` without `quant_method`, and the upstream raise is still live. |
| 005 | mlx-attn-wrapper-varargs | `kv_cache/attention_wrapper.py` | `__call__(self, x, *args, **kwargs)` + transparent delegate. Devstral/ministral3 (and mlx_vlm attention) pass extra positional/keyword args (`attn_scale`, `position_ids`, …); the fixed 3-arg upstream signature would `TypeError`. Batched decode uses `inner.scale`. Also delegates attribute **reads** to the wrapped module via `__getattr__` — model forwards inspect attention attrs directly (mlx_vlm qwen3_5 reads `self_attn.rotary_emb.fused_apply`). |
| 007 | mlx-multimodal-and-mps-shim | `_mps_stub.py`, `managers/mm_utils.py`, `managers/schedule_batch.py` | (a) `.to('cuda')`→`.to('cpu')` redirect; (b) `ShmPointerMMData` slice to logical byte/element count (macOS 16 KB shm page rounding); (c) `Modality.MULTI_IMAGES` enum member. None upstreamed. |
| 008 | mlx-vlm-hybrid-integration | `mlx/model_runner.py`, `kv_cache/attention_contract.py`, `kv_cache/auxiliary_state.py`, `model_executor/model_runner.py`, `arg_groups/overrides.py`, `managers/overlap_utils.py` | The VLM / hybrid path — see the section below. Restores `mlx_vlm` loading (removed upstream), fixes attention detection for `rotary_emb`/rope-less archs, implements the scheduler's mamba-allocator contract on the MLX aux pool, and enables the radix cache for hybrid models (`no_buffer` strategy). |
| 014 | mlx-gemma4-image-only-processor | `utils/hf_transformers/processor.py` | Applied **clean** against v0.5.15.post1 (only patch that did). Builds a Gemma 4 image-only processor when the checkpoint ships only `processor_config.json`. |

### Dropped as upstreamed (not re-derived): 006, 009†, 011, 012, 019
† 009's mlx_lm head-count detection is upstream; its NemotronH-specific handling
is not — patch 008 re-adds it via the attention contract.

## VLM / hybrid path (patch 008 — LANDED 2026-07-19)

v0.5.15 removed the `mlx_vlm` loader path entirely (the upstream MLX backend is
100% text-only), so `*ForConditionalGeneration` checkpoints (Qwen3.5/3.6,
Devstral, Gemma 4) couldn't load even for text, and NemotronH hit a rope-less
attention detection gap. Patch 008 restores all of it as **new v0.5.15
integration work** (not a rebase of any old patch):

1. **mlx_vlm loader + `_TextOnlyVLMShim`** (`mlx/model_runner.py`) — VLM-detect
   `_load_model`, wrap so text requests route to `language_model` and the native
   cache machinery unwraps via `__getattr__`. Needs `pip install mlx-vlm`
   (`--no-deps`; see setup.sh). Image-bearing calls (pixel_values) go through
   the full VLM forward.
2. **Attention detection compat** (`kv_cache/attention_contract.py`) — require
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
   (`kv_cache/auxiliary_state.py`) — the v0.5.15 scheduler drives hybrid-SSM
   slot allocation through `mamba_allocator.alloc_group_begin/end` in the
   batch-admission path. The MLX pool now implements the full allocator surface
   (`alloc_group_begin/end`, `alloc`/`_do_alloc` iterator batching,
   `schedulable_available_size`, `mamba_allocator` alias), mirroring upstream
   `MambaSlotAllocator` semantics.
6. **Eager copy-on-write on prefix match** (`kv_cache/auxiliary_state.py`) —
   the base `MambaComponent.finalize_match_result` stages a deferred COW that
   the general model runner applies on-device inside the forward pass, which
   never runs on MLX. `MlxAuxiliaryStateComponent` overrides it to apply the
   snapshot-dict copy eagerly (host-side, stream-free).
7. **`no_buffer` mamba-radix strategy on MLX** (`arg_groups/overrides.py`) —
   the `auto` strategy resolved to `extra_buffer`, which asserts CUDA/MUSA/NPU
   (FLA kernels). On MLX, `auto` now resolves to `no_buffer` +
   `disable_overlap_schedule=True` — hybrid models run the normal event loop
   instead of the MLX overlap loop when the radix cache is on.
8. **Device-aware FutureMap stash/publish** (`managers/overlap_utils.py`) —
   the normal event loop relays MLX host tensors into device-allocated buffers;
   `.to()` now pins dtype **and** device (no-op on CUDA).

**Result: hybrid models get the radix cache for the first time on M4.**
Greedy-determinism validated on qwen36: a 512-token prefix-cache hit produces
token-for-token identical output to the cold run, and a branched suffix (eager
COW path) answers correctly. Presets for qwen35 / qwen35-9b-8bit / qwen36 /
qwen36-27b / nemotron-30b no longer pass `--disable-radix-cache`.

**Trade-off:** radix-on-hybrid forces `disable_overlap_schedule` (upstream
`no_buffer` constraint), i.e. the normal event loop. For agentic multi-turn
workloads the prefix-cache win (whole prefills skipped) dominates the overlap
loss; revisit if a decode-bound workload regresses.

## Also dropped / regressed on v0.5.15.post1

- **gemma4 / gemma4-31b** — upstream `_attention_kv_config_for_layer` raises
  `NotImplementedError("...does not support sliding-window attention yet...")`,
  and it runs at runner construction, so these raise at boot. Sliding-window
  support is an upstream feature gap (old patch 020 kept `RotatingKVCache`
  native; that path no longer exists). Needs per-layer/window-aware pools.

## Apply / sanity check

```bash
cd components/sglang && git checkout v0.5.15.post1
for p in ../../patches/0[01][0-9]-*.patch; do git apply "$p"; done   # 5 patches, all clean
```

## Historical

- v0.5.12 rebase context: git history + this file's prior revision.
- v0.5.11 rebase strategy: [REBASE-v0.5.11-NOTES.md](REBASE-v0.5.11-NOTES.md).
</content>
