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

## Applied patches (text stack — VALIDATED)

Five patches, applied in numeric order by `scripts/setup.sh`. This is the
production-quality stack for text / agentic-coding workloads.

| # | Patch | Files | Why (rebased against v0.5.15.post1) |
|:-:|-------|-------|-----|
| 002 | mps-backend-defaults | `server_args.py` | **Collapsed.** Only survivors: default `enable_multimodal=False` on MPS (launch presets depend on it) + MLX `--kv-cache-dtype` aliases (`fp8`/`mxfp8`/`turboquant`/`tq`/`4bit`). The aliases are **boot-critical**: the launch default `KV_CACHE=fp8` is not in upstream's `choices`, so without them every preset is rejected by argparse. (torch_native + cuda-graph disable are now upstream.) |
| 003 | mlx-skip-quantization-check | `configs/model_config.py` | Wrap `self._verify_quantization()` in `if not use_mlx():` — MLX checkpoints carry `quantization_config` without `quant_method`, and the upstream raise is still live. |
| 005 | mlx-attn-wrapper-varargs | `kv_cache/attention_wrapper.py` | `__call__(self, x, *args, **kwargs)` + transparent delegate. Devstral/ministral3 (and mlx_vlm attention) pass extra positional/keyword args (`attn_scale`, `position_ids`, …); the fixed 3-arg upstream signature would `TypeError`. Batched decode uses `inner.scale`. |
| 007 | mlx-multimodal-and-mps-shim | `_mps_stub.py`, `managers/mm_utils.py`, `managers/schedule_batch.py` | (a) `.to('cuda')`→`.to('cpu')` redirect; (b) `ShmPointerMMData` slice to logical byte/element count (macOS 16 KB shm page rounding); (c) `Modality.MULTI_IMAGES` enum member. None upstreamed. |
| 014 | mlx-gemma4-image-only-processor | `utils/hf_transformers/processor.py` | Applied **clean** against v0.5.15.post1 (only patch that did). Builds a Gemma 4 image-only processor when the checkpoint ships only `processor_config.json`. |

### Dropped as upstreamed (not re-derived): 006, 009†, 011, 012, 019
† 009's mlx_lm head-count detection is upstream; its NemotronH-specific handling is not (WIP).

## VLM / hybrid path (WIP — deferred)

**The VLM-arch models don't serve on v0.5.15.post1 yet.** `qwen35`, `qwen36`
(the primary agentic recommendation), `devstral`, and `gemma4*` are all
`*ForConditionalGeneration` VLM checkpoints — `mlx_lm` cannot load them, so they
need the `mlx_vlm` loader path that was **entirely removed** upstream (the
v0.5.15 MLX backend is 100% text-only). `nemotron-30b`/`nemotron-omni` are
text-only but hit the rope-less-attention detection gap.

Substantial progress is captured in
[`WIP-phase2-vlm-hybrid-integration.patch`](WIP-phase2-vlm-hybrid-integration.patch)
(kept out of the applied stack — its name doesn't match setup.sh's
`0[01][0-9]-*.patch` glob; 240 lines across 4 files). It re-derives, and gets
`qwen36` from "crash at load" all the way into the serving event loop:

1. **mlx_vlm loader + `_TextOnlyVLMShim`** (`mlx/model_runner.py`) — VLM-detect
   `_load_model`, wrap so text requests route to `language_model` and the native
   cache machinery unwraps via `__getattr__`. Needs `pip install mlx-vlm`
   (`--no-deps`; see setup.sh). *Verified: `mlx_vlm.load` works on the pinned
   `transformers==5.12.1`.*
2. **Attention detection compat** (`kv_cache/attention_contract.py`) — require
   `scale` (the softmax marker that excludes DeltaNet), make `rope` optional so
   `rotary_emb` (mlx_vlm) and position-free (NemotronH) attention are detected.
3. **Serial decode routing** (`mlx/model_runner.py`) — a `_attention_wrapper_
   batchable` flag (inner has `.rope`); route `rotary_emb`/rope-less attention to
   `_decode_with_native_cache` (delegates to the model's own forward). Batched
   decode is an MR>1 throughput optimization only.
4. **Skip general attn-backend init on MLX** (`model_executor/model_runner.py`) —
   `init_attention_backends` no-ops on `use_mlx()`. The MLX tp worker overrides
   `forward_batch_generation`, so the general backends are unused; skipping also
   avoids `GDNAttnBackend.__init__` crashing on the MLX aux-state pool.
5. **`mamba_allocator` alias** (`kv_cache/auxiliary_state.py`) — for the pool
   stats observer.

**Remaining blocker (the reason it's still WIP):** v0.5.15's refactored general
scheduler drives hybrid-SSM (GDN/Mamba) **slot allocation** through a mamba-pool
contract — `mamba_allocator.alloc_group_begin(...)` in the core batch-admission
path (`scheduler.py`), plus `mamba_cache.conv`, etc. — that the MLX
`MlxAuxiliaryStatePool` does not implement. Completing VLM/hybrid support is
**new v0.5.15 integration work**, not a patch rebase: either implement that
allocator contract on the MLX pool (multi-method, with batch-accounting
semantics) or comprehensively disengage the general hybrid machinery on MLX
(the v0.5.12 approach — but `MlxModelRunner._store_auxiliary_state`, called
unconditionally in `prefill_finalize`, now needs the aux pool, so it cascades).

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
