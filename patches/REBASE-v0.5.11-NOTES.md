# Rebase plan: v0.5.11

Current pin: `1f8df9705` (post-v0.5.10 main, 2026-04-12). Target: `v0.5.11` (612785ffd).
659 commits between. `1f8df97` is an ancestor of `v0.5.11` — fast-forward path is valid.

## Strategy: trim, don't carry

v0.5.11 upstreamed our biggest patch (001, the `kv_cache/` subpackage). The right move is to **drop everything that's now upstream and keep only what's load-bearing for the model set we actually run** — namely, the mlx-community 4-bit MLX checkpoints that mirror what the sister teams (3090 / R9700) run on their AWQ/GPTQ stacks:

| mlx-community model | Sister-team analog | Patches needed |
|---------------------|--------------------|----------------|
| Devstral-24B-4bit | 3090 / R9700 Devstral-24B AWQ | 002 (MPS defaults), 005 (varargs wrapper), 007 (VLM fallback), 010–012 (image handoff) |
| Coder-30B-A3B-4bit | 3090 / R9700 Coder-30B AWQ | 002, 003 (MLX quant check) |
| Qwen3-30B-MoE-4bit | 3090 Qwen3-30B REAM AWQ | 002, 003 |
| Qwen3.5-27B-4bit | 3090 / R9700 Qwen3.5-27B AWQ | 002, 003, 004 (lifecycle + hybrid), 006 (OffsetCache subscript), 008 (serial DeltaNet), 013 (`language_model.make_cache`) |
| Qwen3.6-35B-A3B-4bit | 3090 / R9700 Qwen3.6-35B GPTQ | same as 3.5 + VLM handoff |
| Gemma 4 26B-A4B-it-4bit | 3090 / R9700 Gemma 4 26B MoE AWQ | 002, 003, 014 (explicit mask), 015 (RotatingKVCache native) |
| Gemma 4 31B-it-mxfp4 | 3090 / R9700 Gemma 4 31B Dense AWQ | 002, 003, 014, 015 |
| Qwen3-32B-4bit | (no sister analog) | 002, 003 |
| Coder-Next-80B-4bit | R9700 Coder-Next 80B AWQ | currently OOMs on 64 GB M4 — fix is the mlx_vlm import-bloat removal, not a new patch |

Nothing in our patch set exists to support models the sister teams aren't running. All of it serves the shared mlx-community set.

## What v0.5.11 already ships

Upstream finally has a real MLX backend with the `kv_cache/` subpackage that patch 001 pioneered:

```
python/sglang/srt/hardware_backend/mlx/
  model_runner.py        703 lines (up from ~500-ish on 1f8df97)
  scheduler_mixin.py     234 lines  *** new ***
  tp_worker.py           485 lines
  model_runner_stub.py   175 lines
  kv_cache/
    __init__.py
    attention_wrapper.py  150 lines
    contiguous_cache.py   205 lines  — has OffsetCache class
    kv_pool.py             80 lines  — MlxKVPool
    model_patching.py      46 lines
```

Patch-by-patch status against v0.5.11:

| # | Patch | Status against v0.5.11 | Action |
|:-:|-------|------------------------|--------|
| 001 | mlx-radix-cache | **Upstreamed.** `ContiguousKVCache`, `MlxKVPool`, `OffsetCache`, `attention_wrapper.py`, `model_patching.py` all live under `kv_cache/` in v0.5.11. | **DROP entirely.** Verify our scatter-write semantics aren't subtly different from upstream's `update_and_fetch`; if so, fold the delta into 014. |
| 002 | mps-backend-defaults | Still needed — server_args.py defaults aren't upstream. | Rebase hunks. |
| 003 | mlx-skip-quantization-check | Applies cleanly to v0.5.11. | Keep as-is. |
| 004 | mlx-lifecycle-and-hybrid-fixes | Still needed — no DeltaNet/Mamba hybrid handling upstream. | Re-thread into v0.5.11's `_acquire_cache` (likely in `scheduler_mixin.py`). |
| 005 | mlx-attn-wrapper-varargs | Still needed — Devstral varargs forward pass not upstream. | Rebase hunks. |
| 006 | mlx-offsetcache-subscript | Partial — upstream has `OffsetCache` class but no `__getitem__` shim. | One-line subscript shim. |
| 007 | mlx-vlm-fallback | Still needed — no `mlx_vlm` fallback path upstream. | Rebase hunks. |

## In-tree patches 008–015 (post-1f8df97 mods, never serialized as `.patch` files)

These mods exist only in the working tree on top of the 7 numbered patches. Captured pre-rebase at `/tmp/m4-intree-pre-rebase.patch` (1657 lines).

- 008: serial-decode for DeltaNet hybrids (MAX_RUNNING=1)
- 010/011/012: VLM image preprocess + tensor handoff
- 013: route hybrid cache via `model.language_model.make_cache()` (fixes DeltaNet quality from MMLU 16.7% → 93.0%)
- 014: `ContiguousKVCache.make_mask` returns explicit `(N, offset+N)` array when `offset>0` (unblocks chunked prefill at large context)
- 015a: keep `RotatingKVCache` native (Gemma 4 sliding layers)
- 015b: full cache reset on pool reuse (offset/_idx/keys/values)

## Recommended rebase strategy

1. **Branch off from current state.** Create `rebase-v0.5.11` branch in `components/sglang`:
   ```bash
   cd components/sglang && git checkout -b rebase-v0.5.11
   git checkout v0.5.11 -- .
   ```
2. **Re-derive each patch surgically.** Don't blindly apply old patches — read the v0.5.11 file, decide what's needed.
3. **Start with the easy/upstreamed ones.** 001 (verify upstreaming complete; port the bench_one_batch tweak only), 003 (clean), 006 (one-line subscript shim).
4. **Then the structural ones.** 002 (defaults), 005 (varargs), 007 (VLM fallback).
5. **Then the hybrid cache work.** 004 + 013 + 014 + 015 form a logical block — these are the biggest risk because v0.5.11's scheduler_mixin.py may have moved cache-acquisition logic.
6. **Validate with `scripts/eval/bench_smoke.sh`** after each major patch — coder-30b + devstral are the cheapest models that exercise dense / MoE paths.
7. **Regenerate patch files** with `git format-patch` style diffs once green.

## Risks to watch

- `scheduler_mixin.py` is new — may have absorbed our `_acquire_cache` lifecycle code in a way that changes hook points for patch 004.
- v0.5.11's `OffsetCache` definition may have already grown a subscript path that we don't realize duplicates our patch 006. Check before re-adding.
- `model_runner.py` upstream is 703 lines vs our patched ~1000 lines. ~300 lines of our patches (008/010-015) may need re-threading into a now-different structure.

## Smoke test plan

After rebase compiles and starts:
```bash
bash scripts/eval/bench_smoke.sh coder-30b   # dense MoE path
bash scripts/eval/bench_smoke.sh devstral    # dense VLM path
bash scripts/eval/bench_smoke.sh qwen35      # DeltaNet hybrid path (was MMLU 16.7% before patch 013)
```

Don't ship the rebase until all three pass.
