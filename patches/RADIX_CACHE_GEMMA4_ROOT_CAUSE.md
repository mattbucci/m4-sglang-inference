# MlxKVPool + Gemma 4 heterogeneous attention — root cause analysis

**Date:** 2026-05-13
**Stack:** SGLang v0.5.11 + M4 patches 002–012
**Symptom:** `gemma4` / `gemma4-31b` presets crash on first prefill with
`ValueError: [broadcast_shapes] Shapes (2,128) and (1,8,64) cannot be broadcast.`
at `kv_pool.py:87` (`set_kv`). The crash only happens when radix cache is enabled
(the default). `EXTRA_ARGS="--disable-radix-cache"` makes the model serve cleanly.

This document explains exactly why the crash happens and identifies the proper fix.
We are not landing the fix today — the workaround (`--disable-radix-cache` on every
gemma4 preset) is sufficient for evals — but the root cause is structural and
worth documenting before someone bisects 010/011/012 hoping to revert.

## Bug class

`MlxKVPool` assumes **homogeneous per-layer attention shapes**. Gemma 4 violates
this assumption with two interleaved attention types.

## Gemma 4 26B layer pattern

`mlx-community/gemma-4-26b-a4b-it-4bit` `text_config`:

| Field | Value |
|-------|------:|
| `num_hidden_layers` | 30 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `head_dim` | 256 |
| `global_head_dim` | **512** |
| `num_global_key_value_heads` | **2** |
| `attention_k_eq_v` | true |
| `sliding_window` | 1024 |
| `layer_types` | `[sliding, sliding, sliding, sliding, sliding, full, ...]` (every 6th is `full_attention`) |
| Full layer indices | `[5, 11, 17, 23, 29]` (5 of 30) |
| Sliding layer indices | the other 25 |

In `mlx_vlm/models/gemma4/language.py:Attention.__init__`:

```python
self.head_dim = (
    config.global_head_dim
    if self.layer_type == "full_attention" and ...
    else config.head_dim
)
self.use_k_eq_v = (
    getattr(config, "attention_k_eq_v", False) and not self.is_sliding
)
if self.use_k_eq_v and config.num_global_key_value_heads is not None:
    self.n_kv_heads = config.num_global_key_value_heads   # full attention
else:
    self.n_kv_heads = config.num_key_value_heads          # sliding attention
```

So per-layer attention KV shapes are:

| Layer type | n_kv_heads | head_dim | fp8-packed `(n_kv_heads, head_dim/4)` |
|------------|:----------:|:--------:|:--------------------------------------:|
| sliding (25 layers) | 8 | 256 | (8, 64) |
| full (5 layers) | 2 | 512 | **(2, 128)** |

Gemma 4 31B (`mlx-community/gemma-4-31b-it-mxfp4`) follows the same pattern:
60 layers, 50 sliding (16 heads × 256 dim), 10 full (4 heads × 512 dim). Both
sizes are affected.

## How MlxKVPool gets sized

`MlxKVPool` allocates one buffer per layer of fixed shape
`(pool_size, n_kv_heads, head_dim)` (or its fp8/turboquant packed variant).
**One** `(n_kv_heads, head_dim)` for all layers.

Sizing comes from `_get_attn_config` in `mlx/model_runner.py:595`:

```python
layer_list, attn_attr = find_attention_layers(self.model)
for layer in layer_list:
    candidate = getattr(layer, attn_attr, None)
    if candidate is None:
        continue
    inner = candidate._inner if isinstance(candidate, MLXAttentionWrapper) else candidate
    if attn_attr == "mixer" and not all(hasattr(inner, n) for n in ("q_proj","k_proj","v_proj","o_proj")):
        continue
    sample_attn = candidate
    break  # ← samples the FIRST attention layer it finds
n_kv_heads = getattr(sample_attn, "n_kv_heads", None) or getattr(sample_attn, "num_key_value_heads", None)
head_dim = sample_attn.head_dim
```

For Gemma 4 26B, layer 0 is `sliding_attention` → reports `(8, 256)`. Pool gets
sized for the sliding shape. fp8 packed: `(pool_size, 8, 64)`. **Wrong** — see
next section.

The boot log confirms:
```
MlxKVPool: 184283 slots × 30 layers × 8 heads × 256 dim, dtype=mlx.core.uint32, mode=fp8
```

## Why the pool is sized for the wrong layer type

`_acquire_cache` in `model_runner.py:233` decides which layers actually write
to the pool:

```python
native_cache = native_maker()  # model.language_model.make_cache()
for nc in native_cache:
    if type(nc).__name__ == "KVCache":
        cache.append(ContiguousKVCache(...))   # → writes to pool
    else:
        cache.append(nc)                       # RotatingKVCache, ArraysCache → skipped
```

For Gemma 4:

| Layer type | Native cache | Replaced with | Writes to pool? |
|------------|--------------|---------------|:----------------:|
| sliding | `RotatingKVCache` (ring buffer) | kept native (patch 015b reason) | **no** |
| full | `KVCache` (linear) | our `ContiguousKVCache` | **yes** |

So the pool only ever receives writes from the **full-attention layers**
(`(n_kv_heads=2, head_dim=512)` → fp8 packed `(2, 128)` per slot). But the pool
was sized for **sliding** shape `(8, 64)`. First full-attention prefill fires:

```
ValueError: [broadcast_shapes] Shapes (2,128) and (1,8,64) cannot be broadcast.
```

— exactly what we see at `kv_pool.py:87`:

```python
self._k_buffers[layer_id][key][slots] = kq[key]
#  └─ pool slot: (1, 8, 64)         ─┘    └─ new KV: (2, 128) ─┘
```

Patch 012's `isinstance(layer_cache, ContiguousKVCache)` filter correctly skips
the sliding layers' `RotatingKVCache` (no pool write attempted). But it does
nothing about the shape mismatch on the full-attention layers' `ContiguousKVCache`
(which the filter intentionally keeps).

## Why `--disable-radix-cache` is a clean workaround

`init_kv_pool` in `model_runner.py:690`:

```python
def init_kv_pool(self, req_to_token_pool):
    self._req_to_token_pool = req_to_token_pool
    if self.disable_radix_cache:
        return   # ← skip MlxKVPool entirely
    ...
```

And `prefill` in `model_runner.py:854`:

```python
if self.disable_radix_cache:
    cache = self._acquire_cache()
    input_ids = mx.array([new_token_ids], dtype=mx.int32)
    model_output = self.model(input_ids, cache=cache)
    ...
    return MlxPendingPrefill(...)
# else: PoolBackedCache + _sync_new_kv_to_pool path that hits the bug
```

With radix off:
- No `MlxKVPool` is constructed.
- Prefill builds a fresh per-request cache via `_acquire_cache()` (which has the
  correct per-layer shape because it gets it from `language_model.make_cache()`).
- `_sync_new_kv_to_pool` is never called.

This is the same reason the v0.5.11 capability gate at commit `ebe23bb` (May 11)
reported `gemma4` 2/2 PASS — `run_all_evals.sh` always exports
`EXTRA_ARGS="--disable-radix-cache"`. Run gemma4 from a plain `scripts/launch.sh`
invocation and the bug fires immediately on the first prefill.

## Memory pressure is a separate failure mode

First reproduction attempt at 02:53 today: `--disable-radix-cache` set, but the
server still died at scheduler init with BrokenPipeError. Root cause: the prior
nemotron-30b server had left enough MLX-pinned memory that gemma4 boot tripped
on `sys_available=11.02 GB` and the scheduler subprocess SIGQUITed before
returning init info (the parent server died first). Clean retry at 02:56 after
`pkill` + 4s wait: gemma4 booted to `/health=200` in 5 seconds.

This is the same class as the "macOS unified memory pinned across consecutive
server boots" pattern documented elsewhere in this repo. Not a new bug; just
relevant context for anyone debugging the same trace.

## Possible fixes (none landed)

### Fix A — sample a full-attention layer for pool sizing (minimal change)

Modify `_get_attn_config` to prefer a layer whose `is_sliding` attribute is
false (or absent — fall through for non-Gemma models). For Gemma 4 26B this
samples layer 5 instead of layer 0:

```python
for layer in layer_list:
    candidate = getattr(layer, attn_attr, None)
    if candidate is None:
        continue
    inner = candidate._inner if isinstance(candidate, MLXAttentionWrapper) else candidate
    # Prefer full-attention layers — those are the ones that write to the pool.
    # Sliding layers use RotatingKVCache and are kept native.
    if getattr(inner, "is_sliding", False):
        sliding_fallback = sliding_fallback or candidate
        continue
    if attn_attr == "mixer" and not all(hasattr(inner, n) for n in ("q_proj","k_proj","v_proj","o_proj")):
        continue
    sample_attn = candidate
    break
sample_attn = sample_attn or sliding_fallback
```

Pool gets sized `(2, 512)` → packed `(2, 128)`. Full-attention prefills now
broadcast cleanly into pool slot shape `(1, 2, 128)`. Sliding-layer pool entries
remain zero-initialised (25 layers × pool_size × 2 heads × 128 packed = wasted
memory, but the model never reads from them since RotatingKVCache stays native).

**Catch:** if some future model has heterogeneous full-attention layers (multiple
shapes among layers that all use `ContiguousKVCache`), Option A still fails.
This is the right fix for Gemma 4 specifically, not a universal fix.

### Fix B — per-layer pool buffer shapes (structural)

`MlxKVPool` becomes a list of per-layer buffers with their own
`(n_kv_heads, head_dim)`. `_get_attn_config` returns a list (one entry per
layer that writes to the pool) instead of a single tuple. `init_kv_pool`
constructs each layer's buffer at its actual shape. `set_kv(layer_id, ...)`
uses the matching shape per layer.

Memory waste of Option A goes away. Future heterogeneous-attention models work
without further changes. Cost: ~80–150 lines of refactor across `kv_pool.py`,
`kv_quant.py`, `model_runner.py`. Worth doing if a second model with this
pattern lands.

### Fix C — shape-aware skip in `_sync_new_kv_to_pool` (defensive)

Compare the incoming KV's `(n_kv_heads, head_dim)` against the pool's per-layer
buffer shape. Skip the write on mismatch. Same correctness as A (those layers
lose radix functionality) but defends against future regressions in fix-A's
heuristic. Could be combined with A as a belt-and-suspenders pair.

## Status

- Workaround applied 2026-05-13: `--disable-radix-cache` baked into `gemma4` /
  `gemma4-31b` presets (next commit).
- Quality evals on Gemma 4 already use the workaround (via `run_all_evals.sh`),
  so the published MMLU 85 / HumanEval 60 (chat-mode) numbers are valid.
- Single-user serving without radix cache loses agentic prefix-reuse on
  multi-turn flows; not a problem for benches but a meaningful gap for
  production use. Fix A is the minimum to restore radix support on Gemma 4.

## What `--disable-radix-cache` actually costs

The radix cache (patch 001, upstreamed in v0.5.11) reuses KV state across
identical prefixes. For agentic flows it's the difference between re-prefilling
a 256K system prompt every turn (~20 min on M4) vs a sub-second cache hit. With
radix off, every request is a full prefill regardless of prefix overlap.

For evals (one-shot prompts) this is invisible: each MMLU/HE prompt is unique.
For agentic / multi-turn workloads on Gemma 4, radix off makes the model
unusable at long context. That's the practical case for landing Fix A.
