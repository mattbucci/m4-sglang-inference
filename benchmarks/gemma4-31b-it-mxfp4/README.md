# Gemma 4 31B-it-mxfp4 — Partial Long-Context Bench

Two attempts on 2026-04-19/20 both crashed at the 8K context step with:

```
ValueError: [broadcast_shapes] Shapes (2048,2048) and (1,32,2048,4096) cannot be broadcast.
  in mlx_vlm/models/gemma4/language.py:237 scaled_dot_product_attention
```

Working data points (256, 1K, 4K) in
[long_context_partial_*.txt](long_context_partial_20260420T064531Z.txt).

## What we know (updated 2026-04-20)

- The error is in `mlx_vlm.models.gemma4.language.Attention.__call__` during
  the **second** chunked-prefill chunk. Mask shape `(2048, 2048)` doesn't
  broadcast against scores `(1, 32, 2048, 4096)` — keys have grown to 4096
  (chunk1 + chunk2) but the mask is still square.
- Patch 014 rewrote `ContiguousKVCache.make_mask` to return an explicit
  `(N, offset+N)` causal mask when `offset > 0`. Did **not** fix Gemma 4.
- Initial hypothesis (KV-shared layers) was **wrong**: Gemma 4 31B-it has
  `num_kv_shared_layers=0` — all 60 layers are concrete and own their
  own caches.

## Real root cause — patch 015

Gemma 4 `make_cache()` returns a **mix** of cache types:
- Full-attention layers (10 of 60) → `KVCache` (standard)
- Sliding-attention layers (50 of 60, `sliding_window=1024`) →
  `RotatingKVCache` (ring buffer, caps at window_size)

Our `_acquire_cache` was replacing **both** `KVCache` AND `RotatingKVCache`
with `ContiguousKVCache`. That broke the sliding semantics: a
`ContiguousKVCache` stores all appended keys (no rotation), so after
chunk 1 the sliding layer's cache holds 2048 keys instead of the
expected 1024. At chunk 2, mask creation assumes rotating semantics
(capping offset at `max_size-1=1023`) and returns a mask shape that
doesn't match the actual key count.

Patch 015 fix: only replace `KVCache` with our quantized
`ContiguousKVCache`. Keep `RotatingKVCache` native so sliding-attention
layers stay rotating. Sliding-layer KV is tiny (window_size=1024) so
losing fp8/turboquant on those layers is irrelevant to the memory budget.

## Workarounds

- For now, run Gemma 4 31B-it at **`--chunked-prefill-size 16384`** so any
  context up to 16K fits in one chunk and never triggers the second-chunk
  mask path. Throughput data ≤ 16K is reliable.
- Long-context (≥32K) on this checkpoint is **blocked** until the
  KV-shared cache mask issue is properly threaded.

## Results so far (turboquant KV cache, MEM_FRAC=0.6)

### Post-patch-015 (2026-04-20)

Chunked prefill now works through 8K after patches 015a (keep
RotatingKVCache native) and 015b (reset `_idx` + drop `keys`/`values`
on cache pool reuse). 16K crashed — but OOM guard fired cleanly
(free dropped to 7.88 GB, below 8 GB threshold; no system freeze).

| Context req | Actual tokens | TPOT (ms) | Throughput (tok/s) | Prefill (s) |
|-------------|--------------:|---------:|-------------------:|------------:|
|         256 |           247 |    116.2 |                8.6 |         7.4 |
|          1K |           985 |    235.9 |                4.2 |        15.1 |
|          4K |          3935 |    751.6 |                1.3 |        48.1 |
|          8K |          7866 |   1477.9 |                0.7 |        94.6 |
|         16K |               | OOM GUARD (free < 8 GB kill threshold) |  |  |

### Pre-patch-015 (archived)

Chunked prefill crashed at chunk 2 of the 8K test:

| Context | TPOT (ms) | Throughput (tok/s) | Prefill (s) |
|--------:|---------:|-------------------:|------------:|
|     256 |    103.8 |                9.6 |         6.6 |
|      1K |    180.1 |                5.6 |        11.5 |
|      4K |    522.4 |                1.9 |        33.4 |
|      8K |     CRASH (mask broadcast)            |             |
