# Gemma 4 31B-it-mxfp4 — Partial Long-Context Bench

Two attempts on 2026-04-19/20 both crashed at the 8K context step with:

```
ValueError: [broadcast_shapes] Shapes (2048,2048) and (1,32,2048,4096) cannot be broadcast.
  in mlx_vlm/models/gemma4/language.py:237 scaled_dot_product_attention
```

Working data points (256, 1K, 4K) in
[long_context_partial_*.txt](long_context_partial_20260420T064531Z.txt).

## What we know

- The error is in `mlx_vlm.models.gemma4.language.Attention.__call__` during
  the **second** chunked-prefill chunk. Mask shape `(2048, 2048)` doesn't
  broadcast against scores `(1, 32, 2048, 4096)` — keys have grown to 4096
  (chunk1 + chunk2) but the mask is still square.
- Patch 014 rewrote `ContiguousKVCache.make_mask` to return an explicit
  `(N, offset+N)` causal mask when `offset > 0`. Did **not** fix Gemma 4.
- Hypothesis: Gemma 4 reads mask from `cache[first_full_cache_idx]`, but
  the cache index used for mask creation may not be the one being updated
  by `update_and_fetch` (Gemma 4 has KV-shared layers via
  `layer_idx_to_cache_idx`). So at chunk 2 start, that specific cache's
  `offset` is still 0, my code returns "causal", which expands to (N, N).

## Workarounds

- For now, run Gemma 4 31B-it at **`--chunked-prefill-size 16384`** so any
  context up to 16K fits in one chunk and never triggers the second-chunk
  mask path. Throughput data ≤ 16K is reliable.
- Long-context (≥32K) on this checkpoint is **blocked** until the
  KV-shared cache mask issue is properly threaded.

## Results so far (turboquant KV cache, MEM_FRAC=0.6)

| Context | TPOT (ms) | Throughput (tok/s) | Prefill (s) |
|--------:|---------:|-------------------:|------------:|
|     256 |    103.8 |                9.6 |         6.6 |
|      1K |    180.1 |                5.6 |        11.5 |
|      4K |    522.4 |                1.9 |        33.4 |
|      8K |     CRASH (mask broadcast)            |             |
