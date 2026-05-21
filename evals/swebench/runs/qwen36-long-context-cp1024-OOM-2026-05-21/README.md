# Long-context: chunked-prefill=1024 doesn't help — chunk size isn't the lever

## Hypothesis under test

Yesterday's allocator-log analysis claimed per-chunk attention scales
as `chunk × full_context × heads × bytes`. At chunk=2048 + 65K context
this is ~17 GB per layer. Halving chunk to 1024 should halve it to
~8.5 GB per layer.

## Result: chunk size doesn't matter at all

Both runs (chunk=2048 from `qwen36-long-context-ctx65k-OOM-2026-05-21/`
and today's chunk=1024) died at almost identical points in the prefill:

| Config | OOM-kill point | Server status |
|---|---|---|
| chunk=2048, ctx=65536, mf=0.4 | ~40K / 60K processed | DOWN |
| chunk=1024, ctx=65536, mf=0.4 | ~43K / 60K processed | DOWN |

Memory consumption is roughly **0.15-0.2 MB per processed prefill
token**, independent of chunk size:

```
free pre-request:   7 GB
free at ~43K processed:  drops below 8 GB → OOM guard fires
delta: ~1 GB lost per 5K tokens prefilled  ≈ 0.2 MB/token
```

This is **NOT consistent** with the per-chunk attention theory. If
attention scores were the dominant cost, halving chunk should have
roughly doubled the prefill we could complete before OOM.

## Revised theory

The KV cache itself is small (bytes_per_slot=23040 ⇒ 23 KB/token ⇒
~1.4 GB at 60K), well within `kv_budget=1.73 GB`. But each prefilled
token apparently triggers ~0.15-0.2 MB of *additional* transient
memory beyond the KV slot. Possible sources:

1. **Per-token unified-memory page touches**: MLX allocations on Apple
   Silicon get pulled in lazily. Each token's KV write touches new
   pages that get marked active and contribute to OS memory pressure
   even after the "wired memory" accounting says we're fine.
2. **Python-side intermediate tensors** (per-token logit slices, mask
   construction, position-id buffers) that accumulate across chunks
   without GC.
3. **The MoE expert dispatch buffers** allocated per token; qwen36 has
   128 experts with top-8 routing. Per-token routing weights and
   gather indices accumulate.

Whatever the actual mechanism, the empirical observation is clear:
**memory consumption during prefill is linear in tokens-processed,
not chunk-size**, at ~0.15-0.2 MB per token.

## Updated long-context envelope

Given the linear cost:

- Boot reserves ~57 GB → 7 GB free pre-request
- Each prefilled token costs ~0.2 MB
- OOM threshold at ~7 GB free (with current OS overhead)
- Practical ceiling: **roughly 35K tokens** of input prefill

Matches yesterday's empirical finding (32K worked at mf=0.4 with 88
MB free after — right at the edge).

## What this means

1. **Stop tuning chunked-prefill-size for long-context memory.** It
   doesn't move the needle. The May-11 `chunked-prefill 2048` is fine
   for what it controls; it just can't unlock 64K.

2. **The actual variance source from earlier loop iterations is
   probably NOT chunked-prefill scheduling either.** That hypothesis
   was based on the assumption that chunk-size matters for prefill
   behavior; turns out it doesn't (at least for memory).

3. **The real ceiling needs flash-attention-style reduced-activation
   attention in MLX**, OR a memory-aware decoder that releases
   per-token transient memory more aggressively. Neither is a config
   knob; both are upstream MLX/SGLang work.

4. **CLAUDE.md "256K primary target" needs to be re-scoped.** The M4
   today can prefill ~32-35K tokens of context for a 35B-MoE model.
   That's the actual hardware ceiling on the current toolchain.

## What still works

- SWE-bench agentic flows (8-25K typical per turn): well under
  ceiling ✓
- Decode at any tested context: ~60 tok/s sustained ✓
- The 5/13 = 38.5% M4-scorable RESOLVED rate: unaffected ✓

## Files

- This README — analysis
- Server log details captured in launch.log fragments (gitignored
  as *.log)
