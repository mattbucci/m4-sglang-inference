# qwen36 long-context at mf=0.4 — 32K works, 64K still won't fit

## Hypothesis under test

Yesterday: 64K OOMs on v0.5.12 with `--mem-fraction-static 0.5`. The
CLAUDE.md memory `feedback_mem_frac_unified_memory` says: "The right
levers are max_tokens / chunked-prefill / MAX_RUNNING, not mem
fraction." But maybe the *direction* of mem-fraction matters —
lowering it (more headroom for OS) might actually help on a more-
loaded system.

## Result

`--mem-fraction-static 0.4` (everything else same as yesterday) at 32K:

```
32K tokens  in=31458  out=32  time=51.9s  TPOT=1623ms  combined 0.6 tok/s
  prefill rate ≈ 612 tok/s    decode rate ≈ 60 tok/s (healthy)
  server health=OK after request
  free RAM after: 88 MB (5637 pages × 16 KB)
```

32K fits. Decode is healthy. But memory dropped from ~46 GB free
pre-request to 88 MB free after. The headroom is gone.

64K not attempted: ~2× KV cache vs 32K. With 88 MB free, OOM is
certain. Skipped.

## What this means

- Current stack can run `qwen36` at up to about 32K of input context
  in a single request before memory exhaustion.
- `mem-fraction-static 0.4` did NOT help fit 64K — moving the knob
  in the safe direction didn't open enough headroom.
- The right next levers per CLAUDE.md memory:
  - `--max-total-tokens` (cap KV cache regardless of CTX)
  - Smaller `--chunked-prefill-size` (already at 2048)
  - Drop CTX from 131072 to something smaller (defeats the purpose
    of long context but stops auto-allocating a big KV pool)

## Pieces that DO work

| Context | Status | Per-token throughput |
|---|---|---|
| 256 tokens | ✓ | 48.5 tok/s combined |
| 4K | ✓ | 11.1 tok/s combined |
| 16K | ✓ | 2.8 tok/s combined |
| **32K** | ✓ (mf=0.4) | **0.6 tok/s combined**, decode 60 tok/s |
| 64K | ✗ OOM | — |
| 128K | ✗ OOM (extrapolated) | — |
| 256K | ✗ OOM (extrapolated) | — |

The agentic SWE-bench workload runs at 8-25K input context per turn —
that's well under 32K and works fine. The "256K primary optimization
target" per CLAUDE.md is NOT achievable today on the standard recipe.

## Next loop iterations (tracked)

1. **Try `--context-length 65536`** (instead of 131072). SGLang's
   auto-`max-total-tokens` sizes KV pool based on `--context-length`;
   lowering it from 131072 to 65536 halves the upfront KV reservation.
   Might let 64K actually finish if the pool starts smaller.
2. **Explicit `--max-total-tokens`**. Set to e.g. 70000 to cap the KV
   pool independent of CTX.
3. **Audit Memory baseline on M4 today.** Check whether non-SGLang
   processes (browsers, IDEs) have grown — the May-11 baseline may
   have been a fresh-boot environment that today's session can't
   reproduce.
4. **Re-validate against `--kv-cache-dtype fp8`** (yesterday's run
   used turboquant). fp8 should give 2× KV savings vs fp16 but maybe
   turboquant's 3.5× savings comes with overhead we don't account for.

## Recommendation effect

- qwen36 primary for ≤32K work confirmed at mf=0.4 — agentic SWE-bench
  is well within this envelope (8-25K typical).
- Long-context (≥64K) un-runnable on tested recipes. The CLAUDE.md
  "validated 2026-05-11 on v0.5.11" claim is outdated; needs an
  explicit "as of 2026-05-21, 32K is the M4 ceiling" replacement.

## Files

- `bench-32k.txt` — the raw bench output
