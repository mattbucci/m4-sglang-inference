# Long-context prefill memory growth — attribution

## Verdict

**Root cause: unbounded MLX buffer-cache accumulation across chunked-prefill
steps.** Freed per-chunk transient buffers (MoE dispatch, attention scratch)
are retained in MLX's buffer cache; chunk shapes shift as the sequence
grows, so retained buffers are not reused and long prefills consume unified
memory at ~0.63 MB/token until jetsam. **Fix: cap the cache**
(`mx.set_cache_limit`, wired into `MlxModelRunner.__init__`, default 4 GB,
env-tunable via `SGLANG_MLX_CACHE_LIMIT_GB`). With the cap, the historical
128K recipe fits again. Not a lib regression, not config-tunable — a
one-line runtime policy in the MLX backend.

## Measurements (qwen36 = Qwen3.6-35B-A3B-4bit, turboquant KV, chunked 2048, oom_guard armed)

| Run | mlx | Cache cap | Config | 32K growth | Outcome |
|---|---|---|---|---|---|
| phase0 | 0.32.0 | none | CTX 140K, mf 0.5 | 0.62 MB/token | guard-killed at ~95% of 32K prefill (free 7.4 GB) |
| phase0-32k-recipe | 0.32.0 | none | CTX 32K, mf 0.4 | 0.63 MB/token | guard-killed at ~95% of 32K prefill |
| phase0-mlx0312 | 0.31.2 | none | CTX 32K, mf 0.4 | 0.63 MB/token | guard-killed — identical collapse |
| (mlx 0.31.1) | 0.31.1 | — | — | — | unbootable: mlx-lm 0.31.3 requires mlx>=0.31.2 |
| phase0-cachelimit | 0.32.0 | 4 GB | CTX 32K, mf 0.4 | flat (no guard warns) | **32K completes** (in=31,458; 52.7s) |
| phase1-deep 64K | 0.32.0 | 4 GB | CTX 140K, mf 0.5 | flat | **64K completes** (in=62,916; 139.3s) |
| phase1-deep 128K | 0.32.0 | 4 GB | CTX 140K, mf 0.5 | flat plateau (~11.5 GB free through prefill) | **128K completes** (in=125,830; prefill 415.8s ≈ 6.9 min; decode 0.1 tok/s — matches the historical receipt) |
| phase2-256k 192K | 0.32.0 | 4 GB | CTX 262K, mf 0.5, chunked 2048 | flat until ~133K tokens | guard-killed at ~133K processed — the contiguous attention cache's capacity-doubling boundary (131K → 262K realloc spike), not steady growth |
| presize-160k | 0.32.0 | 4 GB | CTX 210K, mf 0.5, **chunked 4096**, patch-015 pre-size | free oscillating 1–11 GB per chunk | guard-killed at ~100K — chunked-4096 transient floor (see below), NOT the pre-size design |
| presize-160k-ctx175k | 0.32.0 | 2 GB | CTX 175K, mf 0.5, **chunked 4096**, patch-015 pre-size | same oscillation | guard-killed at ~113K |
| presize-128k-control | 0.32.0 | 4 GB | CTX 140K, mf 0.5, **chunked 4096**, patch-015 pre-size | — | guard-killed at ~103.5K |
| postrevert-128k | 0.32.0 | 4 GB | CTX 140K, mf 0.5, **chunked 4096**, 015 reverted | — | guard-killed at the IDENTICAL ~103.5K (pending-token 27526 in both logs) — pre-size ≡ ladder below the 131K crossover; the 4096 chunk size is the killer |
| postrevert-128k-chunked2048 | 0.32.0 | 4 GB | CTX 140K, mf 0.5, chunked 2048 | worst WARN 10.86 GB | **128K completes** (in=125,830; prefill 424.9s — matches phase1-deep 415.8s) |
| ladder-160k-chunked2048 | 0.32.0 | 4 GB | CTX 175K, mf 0.5, chunked 2048, doubling ladder | survives the 131K grow | guard-killed at ~156K — post-doubling the ladder holds 262,144-token bf16 buffers per attention layer (the power-of-two overshoot) and the budget runs out |
| presize-160k-chunked2048 | 0.32.0 | 4 GB | CTX 175K, mf 0.5, chunked 2048, patch-015 pre-size | worst WARN ~10.3 GB | **160K completes** (in=157,287; prefill 606.9s; decode 0.1 tok/s) — exact-size buffers (163,872), no spike, no overshoot |
| presize-192k-chunked2048 | 0.32.0 | 4 GB | CTX 210K, mf 0.5, chunked 2048, patch-015 pre-size | — | guard-killed at ~180K — the bf16 contiguous cache (196,640-token buffers) + 210K pool exceed the budget; no single spike, steady exhaustion |

Growth computed from mem_profile.sh (free+inactive delta over the prefill
window ÷ server-verified prompt tokens); per-run receipts in the sibling
directories (mem_profile.csv, bench-results.txt, server-log tails).

## Chunk-size transient floor

The deep-prefill envelope is **chunk-size sensitive**: at `chunked 4096` the
per-chunk activation transients (MoE dispatch + attention scratch at 4096
tokens) swing free memory 1–11 GB per chunk and cross the guard line at
~100-113K depth regardless of pre-sizing, CTX, or the MLX cache cap; at
`chunked 2048` the same 128K request completes with a worst-case dip of
10.9 GB. The launch presets default to `CHUNKED=4096` — every deep run must
pin `CHUNKED=2048` (now part of the documented recipe). vm_stat confirms
full memory recovery between runs (no leak, no box-state degradation).
Patch-015-style pre-sizing is unproven either way below the 131K crossover
(identical death depths); the ladder-vs-presize A/B at chunked 2048 / 160K
is the deciding experiment.

## Attribution chain

- The collapse rate is invariant to mlx-core version (0.32.0 ≡ 0.31.2 on
  identical code) and to pool config (mf 0.4/CTX 32K ≡ mf 0.5/CTX 140K) —
  ruling out the "floated mlx libs" hypothesis for the current stack's
  severity and the "pool sizing" configuration hypothesis.
- The severity is code-path-dependent: ~0.2 MB/token on the earlier stack
  pin (its receipts) vs ~0.63 MB/token on the current pin — consistent with
  the current runner producing more distinct transient buffer shapes per
  chunk (MoE + hybrid layer mix via the VLM path), all retained by the
  unbounded cache.
- Capping the cache removes the growth entirely at no measured prefill-speed
  cost (32K prefill ~53s capped vs guard-death uncapped; 64K scales
  linearly).
- The historical era escaped because its code produced a retention rate low
  enough (~0 observable at 128K) that the cache never outran RAM. The
  earlier ~0.2 MB/token era hit the wall at ~32K; the current uncapped rate
  moved the wall to ~30K. The underlying mechanism (unbounded cache +
  shape-shifting chunks) is the same across all three; only the per-chunk
  retention volume differs.

## Consequences

- The bisect arms (old-tree worktree builds) are unnecessary — closed
  without arm construction.
- `--chunked-prefill-size` insensitivity is explained: chunk size changes
  the shapes, not the retention policy.
- The ceiling formula (max_prefill ≈ free_GB × 5120) described the uncapped
  cache's burn rate, not a structural limit; superseded by the cap.
- Decode TPOT at depth (1.6 s at 32K, 4.4 s at 64K, 13 s at 128K) is now a
  binding long-context constraint — a separate optimization target
  (attention read bandwidth), not a memory bug.
- **Certified ceiling: 128K.** The next boundary is mechanistic: at 131K
  tokens the per-layer ContiguousAttentionKVCache doubles capacity
  (131K→262K), and the realloc spike across attention layers blows the
  remaining budget. Going past 128K needs incremental (non-doubling) cache
  growth or pool-backed prefill writes — queued as its own item.
