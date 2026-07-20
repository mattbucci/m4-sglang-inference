# beyond-128k: the deep-prefill memory budget

| | |
|---|---|
| **Type** | experiment |
| **Status** | phase 1 complete (160K validated); phase 2 ready |
| **Execution host** | m4-box |
| **Wall clock** | Phase 2 implementation: days (serving hot path). Each deep validation run 15-45 min. |
| **GPU time** | Exclusive serving windows for validation (guard mandatory) |
| **Depends on** | Gate protocol for any patch-stack change |
| **Provides to** | The 256K primary target; decode-TPOT-at-depth shares the same fix surface |

## Objective

128K single-user is validated; 160K+ dies. Characterize and remove the
constraint that blocks 192K/256K.

## Mechanism (measured — receipts in `benchmarks/longctx-bisect/`)

The constraint is the **total deep-prefill memory budget**, not any single
allocation event:

```
weights (wired ~19 GB)
+ attention KV pool + mamba aux pool (auto-sized, turboquant)
+ per-request ContiguousAttentionKVCache — bf16, per attention layer
+ MLX buffer cache (capped, patch 008)
+ per-chunk activation transients
```

- **Chunk size sets the transient floor** (the load-bearing discovery):
  at `chunked 4096`, per-chunk activation transients swing free memory
  1–11 GB and kill every deep run at ~100-113K regardless of cache policy;
  at `chunked 2048` the 128K recipe completes with a worst dip of 10.9 GB.
  Deep runs MUST pin `CHUNKED=2048` (presets default to 4096).
- The per-request contiguous cache starts at 4,096 tokens and doubles with a
  copy (`attention_kv_cache.py:_grow`); its 131072→262144 step transiently
  holds old+new+copy across every attention layer — the attributed killer of
  the 192K run at chunked 2048.
- Per-request pre-sizing (patch 015, landed) is memory-identical to the
  ladder below the 131K crossover; above it, the A/B at 160K / chunked 2048
  decided: the ladder dies at ~156K holding the 262,144-token overshoot,
  exact pre-size completes 157,287 server-verified tokens. **160K is the
  validated ceiling.** 192K dies at ~180K prefilled — steady budget
  exhaustion, no spike.
- The bf16 contiguous cache remains the dominant marginal cost at depth,
  and unlike the pool it is neither quantized nor budget-capped — the
  phase-2 target either way.

## Phase 2 — remove the bf16 monolith (the actionable path)

Two directions, in preference order:

1. **Quantize the per-request contiguous cache** to match the pool
   (turboquant packs ~7× vs fp16). The serial decode path
   (`_decode_with_native_cache`) reads this cache every token, so this also
   attacks decode TPOT at depth (13 s/token at 128K) — one fix, both open
   long-context constraints. Cost: dequant on read in the attention forward;
   MLX quantized-matmul primitives may cover it.
2. **Pool-backed prefill writes** — write each chunk's KV straight into the
   (quantized) pool and attend from pool-gathered KV
   (`PoolBackedAttentionKVCache` exists for the radix prefix path and shows
   the gather pattern). Eliminates the per-request cache entirely; larger
   change to the hot path.

Validation ladder for any phase-2 change: gate protocol → qwen36 tripwire
compare (armed baselines) → 128K control (`CTX=140000 MEM_FRAC=0.5
CHUNKED=2048 --disable-radix-cache turboquant`, must complete
server-verified) → 160K →
192K → 256K, all with `oom_guard.sh` + `mem_profile.sh` and receipts to
`benchmarks/longctx-bisect/`.

## Success criteria

- 192K-labeled run completes with server-verified `usage.prompt_tokens`
  ≥ 0.95 × label, no guard kill.
- No tripwire regression at 1024/8192/32768; 128K control still passes.
- Decode TPOT at 128K materially improved if direction 1 lands (secondary).

## Kill criteria

- Quantized-cache decode quality drifts (probe verdicts flip or greedy
  determinism breaks vs the fp16 cache) → record, fall back to direction 2.
- Any change that fails the 128K control → revert through the gates, as
  with the pre-size attempt.
