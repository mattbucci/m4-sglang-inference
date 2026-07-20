# beyond-128k: pre-size the contiguous attention cache to reach 192K/256K

| | |
|---|---|
| **Type** | experiment |
| **Status** | ready |
| **Execution host** | m4-box |
| **Wall clock** | Implementation ~2-3h; each deep validation run 15-45 min (192K prefill ~20 min at ~300 tok/s chunked) |
| **GPU time** | Exclusive serving windows for the validation runs (guard mandatory) |
| **Depends on** | Patch stack gates (land any change through test_patch_gates.sh + import_smoke) |
| **Provides to** | The 256K primary target; decode-TPOT-at-depth work (phase 2) |

## Objective

128K is validated; 192K+ dies at the contiguous-attention cache's
capacity-doubling boundary. Remove the boundary by right-sizing the per-request
cache when the request's total length is already known, then validate 192K and
push toward 256K.

## Mechanism (located)

- `srt/hardware_backend/mlx/model_runner.py:237` — `self._max_seq_len = 4096`;
  every `ContiguousAttentionKVCache` starts at 4,096 tokens.
- `srt/hardware_backend/mlx/kv_cache/attention_kv_cache.py:83` — `_grow()`
  doubles capacity and slice-copies `offset` tokens into the new buffers. A
  deep prefill walks the whole ladder (4K→8K→…→131072→262144); at the
  131072→262144 step the transient is old buffers + new buffers + copy across
  every attention layer at bf16 — the observed ~133K death during the 192K
  attempt (`benchmarks/longctx-bisect/`).
- Key property: `_grow()` skips the copy when `offset == 0` — growing an
  **empty** cache is a pure allocation. Pre-sizing right after acquisition
  costs no copy traffic and produces exactly one allocation at the final size
  (which is also smaller than the doubled power-of-two: 196,608 vs 262,144 at
  192K, a 25% saving per layer).

## Design

Pre-size each acquired cache to the request's total expected length:
`min(len(req.origin_input_ids) + max_new_tokens + margin, context_len)`.
`prefill_start` receives `req`, so the length is known at first acquisition;
continuation chunks reuse the request's cache object and never hit `_grow`.
Short requests keep today's 4,096 default (pre-size only when the request
exceeds it) so the mixed-workload memory profile is unchanged.

Open implementation question: confirm which path continuation chunks take
under chunked prefill with the radix cache disabled (`prefill` /
`MlxPendingExtend` vs repeated `prefill_start`), and that the pooled-cache
reuse path (`_acquire_cache` pool pop) re-sizes correctly (a pooled cache has
`offset = 0`, so an upfront `_grow` remains copy-free).

## Method

1. Implement per-request pre-sizing (new numbered patch; regenerate from the
   working tree). Land through the full gate protocol
   (`test_patch_gates.sh`, `import_smoke.py`).
2. Sanity: probe gate subset on qwen36 (the change touches the serving hot
   path) + the schema-v2 tripwire compare on qwen36 (armed baselines catch a
   decode regression >10%).
3. Validate 160K then 192K single-user:
   `CTX=210000 MEM_FRAC=0.5 EXTRA_ARGS="--disable-radix-cache"
   launch.sh qwen36 --kv-cache turboquant`, `oom_guard.sh` + `mem_profile.sh`
   running, `scripts/bench/bench_long_context.py` (server-verified
   `usage.prompt_tokens`), long urllib timeout. Watch for the NEXT
   constraint to surface (memory plateau vs a new boundary) — record
   whatever it is with receipts either way.
4. If 192K passes: 256K attempt (`CTX=270000`, same recipe). The KV pool and
   DeltaNet aux state budgets are the expected next walls.
5. Receipts to `benchmarks/longctx-bisect/` (extend the existing run table).

## Success criteria

- A 192K-labeled run completes with server-verified `usage.prompt_tokens`
  ≥ 0.95 × label and no guard kill; receipts committed.
- No decode/tripwire regression at 1024/8192/32768 on qwen36 (schema-v2
  compare PASS).
- 256K attempted with the outcome recorded (pass or the next boundary
  characterized).

## Kill criteria

- Pre-sizing lands but 192K still dies at the same ~133K point → the
  boundary attribution is wrong; stop and re-bisect with mem_profile before
  more code.
- Any hot-path change that flips a probe verdict or trips the tripwire →
  revert, record, re-scope.

## Phase 2 (separate serving windows)

Decode TPOT at depth: 13 s/token at 128K. The serial decode path reads the
full bf16 contiguous cache every token; candidate directions — quantized
contiguous cache (mirror the pool's turboquant), attention-read chunking, or
routing deep-decode through the pool-backed batched path. Spec after phase 1
lands.
