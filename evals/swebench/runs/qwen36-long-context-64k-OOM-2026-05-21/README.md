# qwen36 long-context bench — 64K OOMs on v0.5.12 + per-feature stack

## Quick bench

Ran `scripts/bench/bench_long_context.py --contexts 256 4096 16384 65536`
against fresh qwen36 server at CTX=131072 with the CLAUDE.md-blessed
"long-context launch flags (validated 2026-05-11 on v0.5.11)":

```bash
CTX=131072 EXTRA_ARGS="--kv-cache-dtype turboquant \
    --chunked-prefill-size 2048 \
    --mem-fraction-static 0.5" \
    bash scripts/launch.sh qwen36
```

## Result

| Context | Input | Time | TPOT | Combined throughput | Status |
|---|---:|---:|---:|---:|---|
| 256 tokens | 246 | 1.3 s | 20.6 ms | 48.5 tok/s | ✓ |
| 4K tokens | 3,934 | 5.8 s | 90.0 ms | 11.1 tok/s | ✓ |
| 16K tokens | 15,729 | 22.6 s | 353 ms | 2.8 tok/s | ✓ |
| **64K tokens** | 65,536 | — | — | — | **OOM-killed mid-prefill** |

The 64K request started prefilling at ~570-810 tok/s. The OOM guard
fired at `free=6.64GB < 8GB` after ~30 s of prefill, when the chunked
prefill had consumed roughly 40K of the 64K input tokens (pending-token
counter was 24004 from initial 60868). Server process killed; the
remaining 14 minutes saw `health: DOWN`.

This is the **same recipe that was previously validated for 128K** on
v0.5.11 per `CLAUDE.md`. Now it can't fit 64K on v0.5.12.

## Why this matters

This is a **regression at the primary optimization target**.
`CLAUDE.md`:

> **Primary:** single-user **256K context** performance (decode tok/s,
> TPOT). Measure at long context first — that is the workload Apple
> Silicon is uniquely good at.

We just lost the ability to even *run* a 64K bench under the prescribed
recipe. Either the bench broke or the recipe broke. Either way, the
primary target is currently inaccessible.

## Possible causes (not investigated)

1. **v0.5.11 → v0.5.12 memory regression.** v0.5.12 added on-the-fly
   `mlx_q4`/`mlx_q8` quantization (+69 lines in `model_runner.py`).
   These code paths are dormant when `quantization=None` (our default),
   but the import surface may have grown. Less plausible.
2. **System overhead grew between May 11 and today.** macOS background
   processes, browser caches, Spotlight reindexing — any of these
   could have eaten ~8-12 GB of headroom that the 0.5 mem_fraction
   relied on.
3. **MLX library updated** (not pinned in setup.sh). The mlx_lm /
   mlx-vlm path may have higher per-layer activation memory now.
4. **macOS jetsam thresholds changed.** Less compressed-memory budget
   means OOM-guard's 8 GB free trigger fires sooner.

Without isolating, can't say which. The empirical fact is: 64K bench
OOMs today.

## Actionable

1. **Update CLAUDE.md's "validated 2026-05-11" claim** — it's outdated.
   Either re-validate on today's stack with the same flags, or document
   the new working flags.
2. **Lower mem-fraction-static for 64K+ work.** Standard 0.5 isn't
   enough now. Try 0.4 or 0.35 on a future loop tick. (Per memory
   `feedback_mem_frac_unified_memory`: this is the safe direction —
   the lever moves DOWN for long context.)
3. **Investigate via 32K bench**: 16K worked, 64K OOM'd. The boundary
   is somewhere between. A 32K probe would narrow the actionable
   recipe.
4. **Fix OOM guard's pgrep scope.** Today the guard killed my bash
   shell (PID 54842) because the command line contained the literal
   text "sglang.launch_server" inside the `eval '...'` block.
   `pgrep -f "sglang.launch_server"` matches command-line text, not
   process. Should be `pgrep -af 'python.*sglang\.launch_server'`
   anchored to the binary invocation, or use `pgrep -P` to scope to
   the launcher's children.

## What still works at long context

Decode-rate-only (excluding prefill): ~60 tok/s sustained at 8K+ context
per yesterday's launch.log samples — that part isn't regressed.

The throughput numbers from the working tests are total time / output
tokens (prefill amortized over output), so combined throughput LOOKS
slow but is mostly prefill cost:

| Context | Prefill (calc.) | Decode | Implied prefill rate |
|---|---:|---:|---:|
| 256 | 0.2 s | 1.07 s | 1170 tok/s |
| 4K | 4.7 s | 1.07 s | 832 tok/s |
| 16K | 21.5 s | 1.07 s | 731 tok/s |

Prefill rate degrades with context (memory-bound, attention quadratic),
which is expected. At this rate 64K prefill would take ~90-120 s if it
fit in memory.

## Files

- `bench-results.txt` — raw bench output up to OOM
- The `/tmp/qwen36-bench-boot.log` server log was lost when the OOM
  guard cleared `/tmp` references; only the summary above survived
