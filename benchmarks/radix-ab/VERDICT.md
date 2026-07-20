# radix-ab — qwen36 MR=1: radix cache vs overlap schedule

Both arms identical except the radix flag: `CTX=36000 MEM_FRAC=0.45
SGLANG_MLX_CACHE_LIMIT_GB=2 --kv-cache turboquant`, single user, pinned
depth instrument (`--random-range-ratio 1`, 64 output tokens). Radix-on
hybrids run the normal event loop (`no_buffer` constraint); radix-off runs
the MLX overlap loop.

## Decode (tok/s, MR=1)

| depth | radix-on | radix-off | radix-on cost |
|------:|---------:|----------:|:-------------:|
| 1024  | 63.9 | 81.1 | −21.2% |
| 8192  | 59.4 | 73.8 | −19.5% |
| 32768 | 48.6 | 57.3 | −15.2% |

Radix-off matches the independently-armed baseline serve within ±1.5%
(57.3 vs 57.8 @32K) — instrument reproducibility receipt.

## TTFT (prefix reuse)

| measurement | radix-on | radix-off |
|---|---:|---:|
| 8K repeat, full prefix hit | **204 ms** | 11,205 ms |
| 32K point after 8K point (partial prefix) | 8,616 ms | 59,138 ms |
| distinct-seed 8K (no reuse), stable across 10 requests | ~5,100 ms | ~10,900 ms |

Full-prefix hit skips the entire prefill: **55× TTFT win**. (The
distinct-seed radix-on TTFT is ~half the radix-off one because the seeded
prompts still share partial token-level prefixes.)

## Cross-request memory (10 × 8K distinct prompts, free+inactive GB)

| | start | trough | end |
|---|---:|---:|---:|
| radix-on | 28.4 | **10.2** | 18.0 |
| radix-off | 27.8 | 14.7 | 22.4 |

Radix-on runs ~4.5 GB deeper and settles ~4.4 GB lower — the margin that
killed the genuine-32K prefills on the full-attention MoEs during the M4-C
re-measure (`benchmarks/coder-30b-4bit/results.json` error rows). The
retention is outside the static pool (pool usage is identical between arms).

## Decision

**Hybrid presets keep the radix cache ON** (current default, now
data-confirmed): the primary workload is multi-turn agentic coding with
large repeated prefixes, where a 55× TTFT win on every turn dominates a
15–21% decode loss (an agent turn re-prefilling 30K tokens costs ~60 s
radix-off vs ~0.2–9 s radix-on; the decode delta on a 500-token reply is
~1.6 s).

**Use `--disable-radix-cache` for single-shot deep-prefill work** (benches,
one-off long-document jobs, anything ≥32K on full-attention models): no
prefix ever repeats, the overlap loop decodes 15–21% faster, and the
serving envelope is ~4.5 GB safer. The long-context recipe already does
this.

Open thread for beyond-128k / upstream: what exactly radix-on retains
outside the pool across requests (~0.4 GB/request here) — same
investigation family as the patch-008 buffer-cache cap.
