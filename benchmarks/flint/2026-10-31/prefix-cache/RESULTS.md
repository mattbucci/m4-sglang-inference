# Flint oQ8e prefix cache (patch 021) — gate, memory, determinism

Server: `scripts/launch.sh flint` defaults (oQ8e, CTX=32768, MAX_RUNNING=4,
CHUNKED=1024, 2 GB MLX buffer cache, `--max-total-tokens 131072` = a 3.7 GB
full-layer pool) unless noted. Radix off = the same preset with
`EXTRA_ARGS="--disable-radix-cache"`. Greedy requests, 3-turn conversation
whose turns 2 and 3 extend turn 1 (`prefix_on.json` / `prefix_off.json`;
turn texts inside).

## Prefix hit: time to first token (gate021.log)

| Turn | Prompt tokens | Radix on, cold tree | Radix on, second pass (full-prefix hits) | Radix off |
|-----:|--------------:|--------------------:|------------------------------------------:|----------:|
| 1 | 13,339 | 17.25 s | 0.22 s | 17.43 s |
| 2 | 13,456 | 0.60 s | 0.48 s | 19.17 s |
| 3 | 13,571 | 0.62 s | 0.61 s | 19.47 s |

Radix on also passed, on the same server: `validate_capabilities.py` 3/3,
`probe_codegen.py --max-tokens 4096` STRONG, `test_radix_cache_repeat.py`
4/4 consistent, 0 OOM-guard warnings.

## Memory: kernel-available minimum per phase (memprobe-*.summary.txt)

Sequence on one server: cold 13.3K-token turn, hit turn, `/flush_cache`,
cold turn, hit turn. Pool `--max-total-tokens 65536` (1.9 GB) for the
radix-on rows; sampler at 0.5 s.

| Phase | Radix off | Radix on, lazy pool scatters | Radix on (patch 021 as shipped) |
|-------|----------:|-----------------------------:|--------------------------------:|
| after load | 18.6 GB | 18.6 GB | 16.6 GB |
| cold turn 1 | 18.6 GB | 12.8 GB | 16.0 GB |
| hit turn 2 | 17.9 GB | 14.7 GB | 15.4 GB |
| cold turn 1 after flush | 17.9 GB | 10.2 GB | 15.4 GB |
| hit turn 2 after flush | 17.9 GB | 14.7 GB | 15.4 GB |

The lazy-scatter column is the first build: `_sync_new_kv_to_pool` wrote
the pool with scatters nothing evaluated until a hit gathered from it, so
each chunk's scatter pinned the request KV buffers it read (their next
in-place write then copied the whole buffer) and the pool versions it
replaced; the pinned buffers outlived the flush. The shipped build hands the
touched pool buffers to the same `mx.async_eval` as the step
(`pending_pool_arrays`); what remains below radix-off is the pool itself.
With the 3.7 GB preset pool and a full-vocabulary logprob request on top,
the first build hit Metal's working-set limit
(`kIOGPUCommandBufferCallbackErrorOutOfMemory`); the shipped build serves
the same request.

## Determinism (hit-vs-cold-*.log, stats_fixed2.log, cold_chunk*.json)

Repeated cold runs are token-identical at every size; repeated hits on the
same cached prefix are token-identical (gate second pass). A hit versus a
cold run of the same prompt, three-turn conversation:

| Context (turn-2 prompt) | Flint text identical | Flint first-token top-5 logprob shift | qwen36 text identical | qwen36 shift |
|------------------------:|:--------------------:|--------------------------------------:|:---------------------:|-------------:|
| 2,975 tokens | yes | ≤ 0.36 nats (tail), top token 0.00 | yes | — |
| 6,873 / 7,368 tokens | no (reasoning prose, char 26) | ≤ 1.13 nats (tail), top token 0.00 | yes | ≤ 0.43 nats (tail), top token 0.003 |
| 13,396 / 14,386 tokens | no (reasoning prose, char 14) | — | yes | — |

Over 24 random positions (3.1K–16.1K tokens) of a captured 16.6K-token
agent request, greedy, the next-token distribution against the cold run at
chunk 1024 (`kvhit_stats.py`; `kvcold_record.py` / `kvcold_compare.py` for
the chunk-size arm):

| Arm | Top token same | Top-1 logprob shift, median / max | True-token shift, median / max |
|-----|:--------------:|----------------------------------:|-------------------------------:|
| hit, 1 new token after the cached prefix | 21/24 | 0.016 / 4.39 nats | 0.021 / 8.05 nats |
| hit, 2–1000 new tokens | 20/24 | 0.048 / 1.70 | 0.034 / 5.23 |
| cold run at chunk 2048 | 20/24 | 0.016 / 1.87 | 0.022 / 4.04 |

Every mismatch sits at a position where the cold top token has at most
0.74 probability (logprob −0.30 to −1.58: positions 4033, 9245, 9915, 12685,
15467), and the same positions move between the two cold runs. A hit
therefore differs from a cold run exactly as two cold runs on different
chunk grids differ: the new tokens prefill from the cached length on a
chunk boundary the cold run never had, and the bf16 accumulation order
moves the logits by that much. The attention kernel itself is
layout-invariant (contiguous, strided-view and pool-gathered K/V give
bit-identical `scaled_dot_product_attention` output at 2K–14K keys, bf16,
GQA 32/4). The same shift exists on qwen36's pre-existing hybrid path; its
answers stayed identical because its top token led by ≥ 5 nats.

## opencode `bash` tool: required `description` (hit-vs-cold-and-rule-placement.log)

Replay of the harness's captured request (opencode 1.15.4 system prompt,
nine tools, the 3K-character task prompt), model sampling, 10 samples per
row, counting bash calls that carry `description`:

| Prompt | with `description` |
|--------|-------------------:|
| unchanged | 0/12 |
| rule at the end of the task prompt | 1/11 |
| rule at the top of the task prompt | 0/15 |
| rule appended to the system prompt | 1/14 |
| stronger rule + example call, system prompt | 0/9 |
| stronger rule + example call, end of task prompt | 3/13 |

(On a two-line user prompt the same rule gives 10/10; unchanged it gives
2/10.) The harness proxy backfills the argument instead
(`evals/swebench/no_thinking_proxy.py`, `BACKFILL_BASH_DESCRIPTION=1`).

## Envelope with the radix cache on (`envelope_sampled` run, preset defaults, 3.7 GB pool)

| Depth | Prompt tokens | Prefill tok/s | Decode tok/s | Radix-off prefill / decode |
|------:|--------------:|--------------:|-------------:|---------------------------:|
| 4K | 3,495 | 924 | 50.3 | 931 / 58.6 |
| 16K | 13,863 | 794 | 47.0 | 805 / 54.9 |
| 32K | 27,039 | 645 | 43.1 | 633 / 49.7 |

Kernel-available minimum during the 27K prefill: 12.8 GB (17.3 GB with
the radix cache off); vm pressure level never above 1. Decode is ~12%
lower because the hybrid `no_buffer` bookkeeping disables the overlap
schedule.

## Finished-request release (patch 023): one-token hit cycles (memcycle-*.log)

Server: oQ8e, `CTX=65536`, a 65536-slot pool, 2 GB MLX cache, radix on,
chunk 1024. Each cycle: `/flush_cache`, a cold prefill of the first 10,068
tokens of a captured 16.6K-token agent request, then a hit on the first
10,737 (669 new tokens); `max_new_tokens` 1 with output logprobs, the shape
of the hit-vs-cold probes. `vm_stat` wired pages sampled after each request
(Metal's working-set limit on this box is 51.8 GB; `kern.memorystatus_level`
reads 94% available once the server has idled a minute, so it is not the
number to watch).

| Cycle | Without 023, wired after the hit | With 023 |
|------:|---------------------------------:|---------:|
| 0 | 53.0 GB | 46.7 GB |
| 1 | 53.6 GB | 46.7 GB |
| 2 | 54.0 GB, Metal OOM on the next cold prefill | 46.5 GB |
| 3 | — | 46.4 GB |

In-process (`mx.get_active_memory()` logged at prefill finalize and at
request removal): without 023 the runner's per-request cache table gains
one entry per request and no request is ever removed, because the
scheduler calls the worker's KV-release hook only for finished decoding
requests and the stale-rid sweep runs only on decode batches. Each retained
entry is the request's full-layer K/V, 0.27–0.29 GB at 10K tokens, cold or
hit alike, surviving `/flush_cache`; the first request that decodes sweeps
them all (2.2 GB back on a 2-token request after ten 1-token ones). The
first probe server on the unfixed stack died this way after 66 one-token
requests and 43 flushes. With 023 every request is removed as it finishes
and active memory holds at 41.5–41.9 GB across the cycles; the hits still
report their cached prefix.

## Harness ceiling (evals/swebench/runs/flint-oQ8e-prefix-cache-2026-10-31)

A 77K-token request (61K context + a 16K output) with the 3.7 GB pool
took the process to Metal's 51.8 GB working set
(`kIOGPUCommandBufferCallbackErrorOutOfMemory` on the next cold prefill,
kernel-available 9.0 GB). The SWE-bench harness bounds Flint at 64K:
`CTX=65536`, a 65536-slot pool (1.85 GB), opencode `limit.context` 65536.

