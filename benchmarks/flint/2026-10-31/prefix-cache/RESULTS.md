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

## Determinism (hit-vs-cold-*.log)

Repeated cold runs are token-identical at every size; repeated hits on the
same cached prefix are token-identical (gate second pass). A hit versus a
cold run of the same prompt:

| Context (turn-2 prompt) | Flint text identical | Flint first-token top-5 logprob shift | qwen36 text identical | qwen36 shift |
|------------------------:|:--------------------:|--------------------------------------:|:---------------------:|-------------:|
| 2,975 tokens | yes | ≤ 0.36 nats (tail), top token 0.00 | yes | — |
| 6,873 / 7,368 tokens | no (reasoning prose, char 26) | ≤ 1.13 nats (tail), top token 0.00 | yes | ≤ 0.43 nats (tail), top token 0.003 |
| 13,396 / 14,386 tokens | no (reasoning prose, char 14) | — | yes | — |

Cause: the hit prefills the new tokens from the cached prefix length and
splits the first new chunk at the 64-token checkpoint grid (the tracked
snapshot), where the cold run prefills on its 1024-token chunk grid; the
different accumulation order moves bf16 logits at the noise level. The
attention kernel itself is layout-invariant (contiguous, strided-view and
pool-gathered K/V give bit-identical `scaled_dot_product_attention` output
at 2K–14K keys, bf16, GQA 32/4). The same shift exists on qwen36's
pre-existing hybrid path; its answers stayed identical because its top token
led by ≥ 5 nats.

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
