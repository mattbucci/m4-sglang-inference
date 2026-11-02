# Session endurance + append-to-cached-prefix turn tax (spec 12)

Serve under test: qwen36 radix-ON (the agentic default), CTX=42000,
MEM_FRAC=0.45, CHUNKED=2048, SGLANG_MLX_CACHE_LIMIT_GB=2, turboquant,
`--enable-cache-report`. Load: agentic-shaped turns (~1.5K tokens of
tool-output text appended per turn, 256-token replies, conversation rolls at
~32K). oom_guard armed throughout; per-turn JSONL + 1 s memory profiles in
this directory.

## Phase A — turn tax (append to a verified cached prefix)

| cached prefix (server-verified) | append | TTFT median | decode ms/tok | tax ratio |
|---:|---:|---:|---:|---:|
| 8,648 | 1 tok | 45 ms | 17.1 | **2.6×** |
| 8,648 | 64 tok | 199 ms | 17.1 | 11.6× |
| 34,535 | 1 tok | 117.6 ms | 21.8 | **5.4×** |
| 34,535 | 64 tok | 315.6 ms | 21.8 | 14.5× |
| ~64K | — | no receipt | — | envelope null (radix-on serve at CTX=70000 did not yield the rung) |

**R9700's R97-J 85× tax does NOT reproduce on MLX** — the M4 append-1 cost
is 2.6× decode at 8K and 5.4× at 32K (~120 ms absolute: imperceptible in an
agent turn). 5.4× is marginally over the spec's 5× threshold, so a
fix-experiment row is filed (profile whether the cost is pool-gather or
contiguous-cache rebuild) at LOW priority given the absolute number.

## Phase B — 150-turn endurance, three arms

| arm | turns survived | free+inactive start → trough → end | verdict |
|---|---:|---|---|
| radix-on | **killed at turn 111** (guard, 8 GB floor) | 29.3 → 8.1 → (kill) | ~0.19 GB/turn average erosion; cache hits verified to the end (turn 110: 27,520/29,297 cached, TTFT 3.3 s) |
| radix-off control | 150/150 | 29.3 → 14.2 → 17.9 | survives — the erosion is radix-attributed; but every turn pays full re-prefill: **~30 s TTFT/turn vs radix-on's ~3 s** |
| radix-on + `/flush_cache` every 25 turns | **150/150** | 28.8 → 10.8 → 19.9 | memory sawtooths back up at each flush; cache hits stay strong between flushes; per-flush cost ≈ one cold re-prefill (~30 s) per 25 turns |

## Phase C — the agentic follow-up (chat shape), `turn-tax-chat-*.json`, `turn-hit-determinism-*.json`

Phase A appended to the primed token sequence through `/v1/completions`, which
never crosses the chat template's generation tail. An agentic client replays
the history through `/v1/chat/completions`: the template renders the previous
assistant turn without its reasoning, so the follow-up prompt diverges from the
primed prompt 1 token (Qwen3.8: `<think>\n` vs `\n\n`) or 2 tokens (Qwen3.6)
before the primed end. The end-of-generation DeltaNet state therefore never
serves a follow-up; only checkpoints inside the prompt do. Driver:
`scripts/bench/measure_extend_cost.py --multi-turn --prime-max-tokens 64`
(thinking on, `--enable-cache-report`, CTX=65536, preset chunk sizes:
`qwen36` 4096, `qwen36-27b` 2048). "First follow-up" is the turn right after
the primed reply; later turns hit the checkpoint the first one tracked.

Before patch 026 the MLX runner tracked a 64-token-grid checkpoint only in the
first prefill chunk, so a prompt longer than one chunk kept states only at
chunk ends and the first follow-up recomputed the whole tail of the last chunk.
Continuation chunks now track too (`extend_start`), and the first follow-up
lands within a grid of the prompt end:

| model | primed prompt | first follow-up recompute, before → after | TTFT before → after | later turns (k=1) |
|---|---:|---:|---:|---:|
| Qwen3.8-27B-4bit (`qwen36-27b`) | 8,699 | 525 → 77 tok | 5.2 s → 1.0 s | 0.4 s |
| | 25,905 | 1,347 → 67 tok | 13.8 s → 0.9 s | 0.9 s |
| | 52,068 | 887 → 55 tok | 10.9 s → 1.1 s | 1.1 s |
| Qwen3.6-35B-A3B-4bit (`qwen36`) | 8,657 | 479 → 31 tok | 0.80 s → 0.19 s | 0.14 s |
| | 25,863 | 1,301 → 21 tok | 2.6 s → 0.20 s | 0.18 s |
| | 52,026 | 2,889 → 73 tok | 8.2 s → 0.53 s | 0.25 s |

Receipts: `turn-tax-chat-qwen38-base-*.json` / `turn-tax-chat-qwen36-base-*.json`
(upstream auxiliary-pool sizing, before 026), `turn-tax-chat-qwen38-base-64slots-*.json`
(before 026 with `AUX_SLOTS=64`: identical to base — the pool size does not
change a serial session), `*-p026fix-*.json` (after). Decode is unchanged
(Qwen3.8 79–99 ms/token, Qwen3.6 17–23 ms/token); the residual ~1 s on
Qwen3.8 is its 100–125 tok/s prefill over the ~60–75 recomputed tokens plus
the fixed per-turn cost.

**Auxiliary-state pool (`AUX_SLOTS`).** Each checkpoint in the tree holds one
DeltaNet/Mamba slot; upstream sizes the pool at 4 × `MAX_RUNNING`. In the first
baseline run a raw-completion decode comparator issued between the prime and
the follow-up (a second tree branch) evicted the primed turn's chunk
checkpoints and the follow-up recomputed the whole prompt (80 s at 8.7K, 245 s
at 25.9K, 545 s at 52K). The hybrid presets now launch with
`--max-mamba-cache-size 64` (`AUX_SLOTS`, ~79 MB per slot on Qwen3.8-27B,
~33 MB on Qwen3.6-35B-A3B, filled lazily) so a long session with tool-call
branches keeps its checkpoints.

**Determinism** (`probe_multiturn_hit_determinism.py`, 96 greedy tokens, two
hits from different checkpoints, then a cold run): every arm's hits are
reproducible, and a hit from a checkpoint inside a chunk differs from the cold
run the way two cold runs on different chunk sizes do (Qwen3.8 at 8.3K:
first divergent token 50 of 96; the pre-026 hit from a chunk-end state was
identical to cold at that depth, so this is the price of the mid-chunk split).
Validator 3/3, thinking gate VERIFIED on both models after the patch.

## Guidance (the daily qwen36 + opencode workload)

- **Radix-on is the right default** (10× per-turn latency win) **with a
  `/flush_cache` cadence**: every ~25 turns holds free+inactive comfortably
  above the guard floor for 150+ turns at ~1.2% amortized latency cost.
- Unmitigated radix-on sessions are bounded at **~100 turns** under a
  32K-rolling load before the box reaches the 8 GB guard line (and macOS
  has no OOM killer past it).
- Radix-on serving also cannot host deep one-shot prefills: 32K+ prefill
  with a big pool guard-kills (receipts: this session's CTX=140000 kill at
  free 7.85 GB, plus the M4-C coder-30b class) — deep single requests use
  the radix-off long-context recipe.

## Cross-rig

Turn-tax comparison for the R97-J fleet ask: M4/MLX 2.6× @8K, 5.4× @32K
(append-1, cache-verified) vs R9700 85× @176K — the MLX radix extend path
does not exhibit the CUDA extend-pass pathology at these depths. Delivered
to sister READMEs with the MLX buffer-cache-cap note.

## On patches 021–023 (serve-p023.log, turns-radixon-p023.jsonl, mem-radixon-p023.csv, extend-cost-p023*.json)

Same serve configuration and driver. The radix-on arm, which the stack above
lost at turn 111, runs out the full 150 turns with flat memory:

| arm | turns survived | free+inactive start → trough → end | TTFT median | decode | cache hits |
|---|---:|---|---:|---:|---|
| radix-on, patches 021–023 | **150/150** (0 guard warnings) | 28.4 → 24.7 → 26.2 GB | 2.8 s | 56 tok/s | to the end (turn 150: 13,349 of 15,128 cached after the ninth roll) |

Wired memory stays between 26.8 and 30.2 GB for the whole run. The erosion
in the earlier arm (0.19 GB/turn) is gone; what the three patches changed on
this path is the per-step evaluation of the pool scatters (021), the hit
cache's append across the split final chunk (022) and the release of each
finished request's caches at the scheduler hook (023). No flush cadence is
needed for this load.

Phase A on the same stack, fresh server (`extend-cost-p023-fresh.json`;
the post-endurance run `extend-cost-p023.json` is within 5 ms of it):

| cached prefix (server-verified) | append | TTFT median | decode ms/tok | tax ratio | time to 64 tokens |
|---:|---:|---:|---:|---:|---:|
| 8,648 | 1 tok | 80.0 ms | 16.0 | 5.0× | 1.10 s (earlier 1.14 s) |
| 8,648 | 64 tok | 231.4 ms | 16.0 | 14.5× | — |
| 34,535 | 1 tok | 259.6 ms | 18.8 | 13.8× | 1.46 s (earlier 1.51 s) |
| 34,535 | 64 tok | 483.9 ms | 18.8 | 25.8× | — |

The first-token cost is about twice the earlier receipt and decode is
faster by the same amount: since patch 021 the pool scatter is evaluated
with the step and reads the rebuilt contiguous cache, so the rebuild of the
gathered prefix lands in the first token instead of the first decode step.
Time to 64 tokens is unchanged. Profiling and the candidate fix are
experiments row 4.
