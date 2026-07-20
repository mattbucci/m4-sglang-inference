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
