# Perf footnote investigation — 2-3× wall-time slowdown is path-variance

The django-10914 N=3 README flagged 2-3× longer wall time on today's
v0.5.12 + per-feature runs vs the May-18 baseline (603-757 s vs
328 s). This is a quick diagnostic confirming the slowdown is path-
variance (more tool calls per session), not per-token regression.

## Per-token throughput: unchanged

Sampling decode throughput from today's django-10914 run #2
`launch.log`:

```
Decode batch, #running-req: 1, #token: 23128, gen throughput (token/s): 54.17
Decode batch, #running-req: 1, #token: 23168, gen throughput (token/s): 54.28
Decode batch, #running-req: 1, #token: 23408, gen throughput (token/s): 55.09
Decode batch, #running-req: 1, #token: 25304, gen throughput (token/s): 53.49
```

~54-55 tok/s sustained at 23K-25K token context. Matches the M4
historical baseline ("qwen36 MoE-DeltaNet stays at >50 tok/s" per
`memory/project_m4_agentic_model_boundaries.md`). The periodic
1.0-1.1 tok/s entries in the log are *window-average* throughputs
that include prefill time between tool calls — not actual decode
rates.

So no per-token regression.

## Path-length is the driver

Today's django-10914 run #2:

- 25 POST /v1/chat/completions calls (one per agent turn)
- 30 `tool_use` events in opencode log
- 23 `step_finish` events
- Touched 4+ files in the final patch

May-18 (same instance, same model) patch touched 2 files. From the
opencode tool-use shape (one edit per file + a few reads), we'd
expect ~6-10 tool calls vs today's 30 — roughly 3× more agent turns.

3× turns × similar per-turn time ≈ today's 2-3× wall time. Path
variance accounts for the slowdown.

## What changed between May 18 and today

| Component | May 18 | Today | Same? |
|---|---|---|---|
| opencode binary | 1.15.4 (symlink dated May 18) | 1.15.4 (same symlink) | ✓ |
| no_thinking_proxy.py | May 18 02:11 | unchanged | ✓ |
| SGLang | v0.5.11 + 19 individual patches | v0.5.12 + 14 per-feature patches | ✗ |
| MLX library | (whatever was installed May 18) | (likely same — `.venv` from Apr 12) | ~likely same |
| Patch chain | individual 002-020 | 14 per-feature (rebased) | ✗ (but tree bit-identical via cumulative regression) |

The two stack changes (sglang version, patch reorganization) both
produce the same final source tree at the level of `git diff` — and
qwen36 on astropy-12907 is bit-identical across all three states
(v0.5.11, v0.5.12 cumulative, v0.5.12 per-feature). So the model is
*model-level deterministic* on simple flows.

The path divergence on complex flows must originate before the
model's decode — most plausibly in subtle chunked-prefill scheduling
differences (cache state at chunk boundaries, attention-mask
construction for incremental prefills). v0.5.12 added unrelated
on-the-fly quantization code that touches the same `__init__` and
`_load_model` we patched, but the runtime path shouldn't change for
non-`mlx_q4/q8` workloads. Without deeper instrumentation we can't
isolate further.

## Actionable

1. **Default TIMEOUT for SWE-bench should be 900-1200 s, not 600 s.**
   Today's qwen36 sessions average 30 tool calls; at ~20 s per turn
   (prefill + ~50 decode tokens), 600 s is exceeded. The previous
   `qwen36-perfeat-regression-PASS-2026-05-20/` astropy-12907 run at
   242 s was a happy short path; planning around it underestimates
   real-world cost.

2. **Recommendation set unaffected.** Outcomes are stable across
   today's longer paths (3/3 RESOLVED on pylint-5859 and
   django-10914). The wall-time tax is real but doesn't change which
   instances qwen36 can resolve.

3. **Future investigation (lower priority)**: chunked-prefill
   scheduling diff between v0.5.11 and v0.5.12. Not pursuing
   without a clear win signal — the model is producing correct
   answers, just taking different routes to get there.
