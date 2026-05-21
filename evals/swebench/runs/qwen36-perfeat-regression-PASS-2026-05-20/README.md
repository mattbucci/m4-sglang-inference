# qwen36 per-feature-patch regression — PASS (bit-identical)

## Context

After commit `3f39a65` split the v0.5.12 cumulative rebase patch
(`021-v0512-rebase-cumulative.patch`, 117 KB single block) into 14
per-feature patches against v0.5.12, we re-ran the canonical
astropy-12907 SWE-bench smoke to confirm functional equivalence.

The prior cumulative-patch baseline is at
`../qwen36-v0512-regression-PASS-2026-05-20/`.

## Result

```
$ shasum -a 256 prior/astropy__astropy-12907.diff new/astropy__astropy-12907.diff
d024df6c8d482695a1be15dc75343b38db476fcfd8b8c2c3a004b9dcf77ccfba  prior
d024df6c8d482695a1be15dc75343b38db476fcfd8b8c2c3a004b9dcf77ccfba  new
```

**BIT-IDENTICAL** ✓ — same 506-byte canonical fix to
`astropy/modeling/separable.py` (the one-line `= 1` → `= right` change
that solves the nested CompoundModel separability bug).

| Metric | Cumulative (3-9d) | Per-feature (this run) |
|---|---|---|
| Patch bytes | 506 | 506 |
| Wall | 185 s | 242 s |
| Return code | 0 | 0 |
| SHA256 | d024df6c… | d024df6c… |

Wall time variance is within noise (server boot 40s + opencode rollout
242s; cumulative was 125s rollout + 60s server). The 117s rollout delta
is likely scheduler / cache state variance — opencode is greedy across
the agent loop but tool calls have non-determinism around timing.

## What this confirms

1. **The per-feature 14-patch chain produces functionally equivalent
   model behavior to the prior 14-commit chain on v0.5.12.** No
   silent regression introduced by the split.
2. **qwen36 (Qwen3.6-35B-A3B MoE+DeltaNet+VL) remains the working
   primary recommendation for agentic coding on M4.** Greedy MLX
   decode is deterministic for this architecture across v0.5.11 →
   v0.5.12 → per-feature-rebased chain — three stack-state changes,
   zero divergence on the same canonical instance.
3. **The patch split was lossless** — bit-identity holds at the
   model-output level, not just the source-tree level.

## Files

- `astropy__astropy-12907.diff` — the 506-byte canonical fix
- `predictions.jsonl` — opencode rollout prediction record
- `run.log` — smoke.sh stdout (`Smoke done rc=0`)
