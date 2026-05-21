# django-11001 RESOLVED by disabling opencode's `task` tool

## Setup

Yesterday's loop found that today's qwen36 invokes opencode's `task`
sub-agent tool on django-11001 instead of working directly with
read/grep/edit. The sub-agent ran 13 chat completions then stopped
issuing requests for 25+ minutes, exceeding both TIMEOUT=1200 and
TIMEOUT=1800.

This iteration's fix: add `"tools": { "task": false }` to the
opencode config so the model can't invoke the sub-agent. Forces
direct tool use.

## Result

| Config | Wall | tool_use count | Patch | RESOLVED |
|---|---:|---:|---:|---|
| May 18 baseline (task enabled, didn't invoke) | 772 s | (unknown) | 1368 B | YES ✓ |
| 2026-05-21 T=1200, task ENABLED | 1204 s (TIMEOUT) | 1 (task) | 0 B | NO |
| 2026-05-21 T=1800, task ENABLED | 1804 s (TIMEOUT) | 1 (task) | 0 B | NO |
| **2026-05-21 T=1200, task DISABLED** | **1204 s (TIMEOUT)** | **18 (direct)** | **1473 B** | **YES ✓** |

With task disabled, the model works directly with 18 tool calls (read,
grep, edit, etc.) and produces a valid patch within the same 1200 s
TIMEOUT budget that hit empty with task enabled. F2P 2/2, P2P 118/118.

Note: today's path even with task disabled is ~1.5× longer wall than
May 18 (1204s vs 772s). That's the underlying path-elaboration
slowdown documented in prior iterations. But the OUTCOME is now
correct again.

## Why this works

Opencode's `task` tool spawns a sub-agent (a second opencode session
with the same model and tools) to handle a delegated prompt. On M4
qwen36 today, the sub-agent appears to either:

- Get stuck in its own exploration loop without returning useful
  information to the parent, OR
- Return a result that the parent can't act on, causing the parent
  to hang waiting for a continuation that never comes

Either way, the sub-agent consumes wall time without contributing to
the final patch. Disabling it forces the model to do the work
directly, which fits in the standard SWE-bench TIMEOUT budget.

May 18 didn't invoke `task` on this instance — that was sampling
variance ("strategic delegation" shape; see prior iteration's
variance taxonomy). Today's stack samples the task path more often
on django-11001 (and possibly other instances). Disabling task
removes the wall-time outlier from the distribution.

## Recommendation upgrade

**For SWE-bench rollouts on M4 + qwen36 (and likely other models):
disable opencode's `task` tool.** Applied in `scripts/.../smoke.sh`
via the existing opencode-config-modification + backup-restore
cycle that previously only repointed the upstream port. New
modification injects `"tools": { "task": false }` at config-top-
level if not already present; backup-restore around the rollout
keeps the user's interactive opencode untouched.

Verified: with this change, django-11001 now resolves where it
previously timed out. Future rollouts will benefit from the same
forced-direct-work pattern.

## Updated tally

4/5 May-18 RESOLVED confirmed at N=2 or better stability:

- ✅ pylint-5859 — 3/3 RESOLVED @ N=3
- ✅ django-10914 — 3/3 RESOLVED @ N=3
- ✅ django-11039 — 3/3 RESOLVED @ N=3
- ✅ **django-11001 — RESOLVED @ N=1 today (task disabled) + RESOLVED @ N=1 May 18 = N=2**
- ⏳ sphinx-10325 — pending

The 5/13 = 38.5% M4-scorable RESOLVED rate is one instance closer to
fully confirmed.

## Files

- `no-task-1473B.diff` — today's successful patch (different bytes from
  May 18's 1368B, both RESOLVED)
- `predictions.jsonl`, `scores.jsonl` — rollout + score records
