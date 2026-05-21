# qwen36 django-11001 N=2 attempt — TIMEOUT, strategic delegation variance

## Result

| N | When | TIMEOUT | Wall | Bytes | RESOLVED |
|:-:|---|---:|---:|---:|---|
| 1 | May 18 (v0.5.11) | unknown | 772 s | 1368 | YES ✓ |
| 2 | Today (v0.5.12 per-feature) | 1200 s | 1204 s | **0** | NO (rc=124) |

Today's run hit TIMEOUT=1200 with **zero patch bytes**. score_local
correctly reports 0/1 RESOLVED (no patch to apply).

## What today's model did differently

opencode log shows only 1 tool_use event before timeout — but the tool
was `task`, which **spawns a sub-agent** with this prompt:

```
description: "Explore Django order_by bug"
prompt: "I need to understand a bug in Django's SQLCompiler.get_order_by()
        method. The issue is that when using multiline RawSQL expressions
        in order_by()..."
```

The sub-agent then chewed through 13 chat-completion calls on the server
(observed in launch.log), reaching 31K-token context, and never returned.
TIMEOUT=1200 fired before any code edit landed.

May 18's run completed in 772 s with a 1368 B working patch — meaning
that run did NOT delegate to a sub-agent; it worked directly with read /
grep / edit tools and finished.

## A new variance shape: strategic delegation

Previous variance shapes characterized across 4 instances:

| Instance | Variance shape |
|---|---|
| django-11039 | comment prose only |
| pylint-5859 | regex-anchor syntax |
| django-10914 | scope of inert doc edits |
| requests-1963 | conceptually distinct fix strategies |

This iteration adds a **fifth shape**:

**django-11001: strategic delegation** — May 18 worked the problem
directly; today delegated to a sub-agent that never finished. Same
model, same instance, different choice of *meta-tool-use pattern*.

## Implication for the recommendation set

This is **NOT** a capability regression. The model still has the right
"engagement" — it tried to investigate the bug. It just picked a
strategy (sub-agent task) that takes longer than working directly,
exceeding even TIMEOUT=1200.

Possible mitigations (future loop work):

1. **Increase TIMEOUT to 1800 s** for the SWE-bench default. The
   smoke.sh just bumped 600 → 900 a few days ago; the sub-agent
   variance suggests 1800 is needed to cover today's path elaborations.
2. **Disable the `task` tool in opencode config**. The sub-agent
   abstraction adds latency without a clear win on M4 contexts;
   leaving it enabled makes wall time unpredictable. opencode.jsonc
   can override tools list per-model.
3. **Accept this as variance**. With higher N, some runs would not
   delegate and would resolve in <800 s like May-18.

## Tally update

3/5 May-18 RESOLVED confirmed N=3 outcome-stable:
  - ✅ pylint-5859, django-10914, django-11039

1 unfinished due to TIMEOUT (capability vs delegation strategy
question deferred):
  - ⏳ django-11001 — today's sub-agent strategy didn't fit in 1200 s
       budget; N=2 inconclusive

1 still pending:
  - ⏳ sphinx-10325

## Files

- `predictions.jsonl` — the empty rollout record
- `scores.jsonl` — resolved=False (no patch)
- `opencode.log` — the single `task` tool_use that delegated
