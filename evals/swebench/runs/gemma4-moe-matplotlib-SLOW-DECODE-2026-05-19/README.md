# gemma4 MoE on matplotlib-18869 — slow decode + TIMEOUT, not jetsam

## Task #79 — boundary mapping (corrected diagnosis)

After pylint-5859 RESOLVED and django-11001 conversational-fallback,
testing matplotlib-18869 (large codebase) to find the boundary
between gemma4 MoE's "works" and "falls back" regions.

## Result: TIMEOUT at 1825 s, 4 tool calls

```
done rc=124 elapsed=1825s diff=EMPTY (0 bytes)
tool_calls=4 (3 glob, 1 read)
```

opencode hit TIMEOUT=1800 having made only 4 exploration tool calls
across 30 minutes of wall.

## Why this happened — slow decode at large context

opencode's prompt for matplotlib loaded **27,973 input tokens** —
significantly more than prior tests:

| Instance | Input tokens | Outcome | Decode rate |
|---|---:|---|---|
| pylint-5859 | ~3K | RESOLVED (48 tool calls, 1003B patch in 1324s) | normal |
| django-11001 | ~19K | Conversational fallback (4 calls, 123s) | model gave up |
| matplotlib-18869 | **28K** | TIMEOUT (4 calls, 1825s) | **0.7 tok/s** |

The launch.log shows decode at **0.70 tok/s** on a 28K-token context
(vs typical >50 tok/s for qwen36 on similar contexts). gemma4 MoE's
heterogeneous attention (sliding-window RotatingKVCache for most
layers + ContiguousKVCache for the few full-attention layers) is
much slower than uniform attention at large contexts on M4.

## Initial misdiagnosis (corrected)

A prior version of this README documented "jetsam suspected" because
the launch.log appeared to stop mid-decode. Actual analysis after
smoke.sh fully exited (1825 s in) shows the harness completed
cleanly — opencode just ran out of TIMEOUT before the model could
synthesize. Decode was working, just glacially slow due to the 28K
context exercising the sliding+full mixed attention path.

This puts gemma4 MoE on matplotlib in the **decode-rate-miss** class,
same as qwen36-27b / nemotron-30b — not the jetsam class. The model
COULD synthesize a patch given enough time; the 30-minute budget
isn't enough at this context size.

## Pattern across the 3 gemma4 MoE tests

| Codebase | Input tokens | Decode rate | Outcome | Class |
|---|---:|---|---|---|
| pylint | ~3K | normal (~50 tok/s) | RESOLVED in 1324s | full agentic |
| django | ~19K | ~50 tok/s on first turn, drops later | Chat-mode fallback in 123s | model-side give-up |
| matplotlib | ~28K | **0.7 tok/s** | TIMEOUT, 4 calls in 1825s | decode-rate miss |

Three distinct failure regimes at different context sizes. The model
is reliable at small contexts, lapses into conversational text at
moderate contexts, and grinds to a halt at large contexts.

## Recommendation refinement

gemma4 MoE @ T=1800 is **small-codebase only**:
- Works at input ≤~5K tokens (full agentic)
- Falls back at input ~10-20K tokens (conversational text)
- Times out at input >~25K tokens (sliding+full attention decode rate)

Larger TIMEOUT (3600s) might unlock matplotlib but the underlying
decode-rate issue means it would still be 2-3 hours wall — not
practically useful. The matplotlib failure is bandwidth-bound, not
synthesis-bound.

qwen36 stays the only context-size-agnostic reliable primary.

## Files

- `run.log` — smoke.sh stdout (Smoke done rc=0, rollout rc=124)
- `predictions.jsonl` — empty-patch record
- `launch.partial.log` — last 15 launch.log lines showing the 0.7 tok/s decode rate

Original analysis revision: this run was NOT jetsam-killed. The
file timestamps initially looked silent because decode was so slow
that subsequent batch log entries were rare. Confirmed by the late-
firing watch task that captured the full smoke.sh completion at
05:41:38.
