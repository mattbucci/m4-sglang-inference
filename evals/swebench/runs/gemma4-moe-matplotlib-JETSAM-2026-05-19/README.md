# gemma4 MoE on matplotlib-18869 — jetsam suspected, no data point

## Task #79 — boundary mapping continued

After pylint-5859 RESOLVED and django-11001 conversational-fallback,
testing matplotlib-18869 (moderate codebase) to find the boundary
between gemma4 MoE's "works" and "falls back" regions.

## Result: server killed mid-decode (jetsam likely)

```
[05:11:06] Server ready
[05:11:07] opencode rollout starts
[05:13:31] Prefill completes: 27,973 input tokens
[05:13:32] First decode batch logged: 0.70 tok/s
[~05:13:33] Server goes silent
[~05:43] smoke.sh TIMEOUT (presumed); script killed without writing "Smoke done"
```

- predictions.jsonl: file exists but EMPTY (no records)
- No tool calls captured
- launch.log ends mid-decode with no error trace

## Why this happened

opencode's prompt for matplotlib loaded **27,973 input tokens** —
significantly more than prior test cases:

| Instance | Input tokens | Result |
|---|---:|---|
| pylint-5859 | ~3K | RESOLVED |
| django-11001 | ~8K (step 1), 19K (step 2) | Conversational fallback |
| matplotlib-18869 | **27,973** | Server died mid-decode |

Combined with gemma4 MoE's heterogeneous attention (RotatingKVCache
for sliding-window layers + ContiguousKVCache for full-attention),
this context size triggered macOS jetsam mid-decode. Decode rate
dropped to 0.7 tok/s on the first (and only) decode batch — the
model was probably already swapping when jetsam fired.

## Pattern across the 3 gemma4 MoE tests

| Codebase | Input tokens | Outcome | Failure mode |
|---|---:|---|---|
| pylint (small) | ~3K | RESOLVED (1003B patch) | n/a |
| django (moderate-large) | 19K | Conversational fallback (4 calls) | Model gives up agentic, types text |
| matplotlib (large) | **28K** | Server crashed | Jetsam (stack-side) |

**The recommendation now has 3 failure modes**:
1. Small codebase: works
2. Moderate codebase: conversational fallback
3. Large codebase: jetsam crash

## Recommendation update

gemma4 MoE @ T=1800 is **best limited to small-codebase instances**.
For codebases with directory trees that produce 10K+ token opencode
prompts, the model either falls back conversationally OR hits
hardware jetsam.

| Pick | Status | Codebase size limit |
|---|---|---|
| qwen36 | Reliable primary | All sizes verified |
| qwen35 (T=1800) | Reliable but slow | All sizes (15× wall) |
| gemma4 MoE (T=1800) | Small-codebase only | Inputs <~5K tokens |

Worth retrying on the SAME instance (pylint-5859) to verify the
pylint win is reproducible. If it is, the recommendation stands as
"gemma4 MoE for small-codebase Gemma-family work."

## Files

- `run.log` — smoke.sh stdout (truncated, no Smoke done)
- `predictions.jsonl` — empty file (rollout abnormally terminated)
- `launch.partial.log` — last log entry was decode at 0.7 tok/s
