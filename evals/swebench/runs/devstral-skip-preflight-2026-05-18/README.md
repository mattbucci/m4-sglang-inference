# devstral with --skip-preflight — confirmed model-side gap, not gate issue

## Why this test

In the int4 bakeoff, devstral's only blocker was the preflight canary
(Mistral chat template rejects the canary's user→assistant(tool_calls)
→tool→user sequence). To know whether Mistral arch can actually do
agentic coding on M4, added `--skip-preflight` and ran devstral against
the standard astropy__astropy-12907 baseline.

## Result: model emits text, not tool calls

```
rc=0  elapsed=343s  diff=EMPTY (0 bytes)  tool_calls=0
```

- Server ran for 5.7 minutes
- opencode session ended cleanly (`step_finish reason=length`)
- Model generated **4096 output tokens / ~21 KB of text** — explanatory
  prose about `separability_matrix` and compound models
- Zero tool calls. The Mistral chat template doesn't engage opencode's
  tool-call protocol; the model just answers the task as if it were a
  question, not an agentic loop.

The preflight had been surfacing a real signal: the same template that
rejected the canary also doesn't accept opencode's tool prompts in the
way that produces tool calls. The preflight gate was over-strict in
how it surfaced the failure (400 Bad Request from server) but the
underlying conclusion is correct — Mistral-arch can't drive opencode
agentically on this stack.

## Recommendation set: UNCHANGED

| Architecture | Result |
|---|---|
| Qwen3.6 MoE (qwen36) | ✓ Works |
| Qwen3.5 Dense+DeltaNet (qwen35) | ✓ Works at TIMEOUT=1800 |
| Qwen3 family Dense (qwen3-32b, qwen36-27b) | ✗ Decode too slow |
| Qwen3 MoE non-DeltaNet (qwen3-moe) | ✗ Parser mismatch |
| Qwen3-Coder MoE (coder-30b) | ✗ Chat template gives up |
| **Mistral arch (devstral)** | ✗ **Chat template emits text, not tool calls** |
| Gemma 4 Dense (gemma4-31b) | ✗ 0 tokens emitted under tool prompts |
| NemotronH (nemotron-30b) | ✗ Decode too slow |
| NemotronH+VLM (nemotron-omni) | ✗ MLX backend layer-indexing bug |

**10 of 10 int4 alternatives to qwen36 are now exhausted with model-side
or stack-side reasoning (not gating artifacts).**

## Harness improvements that survive this iteration

1. **`run_rollouts.py --skip-preflight`** — new flag to bypass the
   tool-call-canary preflight. Useful for any model whose chat
   template doesn't accept the canary's user→assistant(tool_calls)→
   tool→user sequence. Doesn't relax the per-instance health check.

2. **Per-instance health check uses `/health` endpoint**, not the
   tool-call canary. Decouples liveness checking from chat-template
   validation. Mistral-arch models can now flow through per-instance
   restart sweeps without false abort.

## Files

- `run.log` — smoke.sh output
- `predictions.jsonl` — empty-patch record
- `astropy__astropy-12907.log` — opencode session log (text emission, no tool calls)
- `astropy__astropy-12907.env.log` — astropy venv install fail (irrelevant — fell back to no-venv)
