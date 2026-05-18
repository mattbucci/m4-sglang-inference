# SWE-bench Lite 1-instance smoke — coder-30b, 2026-05-18

First end-to-end agentic-coding rollout on the M4 SGLang+MLX stack.

## Config

- **Server:** `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ` served as `coder-30b` on port 23334
- **Launch:** `CTX=131072 EXTRA_ARGS="--kv-cache-dtype turboquant --chunked-prefill-size 2048 --mem-fraction-static 0.5" bash scripts/launch.sh coder-30b`
- **Agent:** `opencode 1.15.4` via `evals/swebench/run_rollouts.py`
- **Instance:** `astropy__astropy-12907` (first SWE-bench Lite instance, deterministic)
- **Per-instance timeout:** 600s
- **Greedy decode** (MLX backend, temperature=0)

## Result

```
rollout_returncode: 0
rollout_seconds:    32.7s (server boot 20s + opencode 12.7s)
opencode elapsed:   23.4s
diff_size:          0 bytes (EMPTY)
tokens:             8086 input / 296 output (cumulative)
```

## What happened

opencode connected to the SGLang server successfully, did one `glob`
tool call to locate `astropy/modeling/separable.py`, generated a
reasoning step, then **gave up and asked the user for the file
content** instead of using a `read` tool call:

> "I don't have access to the actual file content to see the specific
> code changes or issues. To help you with this file, I would need:
> 1. The actual file content to review..."

The model emitted a `step_finish` with reason `stop` after one tool
call + one text turn. The agent loop ended. No code edits, no patch.

## Interpretation

**Plumbing: ✅ verified end-to-end.**

- opencode → SGLang OpenAI-compatible API works (`sglang/coder-30b`
  resolved correctly after installing the global config at
  `~/.config/opencode/opencode.jsonc`).
- Tool-call parsing works (`glob` invocation + structured output).
- Long-context launch recipe survives (131K with turboquant + chunked
  prefill 2048 + mem-fraction 0.5; server boot 20s).
- Server tore down cleanly, 0 orphans.

**Model behavior at task: ⚠️ underwhelming.**

- `Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ` at greedy decode + the
  opencode system prompt did NOT chain `glob → read → edit` calls.
- Could be: (a) opencode system prompt not strongly directive enough
  for this checkpoint, (b) greedy temp 0 producing the most-likely
  ("ask user") completion rather than continuing tool use,
  (c) the DWQ recipe's instruction-following tuning differs from the
  standard 4-bit upload.

## Next steps

1. Try `qwen36` (35B-A3B MoE + DeltaNet) at the same instance — it
   was the README's "long-context flagship" pick and may iterate more
   aggressively.
2. Try a smaller-instance prompt before scaling to 3-5 instance runs.
3. If model behavior persists, compare to the 3090 stack's
   `coder-30b-eval` (AWQ_Marlin) numbers to determine whether this
   is the DWQ checkpoint specifically or the agent-prompt template.
