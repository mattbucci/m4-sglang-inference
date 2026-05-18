# coder-30b + stronger directive prompt — same result as baseline

Followup to `coder-30b-smoke-2026-05-18/` (empty diff after one glob).

## Change

Stronger PROMPT_NO_VENV in `run_rollouts.py` with explicit tool-call
direction:

> **You have direct file-system access via tool calls.** Use the `read`
> tool to view any file's contents, the `glob` and `grep` tools to
> navigate, and the `edit` or `write` tools to apply your fix.
> **Never ask the user for file contents — you can read them yourself.**

## Result

| Metric | Baseline (2026-05-18 a) | Stronger prompt (this run) |
|--------|:-----------------------:|:--------------------------:|
| rollout_seconds | 32.7s | 33.6s |
| opencode tool calls | 1 × `glob` | 1 × `glob` |
| text parts | 2 | 2 |
| step_finish count | 2 | 2 |
| diff size | 0 bytes | 0 bytes |

Identical behavior. The model still emits one `glob`, then a text
turn where it hallucinates the file structure ("Looking at the
/separable.py, I can see..."), then `step_finish` with reason `stop`.
No `read` tool call. No edits. No patch.

## Interpretation

Prompt-template direction is NOT the bottleneck. This is a
**model-level limitation** of `Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ`
under greedy MLX decode + the opencode agent loop. The model treats
the system prompt + problem statement as a Q&A turn rather than a
multi-step tool-call loop. Greedy decode picks the most-likely
continuation (descriptive text about the file) over the next tool
call.

## Next experiments

1. **Try qwen36** (Qwen3.6-35B-A3B MoE+DeltaNet, the README's
   "long-context flagship"). If it iterates through tool calls where
   coder-30b gives up, the agentic-coding recommendation should
   shift to qwen36 even though coder-30b has the higher HumanEval.
2. **Try qwen35** (Qwen3.5-27B DeltaNet, MMLU 90 / HE 100). Tests
   whether static-quality leaders transfer to agentic.
3. **Try ream/reap variants from the 3090 stack** if portable. The
   3090's `coder-30b-eval` (AWQ_Marlin) reportedly scored 40.3% on
   the opencode bake-off; comparison would isolate whether the gap
   is the DWQ quantization, the AWQ vs DWQ recipe, or something
   architectural to the 4-bit drop.

This run is archived as evidence that prompt engineering alone is
insufficient. Model swap is the next lever.
