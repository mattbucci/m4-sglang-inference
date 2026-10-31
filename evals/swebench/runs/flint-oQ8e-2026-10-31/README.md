# flint (oQ8e) SWE-bench Lite 3-instance smoke — 0/3, two harness-level causes

First agentic rollout of the `flint` preset through the opencode harness.
All three instances hit the 1800 s timeout with an empty diff. The
plumbing (tool-call parsing, streaming, multi-turn template, 131K launch)
held; the run failed on two things above the serving stack.

## Config

- **Server:** `mlx-community/flint-oQ8e` served as `flint` on port 23334,
  `CTX=131072 SGLANG_MLX_CACHE_LIMIT_GB=2 EXTRA_ARGS="--chunked-prefill-size 1024"
  bash scripts/launch.sh flint` (preset adds `--reasoning-parser flint
  --tool-call-parser flint --disable-radix-cache --sampling-defaults model`)
- **Sampling:** model defaults (temperature 1.0, top-p 0.95, top-k 20)
- **Agent:** opencode 1.15.4 via `evals/swebench/run_rollouts.py` through
  `no_thinking_proxy.py` (inert for Flint: its template has no thinking toggle)
- **Instances:** first three of SWE-bench Lite (`astropy__astropy-12907`,
  `-14182`, `-14365`), no venv (install failed, read-edit prompt)
- **Per-instance timeout:** 1800 s
- **OOM guard:** running, zero warnings (`../flint-oQ8e-2026-10-31.oom_guard.log`)

## Result

| # | Instance | Wall | Diff | Steps | bash | read | grep | edit | Max input |
|---|----------|-----:|-----:|------:|-----:|-----:|-----:|-----:|----------:|
| 1 | `astropy__astropy-12907` | 1807 s | 0 B | 67 | 65 (all rejected) | 2 | 1 | 0 | 21.0K tok |
| 2 | `astropy__astropy-14182` | 1808 s | 0 B | 36 | 0 | 39 | 11 | 0 | 39.9K tok |
| 3 | `astropy__astropy-14365` | 1809 s | 0 B | 80 | 80 (all rejected) | 0 | 0 | 0 | 18.9K tok |

qwen36 produced a patch on each of these three instances in ~120 s
(`../qwen36-3instance-2026-05-18/`).

## Cause 1: every `bash` call was rejected by opencode

opencode's `bash` tool schema has two required arguments, `command` and
`description` (a 5-10 word summary). Flint sent `command` only, every time:

```
The bash tool was called with invalid arguments: SchemaError(Missing key
  at ["description"]).
Please rewrite the input so it satisfies the expected schema.
```

The model never adapted to that error; it re-sent the same call (65 and 80
times) until the timeout. `read`, `grep` and `glob` calls (schemas with one
required key) were parsed and executed normally, so this is not a parser
defect — the raw `<tool_call>` markup matches what the server returns.

Replaying opencode's captured request (its 12.7K-token system prompt and
ten tool schemas; `capture_only_proxy` + `replay_request.py` in the session
scratchpad) against the server:

| Prompt | Sampling | bash calls with `description` |
|--------|----------|------------------------------:|
| opencode's, unchanged | greedy | 1/1 (streaming and non-streaming) |
| opencode's, unchanged | model defaults | 2/10 |
| minimal 4-tool list, no system prompt | model defaults | 5/5 |
| opencode's + one rule line in the system prompt | model defaults | 10/10 |
| opencode's + the same line at the end of the user prompt | model defaults | 10/10 |

Flint's own evaluation harnesses (mini-swe-agent, Terminus 2) expose a
single-argument bash tool, and under sampling it drops the unfamiliar
required key; once one omission is in the conversation history the
template replays it and the model copies its own prior call. The rule
line used for the re-run (`PROMPT_EXTRA` in `smoke.sh`):

> Tool-call rule: the `bash` tool has two REQUIRED arguments, `command`
> and `description` (5-10 words saying what the command does). Every bash
> call must include both; a call without `description` is rejected.

For interactive use put the same line in the project's `AGENTS.md`.

## Cause 2: 50-70 s per agent turn

Instance 2 never touched `bash`; it read and grepped for 36 turns and ended
re-reading the same 25-line window of `astropy/io/ascii/rst.py` eight times.
Each turn took 52 s median (69 s max) at 20-40K tokens of context, because
the Flint preset runs with the radix cache off: upstream only pools
full-attention KV, and on a sliding-window model it recomputes the whole
prefix, so every opencode turn re-prefills its entire context at the
measured 630-930 tok/s. Thirty-six turns fill the 1800 s budget before the
agent gets to an edit. This is experiments row 6 (`flint-prefix-cache`).

## Files

- `predictions.jsonl`, `predictions/<inst>.diff` (all 0 B)
- `logs/<inst>.log` — opencode JSON event stream (tool-call breakdown above)
- `logs/<inst>.env.log` — venv install attempts (all failed; no-venv prompt used)
- `launch.log`, `proxy.log`, `rollout.log`
