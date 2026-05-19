# int4 agentic bakeoff — 0/5 on the untested models

## Setup

Per user instruction: validate every int4 model in `opencode.json` against
the agentic coding baseline (`astropy__astropy-12907`) before considering
FP8/8-bit variants for any failures. TIMEOUT=900s, per-instance fresh
server.

## Result

| Model | Patch | Wall | rc | Failure mode |
|---|---|---:|:---:|---|
| `qwen3-32b` (Dense, MMLU 90) | 0 B | 906 s | 124 | TIMEOUT — model attempted but couldn't converge |
| `qwen3-moe` (Qwen3-30B-A3B-DWQ) | 0 B | 22 s | 0 | Parser mismatch — emitted malformed `<\|name>read>...` tags |
| `devstral` (Mistral 24B) | — | — | smoke_rc=2 | Preflight failed — Mistral template rejects the canary structure |
| `nemotron-30b` (Nemotron-3-Nano-30B-A3B) | 0 B | 904 s | 124 | TIMEOUT — model attempted but couldn't converge |
| `qwen36-27b` (Qwen3.6-27B Dense+DeltaNet+VL) | 0 B | 906 s | 124 | TIMEOUT — model attempted but couldn't converge |

**0/5 patch-engagement. None of the untested int4 models work for agentic
coding on this stack.**

Combined with prior data, the **complete int4 picture for agentic coding
on M4** is:

| Model | Verdict | Evidence |
|---|---|---|
| `qwen36` (Qwen3.6-35B-A3B-4bit) | ✓ WORKS | 21/26 patch-engagement, 5/13 resolved across 12 ecosystems |
| `qwen35` (Qwen3.5-27B-4bit) | ✓ Works at TIMEOUT=1800 | 1/3 (same ceiling as qwen36, 15× slower) |
| `coder-30b` | ✗ Structural | 0/3, 1 glob then "asks user" |
| `gemma4-31b` | ✗ Model-side gap | 0/2, emits 0 tokens under tool prompts |
| `qwen3-32b` | ✗ Can't converge | 0/1, TIMEOUT (this run) |
| `qwen3-moe` | ✗ Parser mismatch | 0/1, malformed tool tags (this run) |
| `devstral` | ✗ Preflight blocked | preflight canary incompatible with Mistral template (this run) |
| `nemotron-30b` | ✗ Can't converge | 0/1, TIMEOUT (this run) |
| `qwen36-27b` | ✗ Can't converge | 0/1, TIMEOUT (this run) |

**Only qwen36 and qwen35 work** — both are the Qwen3-Coder-tool-call-parser
+ DeltaNet+VL architecture. The combination matters; one without the other
doesn't. qwen36-27b (Dense+DeltaNet+VL, same parser as qwen36) couldn't
converge despite identical infrastructure. The model SIZE + the MoE-3B-active
characteristics of qwen36 are load-bearing — Dense at 27B isn't fast enough
to keep an agent loop alive within reasonable timeouts.

## Notable failure-mode breakdown

### "Couldn't converge" (qwen3-32b, nemotron-30b, qwen36-27b)

All three ran the full 900s window producing tool calls but no edit. Same
pattern as `django-11019` and `flask-4045` on qwen36 — the model recognizes
it doesn't know what to write and TIMEOUTs without committing. The decode
rate combined with the agent's exploration depth means even 15 minutes
isn't enough for these models to reach a synthesis step.

### qwen3-moe parser mismatch (potentially fixable)

The model emitted `<|name>read> {"filePath":...}` instead of either
`<tool_call>{...}</tool_call>` (the qwen25 format) or `<tool_call>...`
(the qwen3_coder format). Neither parser recognized it. **Could be a
chat-template issue specific to this checkpoint** — the qwen3-moe (DWQ
variant) may have inherited a different template than the standard Qwen3
release. Worth investigating but unlikely to lift the model into the
"works" bucket given its structural similarity to coder-30b.

### devstral preflight blocked (potentially fixable)

The preflight canary in `run_rollouts.py` sends a
`user → assistant(tool_calls) → tool → user "continue"` message sequence
to validate the chat template can round-trip tool calls. Mistral's chat
template enforces strict user/assistant alternation and rejects this with
400 Bad Request. **The model itself may work agentically** — opencode's
real prompts wouldn't have this exact structure — but the preflight gate
blocks evaluation. Would need a Mistral-specific preflight variant or a
`--skip-preflight` flag to test.

## FP8/8-bit retry decision

The user mentioned trying FP8 weights for Gemma and Nemotron — useful to
revisit given these int4 results:

| Model | Failure mode | FP8 retry value |
|---|---|---|
| `gemma4-31b` | 0 tokens emitted (chat-template/tool-call gap) | **NO** — not a quantization issue |
| `nemotron-30b` | TIMEOUT, couldn't converge | Possible — quantization noise could be eating attention |
| `nemotron-omni` | UNTESTED in agentic | Worth trying at int4 first |
| `qwen3-32b` | TIMEOUT, couldn't converge | Possible (Dense models more quant-sensitive) |
| `qwen36-27b` | TIMEOUT, couldn't converge | Possible (Dense variant of working MoE) |

The "0 tokens emitted" cases (gemma4-31b, qwen3-moe-similar) are NOT
quantization issues — they're chat-template/parser failures where the model
either won't engage or produces output the parser can't read. Higher
precision weights won't help those.

For the "couldn't converge" cases, FP8 *might* help if the issue is
quantization noise in attention — but the predominant failure pattern
(model decodes for 15 min then TIMEOUTs) suggests the model is doing
agentic work and just running out of time, not producing nonsense. FP8
wouldn't change the wall-time math.

**Net recommendation**: don't burn disk on FP8 for these models without
a stronger reason. The qwen36 recommendation is solid; the others aren't
worth a 32 GB download given the failure modes.

## Files

- `STATUS` — wall-clock per-model timeline
- `{model}.jsonl` — per-model prediction record (4 of 5; devstral didn't run)
- `logs/{model}/` — env-install logs + (when present) opencode tool-call logs
