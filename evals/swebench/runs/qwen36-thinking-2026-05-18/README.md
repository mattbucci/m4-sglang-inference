# qwen36 (with --reasoning-parser qwen3) — 9 tool calls, empty diff

## Setup

- Server: `mlx-community/Qwen3.6-35B-A3B-4bit` at port 23334 (preset `qwen36`)
- Flags: `--reasoning-parser qwen3 --enable-multimodal --disable-radix-cache --tool-call-parser qwen3_coder --kv-cache-dtype turboquant --chunked-prefill-size 2048 --mem-fraction-static 0.5`
- Context: 32K
- Instance: `astropy__astropy-12907` (1 instance, no venv)

## Result

```
rollout_seconds:      350.6s
rollout_returncode:   0
diff_size:            0 bytes
tool_calls:           5×read, 2×grep, 2×glob   (9 total)
text turns:           8 (all whitespace-only!)
step_finish reasons:  8× "tool-calls" (i.e. continued iterating)
```

## What happened

qwen36 iterated through 9 tool calls — actually exploring the repo,
reading multiple files, grepping for `separability_matrix`. Big
contrast with coder-30b which gave up after 1 glob.

**But all 8 `text` events from opencode contained ONLY whitespace
(`"\n\n\n"`).** Confirmed via JSON inspection — the `part.text` field
is always blank-or-newlines.

That's because `--reasoning-parser qwen3` strips the model's `<think>`
content into a separate `reasoning_content` field that opencode's
openai-compatible provider doesn't surface as agent-visible text. The
model is reasoning extensively (350s wall, ~10× longer than coder-30b)
but its reasoning never reaches opencode's agent loop. opencode sees
the tool calls but no text-channel content driving toward `edit`.

After 9 tool calls the agent loop hit some terminator (probably max
iterations or no-new-content state) and exited. No `edit` or `write`
tool call. Empty diff.

## Diagnosis

Plumbing-side: opencode's openai-compatible provider doesn't read
`choices[0].message.reasoning_content` from the SGLang response. The
ai-sdk OpenAI-compatible client only consumes the standard `content`
channel. Anything in `reasoning_content` is invisible.

Model-side: qwen36 under greedy MLX + Qwen3 chat template + the
`reasoning_parser` puts most of its planning into `reasoning_content`
and leaves `content` empty between tool calls. Without the parser,
content keeps the `<think>...</think>` literals (see the
`qwen36-no-reasoning-parser-2026-05-18/` companion run for that
variant).

**Neither variant produces an edit.** The agentic loop fundamentally
isn't completing on this setup.
