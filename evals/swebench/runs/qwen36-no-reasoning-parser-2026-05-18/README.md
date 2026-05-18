# qwen36 (no --reasoning-parser, REASONING_OFF=1) — 4 tool calls, empty diff

## Setup

- Server: same as `qwen36-thinking-2026-05-18/` BUT with `--reasoning-parser qwen3` stripped via `REASONING_OFF=1`.
- Other flags identical: `--enable-multimodal --disable-radix-cache --tool-call-parser qwen3_coder --kv-cache-dtype turboquant --chunked-prefill-size 2048 --mem-fraction-static 0.5`.
- Context: 32K
- Instance: `astropy__astropy-12907` (1 instance, no venv)

## Result

```
rollout_seconds:      71.7s        (vs 350.6s with parser)
rollout_returncode:   0
diff_size:            0 bytes
tool_calls:           2×glob, 2×read   (4 total, vs 9 with parser)
text turns:           2 non-empty
```

## What happened

Without the reasoning parser, the model's content channel includes
**partial `</think>` markers** as raw text:

> Turn 1: "Let me analyze this issue. The problem is with
> `separability_matrix`... Let me start by exploring the codebase to
> understand the structure and find the relevant code.\n</think>"
>
> Turn 2: "Let me read the separable.py file to understand the
> implementation.\n</think>"

The Qwen3 chat template still emits `<think>...</think>` tags around
reasoning — without the SGLang parser those tags pass through to
opencode's content stream. The leaking `</think>` confuses opencode's
parser and the agent loop terminates after 4 tool calls (significantly
fewer than with the parser).

Empty diff, no `edit` call.

## Comparison

| Variant | Wall | tool_calls | text content | diff |
|---------|:----:|:----------:|:------------:|:----:|
| `qwen36` + parser | 350.6s | 9 (5R+2G+2g) | whitespace-only | 0 B |
| `qwen36` no parser | 71.7s | 4 (2R+2g) | leaked `</think>` markers | 0 B |

Both fail. The parser variant explores more but its content is
invisible to opencode; the no-parser variant has visible content but
the chat template's `<think>` tags break the agent loop.

## Conclusion

The qwen36 + opencode + greedy-MLX combination doesn't complete the
agentic coding loop on either path:
- WITH `--reasoning-parser qwen3`: model thinks well, opencode can't
  see the thinking; loop hits a terminator before reaching `edit`.
- WITHOUT parser: chat template's literal `</think>` markers leak
  into content; opencode's tool-call extraction works for `glob`/`read`
  but the agent loop terminates early.

Possible fixes (queued):
1. Pass `chat_template_kwargs={"enable_thinking": false}` through
   opencode's openai-compatible provider — keeps the parser on but
   prevents the chat template from adding `<think>` blocks.
2. Patch opencode's openai-compatible provider to map
   `reasoning_content` → content (would require a plugin or upstream
   PR).
3. Test qwen35 next — DeltaNet hybrid, may have different agent
   behavior under the same parser settings.
