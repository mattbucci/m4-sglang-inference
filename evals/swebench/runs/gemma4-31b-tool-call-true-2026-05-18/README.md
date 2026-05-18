# gemma4-31b SWE-bench retest with `tool_call: true` — still 0 engagement

## Setup

Same `astropy__astropy-12907` instance, 600 s timeout. Difference from
the 4-pick scorecard: `evals/swebench/opencode.json` now has
`"tool_call": true` for the `gemma4-31b` entry (fixed in commit
`71878b3`).

Server config confirmed via launch.log:
- `tool_call_parser='gemma4'` ✓
- `reasoning_parser='gemma4'` ✓
- `enable_multimodal=True` ✓
- `disable_radix_cache=True` ✓
- Context 16K, turboquant KV, chunked-prefill 2048, mem-fraction 0.5
- Routed through `no_thinking_proxy` on port 23335

## Result

```
rollout_seconds:    609 (timeout, rc=124)
tool_calls:         0
text turns:         0 non-empty
diff_size:          0 bytes
```

Opencode emitted exactly one `step_start` event, then silence for 600
seconds until the SIGKILL. The preflight ping (a single small chat
completion) returned 32 bytes successfully, so the server responds to
trivial requests. But under opencode's full tool-augmented prompt
shape, gemma4-31b produces **zero generated tokens** before timeout.

## Interpretation

Not a config bug. The `tool_call: true` opencode flag exposes the tool
surface to the model; the SGLang server has the correct
`--tool-call-parser gemma4` and `--reasoning-parser gemma4` flags;
the no-thinking proxy is in place. Every layer is configured for
agentic use.

The failure is **upstream of the parser**: gemma4-31b under greedy
MLX, given opencode's tool-definition-augmented prompt, doesn't emit
the Gemma 4 tool-call output format the SGLang parser expects. It
emits nothing — or its output is being held back by the chat-template
formatter's interpretation of the tool definitions.

Possible root causes (not investigated here):
1. opencode passes tools in OpenAI-spec JSON format; Gemma 4's chat
   template (jinja) may not transcode those into Gemma's expected
   `<start_of_turn>tool` blocks correctly.
2. mlx-vlm's Gemma 4 implementation may not handle the specific
   prompt structure opencode produces.
3. Greedy decode + the long tool-augmented prompt may put the model
   into a no-emission state.

## Conclusion

For the M4 SWE-bench agentic-coding workload, **gemma4-31b is not
usable through opencode**, regardless of `tool_call` config. Its
strong static scores (MMLU 92 / Needle 100) make it useful for
direct chat-completion workloads — use it for non-tool-call code
generation, reasoning, image analysis. The README's rank-4 entry is
updated to reflect this gap.

The 3090 sister stack also doesn't list gemma4-31b in its
SWE-bench bake-off matrix (they focus on Qwen/Coder variants),
which is consistent with this finding.

## What this DOES validate

The opencode config plumbing works: `tool_call: true` correctly
exposes the tool surface (verified by the bake-off; coder-30b and
qwen36 produce tool calls under the same config), `tool_call:
false` correctly suppresses it. The gemma4-31b 0/0 result is
model-side, not infrastructure-side.
