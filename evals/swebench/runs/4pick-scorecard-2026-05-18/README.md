# SWE-bench Lite 4-pick agentic-coding scorecard (2026-05-18)

Same instance (`astropy__astropy-12907`), same proxy (`no_thinking_proxy`
injecting `chat_template_kwargs={"enable_thinking": false}`), four
README "Recommended picks for long-context agentic coding".

## Result

| Preset | Wall | Tool calls (kinds) | edit? | Diff bytes | Verdict |
|--------|:----:|:-------------------:|:-----:|:----------:|:-------:|
| `coder-30b` | 35 s | 1 (glob) | 0 | 0 | **GIVES UP** |
| **`qwen36`** | **125 s** | **6 (1 edit + 3 read + 2 glob)** | **1** | **506** | **WINNER — real patch** |
| `qwen35` | 603 s (TIMEOUT) | 6 (2 read + 3 glob + 1 grep) | 0 | 0 | **RIGHT BUG, TOO SLOW** |
| `gemma4-31b` | 603 s (TIMEOUT) | 0 | 0 | 0 | **NEEDS opencode config** |

## Detail

### `coder-30b` (Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ)

35-second rollout, one `glob`, then a text turn that asks the user for
the file contents (despite the no-thinking proxy stripping
`<think>` blocks). Structurally broken in the opencode agent loop
under greedy MLX decode. The proxy fixed this for qwen36 but did
NOT fix coder-30b. Model-level limitation.

Recommendation: use coder-30b for direct chat-completion code
generation (where its HumanEval 95 / MMLU 89.5 still shine), **NOT
for tool-call-driven agentic flows**.

### `qwen36` (Qwen3.6-35B-A3B-4bit MoE+DeltaNet)

The clear winner. 125-second rollout, 6 tool calls including 1
`edit`, produces a real 506-byte patch identifying the correct bug
in `_cstack`:

```diff
-        cright[-right.shape[0]:, -right.shape[1]:] = 1
+        cright[-right.shape[0]:, -right.shape[1]:] = right
```

The proxy unlocks this. Without the proxy, qwen36's reasoning goes
to the `reasoning_content` channel that opencode's openai-compatible
provider doesn't surface.

### `qwen35` (Qwen3.5-27B-4bit DeltaNet)

Made 6 tool calls and in its final text turn identified the exact
same line qwen36 patched ("Let me look at the `_cstack` function
more carefully. The issue is on line 245: `cright[-right.shape[0]:,
-right.shape[1]:] = 1`"). But the DeltaNet 27B dense decode is
slower than qwen36's MoE — even after 600 s the model hadn't gotten
to the `edit` call.

This is actionable: qwen35 IS capable of solving these instances,
just needs longer wall budget. Run with `TIMEOUT=1800` or pick the
faster MoE variant.

### `gemma4-31b` (gemma-4-31b-it-mxfp4)

0 tool calls, 0 text content, 603-second timeout. The model is
not engaging with opencode's tool-call protocol at all. Root cause:
`evals/swebench/opencode.json` has `"tool_call": false` for the
gemma4-31b model entry — opencode doesn't even expose the tool
surface to it. Fix: flip to `true` and re-test. Gemma 4 31B
architecturally supports function calling (the `--tool-call-parser
gemma4` flag exists upstream); whether mlx-vlm's quantization
preserves that capability is the open question.

## Recommendation update

Based on this scorecard, the README's "Recommended picks for
long-context agentic coding" should be reordered:

| Rank | Preset | Notes |
|------|--------|-------|
| 1 | **qwen36** (Qwen3.6-35B-A3B MoE+DeltaNet) | Only configuration verified to produce real SWE-bench patches. Use with `no_thinking_proxy`. |
| 2 | qwen35 (Qwen3.5-27B DeltaNet) | Capable but slow. Higher TIMEOUT or smaller instance gets results. |
| 3 | coder-30b (DWQ) | Best static HumanEval, but agent loop broken. Direct chat-completion code-gen only. |
| 4 | gemma4-31b | opencode config gap — flip `tool_call: true` and re-measure. |

This is the data the README needed: agentic coding pick is **qwen36**,
not coder-30b. Higher static HE scores don't predict agentic capability
on the M4 stack.

## Methodology notes

- `instances`: 1 each (`astropy__astropy-12907`), comparable rollouts
- `timeout`: 600 s
- All servers freshly launched per smoke (no chained state)
- All routed through `no_thinking_proxy` on port 23335
- `EXTRA_LAUNCH`: `--kv-cache-dtype turboquant --chunked-prefill-size 2048 --mem-fraction-static 0.5` plus model-specific `--enable-multimodal --tool-call-parser qwen3_coder --disable-radix-cache` for Qwen3.x family

Raw artifacts (one .jsonl per preset) in this directory. Full opencode
logs in `/tmp/swebench-*/logs/`.
