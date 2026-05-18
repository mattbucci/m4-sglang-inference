# qwen36 + no_thinking_proxy — **first M4 SWE-bench patch**

## Setup

- Server: `mlx-community/Qwen3.6-35B-A3B-4bit` (preset `qwen36`) at port 23334
- Flags: `--reasoning-parser qwen3 --enable-multimodal --disable-radix-cache --tool-call-parser qwen3_coder --kv-cache-dtype turboquant --chunked-prefill-size 2048 --mem-fraction-static 0.5`
- Context: 32K
- **Proxy:** `evals/swebench/no_thinking_proxy.py` on port 23335, injecting `chat_template_kwargs={"enable_thinking": false}` on every chat-completion POST
- opencode config repointed at `http://127.0.0.1:23335/v1` for this run
- Instance: `astropy__astropy-12907` (1 instance, no venv)

## Result

```
rollout_seconds:      125.1s
rollout_returncode:   0
diff_size:            506 bytes (NON-EMPTY — first M4 success)
tool_calls:           1×edit, 3×read, 2×glob   (6 total)
text turns w/ content: 4
```

The patch:

```diff
diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py
@@ -242,7 +242,7 @@ def _cstack(left, right):
         cright = _coord_matrix(right, 'right', noutp)
     else:
         cright = np.zeros((noutp, right.shape[1]))
-        cright[-right.shape[0]:, -right.shape[1]:] = 1
+        cright[-right.shape[0]:, -right.shape[1]:] = right

     return np.hstack([cleft, cright])
```

The instance is the well-known `separability_matrix` nested-CompoundModel
bug. The model identified the broken assignment in `_cstack` (assigning
`1` instead of the actual right-side matrix) and replaced it. Whether
this passes SWE-bench's official Docker harness check is TBD (we don't
run Docker on M4); the change is in the right area and looks plausible.

## Comparison vs prior qwen36 attempts (same instance, same model)

| Variant | Wall | tool_calls | edit calls | diff |
|---------|:----:|:----------:|:----------:|:----:|
| `qwen36` + `--reasoning-parser qwen3` (direct) | 350.6s | 9 (5R+2G+2g) | 0 | **0 B** |
| `qwen36` no parser (REASONING_OFF=1) | 71.7s | 4 (2R+2g) | 0 | **0 B** |
| **`qwen36` + parser + no_thinking_proxy** | **125.1s** | **6 (3R+2g+1edit)** | **1** | **506 B** |

The proxy converted "model thinks but agent can't see" into "model
acts and agent sees actions". Tool-call count dropped (6 vs 9) but
intent shifted from "explore exhaustively" to "explore, decide,
edit".

## Mechanism

opencode's `@ai-sdk/openai-compatible` provider:
- Doesn't consume `choices[0].message.reasoning_content` from SGLang
  responses (only standard `content`).
- Doesn't pass `chat_template_kwargs` body extensions (no API for it).

Qwen3 chat template:
- Default: emits `<think>...</think>` blocks where the model reasons.
- Under `--reasoning-parser qwen3`: that thinking is parsed into the
  separate `reasoning_content` field (invisible to opencode).
- Under no parser: tags leak into content (confuses opencode parser).
- Under `chat_template_kwargs={"enable_thinking": false}`: NO `<think>`
  block is added to the prompt at all; the model goes straight to
  acting on the content channel.

The proxy injects that third option on every chat-completion POST.

## Operational note

`smoke.sh` now starts the proxy automatically (default
`NO_THINKING_PROXY=1`). Set `NO_THINKING_PROXY=0` to disable.

When running opencode outside of `smoke.sh` (interactive
sessions, agentic coding), the user can either:
- Start the proxy manually and edit
  `~/.config/opencode/opencode.jsonc` to point at port 23335, or
- Set `chat_template_kwargs={"enable_thinking": false}` in the
  Python harness that drives requests directly (no proxy needed).

## Strategic implication

The README's "Recommended pick: long-context flagship = qwen36" is
**now backed by an actual agentic-coding success** — not just static
benchmarks. qwen36 + no_thinking_proxy is the first M4 configuration
that completes the SWE-bench agentic loop end-to-end with a
non-empty patch.

Next: confirm reproducibility on 2-3 more instances, then run the
4-pick scorecard (task #44) to compare against coder-30b, qwen35,
gemma4-31b under the same proxy config.
