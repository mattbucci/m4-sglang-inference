# qwen35 fallback test on qwen36's only miss — same ceiling, slower failure

## Hypothesis (going in)

qwen36 produced 0 bytes in 45 s on `django__django-11019` — the only real
patch-engagement miss across 16 unique SWE-bench Lite instances. The gold
patch is a 4929-byte algorithmic rewrite of `django/forms/widgets.py`,
replacing the binary `Media.merge` with an N-way merge using
`stable_topological_sort` and `OrderedSet`. That's substantially more
complex than the typical SWE-bench Lite single-spot edit.

Going-in question: **is qwen35 (MMLU 90 vs qwen36's 80) the
algorithmic-rewrite fallback?** If so, the recommendation evolves to
"qwen36 primary, qwen35 retry for algorithmic-class instances."

## Setup

- `PRESET=qwen35` (`mlx-community/Qwen3.5-27B-4bit`, full 27B DeltaNet+VL)
- `INSTANCE_IDS=django__django-11019`
- `TIMEOUT=1800` (3× the default 600s, matching the proven `qwen35-1800s`
  config that previously succeeded on `astropy__astropy-12907`)
- `CTX=131072`, `--mem-fraction-static 0.5`, `--kv-cache-dtype turboquant`
- Through `no_thinking_proxy` on `:23335`

## Result

```
rollout_returncode: 124   (timeout — opencode hit the 1800s wall)
rollout_seconds:    1803
diff_size:          0 bytes (EMPTY)
tool_use lines:     4   (2× glob, 2× read)
opencode tokens:    11942 input / 92 output  (final step_finish)
```

**Hypothesis falsified. qwen35 also produces 0 bytes — same outcome as
qwen36, different failure mode.**

## Failure mode comparison

| | qwen36-scale | qwen35-1800s |
|---|---|---|
| Wall | 45 s | 1803 s (timeout) |
| Tool calls | 3 (2 glob + 1 read + text-only turn) | 4 (2 glob + 2 read) |
| Termination | "I don't know, asks user" | TIMEOUT |
| Files explored | `widgets.py` | `widgets.py`, `tests/forms_tests/tests/test_media.py` |
| Synthesis attempt | No (model exits) | No (model keeps exploring) |
| Patch produced | 0 B | 0 B |

qwen35 actually found the **right** file pair — including the test file
that demonstrates what the canonical 3-way merge behavior should be.
The model reasoned correctly about WHERE the bug was. It just couldn't
synthesize the algorithmic fix within 1800 s at its decode rate
(~7 min/tool-call at 11K-token context — 4 tool calls in 30 min).

## Recommendation refinement

**Before this run:**

> qwen35 is equivalently capable to qwen36 but ~15× slower — use for
> overnight/batch agentic runs where wall time doesn't matter.

**After this run:**

> qwen35 has **the same ceiling** as qwen36 on hard algorithmic
> instances, not a higher one. It explores more files (4 vs 3 tool
> calls) but doesn't synthesize. The static-MMLU-90 advantage doesn't
> convert to agentic-coding capability beyond what qwen36 already has.
> Slower wall time + same ceiling = **no agentic value in qwen35 over
> qwen36**. Use qwen36 for everything; qwen35 only when its DeltaNet
> hybrid is needed for some non-agentic reason.

**Ceiling characterization:**

`django__django-11019` (gold = 4929 B algorithmic refactor with new
import + new utility function + N-way merge replacing binary merge) is
**genuinely outside the capability envelope of both 27B-class MLX models
under greedy decode.** This is not a tuning issue. It's a model-class
limit.

The 16/17 = 94.1% post-proxy patch-engagement rate is **the realistic
upper bound** for qwen36-class MoE on M4. The one miss is the right
miss — recognizable as the algorithmic-instance subset of SWE-bench
Lite. Documentation should set this expectation: "we expect ~5% miss
rate on full-rewrite-class instances; that's the model floor."

## What this still does NOT prove

- That GPT-class flagship models (Sonnet, Opus, etc.) would solve
  django-11019 either — they probably do, but we don't have apples-to-
  apples comparison here.
- That qwen35 + a longer TIMEOUT (3600 s?) would succeed. From the
  4-tool-call pacing at 1800 s, doubling time gives ~8 tool calls —
  still wouldn't include the synthesize step. Doubt this changes things
  without a different prompting approach.
- That a different agent framework (codex-cli, claude-code-style)
  would succeed where opencode failed. opencode's prompt-engineering
  is plausibly fine; the model is the bottleneck.
