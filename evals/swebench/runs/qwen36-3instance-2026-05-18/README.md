# qwen36 reproducibility: 3/3 on SWE-bench Lite instances 1-3

## TL;DR

qwen36 (Qwen3.6-35B-A3B-4bit MoE+DeltaNet) + `no_thinking_proxy` produced
non-empty patches on **3 of the first 3 SWE-bench Lite instances**, all in
~120 s per rollout, all targeting different files in different modules.
The 2026-05-18 README claim ("qwen36 is the agentic-coding lead") is not
a 1-shot fluke.

## Result

| # | Instance | Wall | Diff bytes | Target file | Change |
|---|----------|:----:|:----------:|-------------|--------|
| 1 | `astropy__astropy-12907` | 122 s | 506 | `astropy/modeling/separable.py` | `_cstack`: `= 1` → `= right` |
| 2 | `astropy__astropy-14182` | 132 s | 579 | `astropy/io/ascii/rst.py` | `RST.__init__`: add `header_rows` param |
| 3 | `astropy__astropy-14365` | 116 s | 476 | `astropy/io/ascii/qdp.py` | regex: add `(?i)` case-insensitive flag |

3/3 produce diffs. Each diff is in a different file. Each is a 1-3 line
edit. Total run time 6 m 12 s for three rollouts including setup +
teardown.

## Methodology

- Server: `mlx-community/Qwen3.6-35B-A3B-4bit` at 32K context, turboquant
  KV, chunked-prefill 2048, mem-fraction 0.5, `--disable-radix-cache
  --enable-multimodal --tool-call-parser qwen3_coder`
- Proxy: `evals/swebench/no_thinking_proxy.py` on port 23335, injecting
  `chat_template_kwargs={"enable_thinking": false}` on every chat-completion POST
- opencode 1.15.4 with `~/.config/opencode/opencode.jsonc` pointing at proxy
- `INSTANCES=3 TIMEOUT=600 bash evals/swebench/smoke.sh` (defaults to qwen36)
- 1 instance ran, env install always failed (we don't ship Python repo
  deps), no-venv prompt used, model worked from source code alone

## Strategic implications

1. **Reproducibility confirmed.** The README's "qwen36 = agentic-coding
   lead" claim survives moving from 1-instance smoke to 3-instance test.
2. **~125 s/instance is the wall-time floor on this stack.** Useful for
   sizing larger SWE-bench Lite runs (full 300 instances ≈ 10.5 hours
   single-user; sub-sampling to 30 instances ≈ 60 min).
3. **No model retreat after 1 success.** Even on instance 3 the model
   makes a confident edit. This is what "agentic-coding lead" should
   look like.

## What this DOES NOT prove

- That SWE-bench's official Docker test harness would actually pass
  these patches. The patches look plausible but we don't run Docker on
  M4. Cross-stack scoring (push predictions to the 3090 stack's
  `score_docker.py`) is the next validation step.
- That the model generalizes off the astropy subdomain. The first 3
  instances are all astropy — easier than Django/sympy/sphinx
  instances later in the dataset.
- That the rate holds at 30 / 100 / 300 instances. 3/3 with these
  3-line patches doesn't prove harder instances also yield patches.

## Next experiments queued

- Run instances 4-10 (mixed astropy + a non-astropy instance like
  `django__django-11099` if it shows up early in the dataset)
- Cross-stack score validation: ship `predictions.jsonl` to the 3090
  stack's Docker scorer
- Compare gemma4-31b now that `tool_call: true` is set in opencode.json
- Compare qwen35 with `TIMEOUT=1800` (it identified instance-1's bug
  correctly but was too slow to reach `edit`)
