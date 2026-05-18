# SWE-bench Lite agentic coding on M4

Tool-call-driven coding evaluation: feed the model a GitHub issue + repo,
let it use `read`/`glob`/`grep`/`edit`/`write` tools to produce a patch,
score the patch against the official SWE-bench test harness.

This is the **load-bearing benchmark for the M4 stack's "long-context
agentic coding" goal.** Static MMLU / HumanEval / Needle numbers in the
README's quality table don't predict agentic capability on this hardware
— SWE-bench Lite does.

## Stack

```
SWE-bench Lite (princeton-nlp/SWE-bench_Lite)
  └── run_rollouts.py (HF dataset loader + git clone + opencode driver)
      └── opencode 1.15.4
          └── HTTP → no_thinking_proxy.py :23335
              └── HTTP → SGLang :23334
                  └── MLX backend, native Apple Silicon
```

**`no_thinking_proxy.py`** is the load-bearing piece. opencode's
`@ai-sdk/openai-compatible` provider doesn't pass
`chat_template_kwargs` to SGLang. Qwen3-family chat templates default
to emitting `<think>...</think>` blocks; those either go to
`reasoning_content` (invisible to opencode) or leak `</think>` markers
into the content stream (breaks the tool-call parser). The proxy
injects `chat_template_kwargs={"enable_thinking": false}` on every
chat-completion POST so the model emits actionable content directly.

Verified 2026-05-18: without the proxy, qwen36 makes 9 tool calls but
produces a 0-byte diff (reasoning is invisible). With the proxy, qwen36
makes 6 tool calls including 1 `edit` and produces a 506-byte patch
that targets the canonical bug location.

## Current leaderboard (auto-generated)

Run `python evals/swebench/aggregate.py --markdown` to refresh:

| Model | Rollouts | Patches | Rate | Mean wall | Ecosystems |
|-------|:--------:|:-------:|:----:|:---------:|------------|
| `qwen36` | 21 | 18 | **18/21** | 288 s | astropy, django, matplotlib, **pylint, scikit-learn, sphinx**, sympy |
| `qwen35` | 2 | 1 | **1/2** | 1206 s | astropy |
| `coder-30b` | 3 | 0 | **0/3** | 34 s | astropy |
| `gemma4-31b` | 2 | 0 | **0/2** | 606 s | astropy |

**Real rate (post-proxy, excluding pre-proxy diagnostics): qwen36 is 16/17 = 94%.**
The 3 non-patch qwen36 rollouts breakdown:

- 2 pre-proxy diagnostics (`qwen36-thinking`, `qwen36-no-reasoning-parser`) —
  not failures of the model, failures of the integration. Closed by adding the
  proxy.
- 1 actual rollout failure (`qwen36-scale/django__django-11019`) — model
  produced 0 bytes in 45 s. The rest of `qwen36-scale` was 6/7 with patches up
  to 1714 B.

**qwen35 1/2** is the new finding from `qwen35-1800s-2026-05-18`: at the default
600 s timeout qwen35 doesn't finish (it makes tool calls but can't close the
loop in time at its ~15× slower decode rate). At `TIMEOUT=1800`, qwen35 produces
the **same 506-byte patch as qwen36 on astropy__astropy-12907 in 1810 s**.
Capability-equivalent, latency-prohibitive at default settings. Use qwen36 for
agentic coding; qwen35 only when its DeltaNet hybrid is required for some other
reason and you can afford the wall.

**coder-30b 0/3** is the structural failure: even with the stronger prompt
(`coder-30b-stronger-prompt-2026-05-18`), coder-30b emits 0 tool calls and 0
bytes. Coder-30B's chat template doesn't engage the agentic loop. Not fixable
upstream of the model.

**gemma4-31b 0/2** is the model-side gap: even with `tool_call: true` in
opencode.json, gemma4-31b emits 0 tokens under the tool-prompts opencode
generates. Distinct from the no-think problem — this is the model itself.

**Cross-ecosystem coverage (qwen36, post-proxy):** astropy (8 ✓), django (4 ✓ /
1 ✗), matplotlib (1 ✓), pylint (1 ✓), scikit-learn (1 ✓), sphinx (1 ✓), sympy
(1 ✓) — **7 ecosystems, 17/18 patch-engagement rate.**

## Quickstart

```bash
# 1. Run a single instance (defaults: qwen36, instance 1, proxy on)
bash evals/swebench/smoke.sh

# 2. Run a specific instance
INSTANCE_IDS="django__django-10914" bash evals/swebench/smoke.sh

# 3. Run the first N instances
INSTANCES=5 bash evals/swebench/smoke.sh

# 4. Compare across the 4 README picks
bash evals/swebench/bakeoff.sh

# 5. Refresh the leaderboard from all archived runs
python evals/swebench/aggregate.py --markdown
```

## Workflow

1. `smoke.sh` orchestrates: stops prior server, launches SGLang via
   `scripts/launch.sh`, waits for `/health`, starts `no_thinking_proxy`,
   re-points `~/.config/opencode/opencode.jsonc` at port 23335.
2. `run_rollouts.py` loads SWE-bench Lite from HF, clones each repo at
   the issue's base commit, runs `opencode run --model sglang/<key>`
   with the problem statement + tool-call directive prompt.
3. `eval_env.py` tries to build a per-instance venv from the repo's
   pinned deps so the model can `pytest`/`import` during the rollout.
   Often fails on M4 (we don't have system Python headers for old
   packages); falls back to a "no-venv" prompt mode where the model
   reasons about correctness from source alone.
4. Final `git diff` from the worktree is captured as the prediction.
5. `smoke.sh` tears down proxy + restores opencode config + kills
   server.

## Layout

```
evals/swebench/
├── README.md                       # this file
├── opencode.json                   # M4 provider config (sglang/* keys → 23334)
├── no_thinking_proxy.py            # aiohttp+httpx proxy on 23335 (load-bearing)
├── run_rollouts.py                 # opencode driver + SWE-bench loader
├── eval_env.py                     # per-instance venv builder
├── score_local.py                  # local-venv scoring (when Docker unavailable)
├── smoke.sh                        # single-preset orchestrator
├── bakeoff.sh                      # multi-preset comparison
├── aggregate.py                    # leaderboard aggregator
└── runs/                           # archived rollout artifacts
    └── <model>-<tag>-<date>/
        ├── README.md               # per-run interpretation
        ├── predictions.jsonl
        └── predictions/<inst>.diff
```

## Validation gap

The patches **look** plausible (target canonical issue locations, hit
the upstream-fix lines, multi-file when needed). We can NOT yet
confirm they pass SWE-bench's Docker test harness because we don't run
Docker on M4. The strict validation path is to push our
`predictions.jsonl` to the 3090 stack's `evals/swebench/score_docker.py`
and run the official harness there. Until that lands the leaderboard
above measures **patch-engagement rate**, not pass-rate.

## Related

- `feedback_m4_loop_goal.md` (auto-memory) — the agentic-coding goal
- `project_eval_jetsam_artifact.md` (auto-memory) — eval-harness
  robustness pattern that informed the proxy work
- 3090 stack: `~/AI/2x-3090-GA102-300-A1-sglang-inference/evals/swebench/`
  — the original opencode harness ported here
