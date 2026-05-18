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
| `qwen36` | 10 | 8 | **8/10** | 206 s | astropy, django, matplotlib, sympy |
| `coder-30b` | 3 | 0 | **0/3** | 34 s | astropy |
| `qwen35` | 1 | 0 | **0/1** | 603 s | astropy |
| `gemma4-31b` | 1 | 0 | **0/1** | 603 s | astropy |

The 2 qwen36 non-patches are early diagnostic runs (pre-proxy) — see
`runs/qwen36-thinking-2026-05-18/` and
`runs/qwen36-no-reasoning-parser-2026-05-18/`.

**With proxy, qwen36 is 8/8 across 4 ecosystems.**

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
