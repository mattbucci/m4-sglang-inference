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
| `qwen35` | 3 | 1 | **1/3** | 1404 s | astropy, django |
| `coder-30b` | 3 | 0 | **0/3** | 34 s | astropy |
| `gemma4-31b` | 2 | 0 | **0/2** | 606 s | astropy |

**Real rate (post-proxy, excluding pre-proxy diagnostics): qwen36 is 16/17 = 94%.**
The 3 non-patch qwen36 rollouts breakdown:

- 2 pre-proxy diagnostics (`qwen36-thinking`, `qwen36-no-reasoning-parser`) —
  not failures of the model, failures of the integration. Closed by adding the
  proxy.
- 1 actual rollout failure (`qwen36-scale/django__django-11019`) — model
  produced 0 bytes in 45 s. The rest of `qwen36-scale` was 6/7 with patches up
  to 1714 B. **2026-05-18 follow-up**: re-tried with qwen35 at TIMEOUT=1800
  (`qwen35-django11019-1800s-2026-05-18/`) — same outcome (0 bytes), different
  failure mode (timeout after 1803 s with 4 tool calls). Gold patch is a
  4929-byte algorithmic rewrite of `Media.merge` from binary to N-way using
  topological sort + new `OrderedSet` import. **This instance is out of the
  envelope of both 27B-class MLX models** — not a tuning miss.

**qwen35 1/3** is more nuanced than originally framed:

- `qwen35-1800s-2026-05-18/` (astropy__astropy-12907, TIMEOUT=1800): SUCCESS
  with the same 506-byte patch as qwen36, but at 1810 s wall (15× slower).
- `4pick-scorecard-2026-05-18/` (astropy__astropy-12907, TIMEOUT=600): FAIL
  — model still mid-loop when timeout fired.
- `qwen35-django11019-1800s-2026-05-18/` (django-11019, TIMEOUT=1800): FAIL
  — same algorithmic-class instance qwen36 missed.

**Refined recommendation**: qwen35 is **capability-equivalent to qwen36, not
higher-capability**. The static-MMLU-90 advantage (per the hardened audit;
qwen36 is MMLU 80) does NOT convert to agentic-coding capability beyond what
qwen36 already has. qwen35 = slower wall + same ceiling. Use qwen36 for
agentic coding; qwen35 only when DeltaNet 27B-dense is required for non-
agentic reasons.

**coder-30b 0/3** is the structural failure: even with the stronger prompt
(`coder-30b-stronger-prompt-2026-05-18`), coder-30b emits 0 tool calls and 0
bytes. Coder-30B's chat template doesn't engage the agentic loop. Not fixable
upstream of the model.

**gemma4-31b 0/2** is the model-side gap: even with `tool_call: true` in
opencode.json, gemma4-31b emits 0 tokens under the tool-prompts opencode
generates. Distinct from the no-think problem — this is the model itself.

**Cross-ecosystem coverage (qwen36, post-proxy, unique-instance, on the
hardened harness):**

| Ecosystem | Verified | Notes |
|---|---|---|
| astropy | 6/6 | 12907 (×5 dup runs all hit), 14182, 14365, 14995, 6938, 7746 |
| django | 4/5 | miss = 11019 (algorithmic rewrite, see ceiling note) |
| matplotlib | 1/1 | 18869 |
| sympy | 1/1 | 11400 |
| pylint | 1/1 | 5859 |
| scikit-learn | 1/1 | 10297 |
| sphinx | 1/1 | 10325 |
| seaborn | 1/1 | 2848 (verified pre-jetsam — see `qwen36-seaborn-verified-pre-jetsam`) |
| requests | 1/1 | 1963 (`qwen36-perinst-missing-ecosystems`) |
| xarray | 1/1 | 3364 (timeout but worktree captured a 1178B patch) |
| pytest | 1/1 | 11143 |
| flask | 0/1 | 4045 (real model ceiling — model gave up after 5 tool calls) |
| **TOTAL** | **19/21 = 90.5%** | **12 ecosystems** |

**Method note:** the cross-ecosystem result was unlocked by running each
instance in its own freshly-booted server (`/tmp/run-per-instance.sh` style
wrapper). Multi-instance single-server sweeps at CTX=131K on this M4 hit
recurring macOS jetsam — see `qwen36-missing-ecosystems-JETSAM-2026-05-18/`
and `qwen36-missing-retry-JETSAM-2026-05-18/`. The hardened
`run_rollouts.py` (per-instance preflight, landed 2026-05-18) cleanly aborts
on jetsam detection so contaminated 0-byte rows don't get misread as model
failures. For sweeps beyond a single instance, ALWAYS use the per-instance
restart pattern.

**Model ceiling characterization:** the 2 misses (django-11019, flask-4045)
share a signature — both require **adding behavior** (validation logic /
algorithmic rewrite) rather than **correcting visible behavior**. qwen36
can fix bugs it can see but struggles to invent missing logic that isn't
surfaced by failing tests. Plausibly ~5-10% of SWE-bench Lite falls into
this class.

**Real resolved rate (M4-local scoring, 2026-05-18):** patch-engagement
is not the same as resolved. The full qwen36 prediction set scored on M4
via `score_local.py` (per-instance venv + native pytest, no Docker)
returned **3/21 = 14.3% resolved overall, 3/10 = 30% resolved on the
M4-scorable subset**. Detailed breakdown in
[`runs/qwen36-score-local-2026-05-18/`](runs/qwen36-score-local-2026-05-18/).

Score categories:

| Category | Count | Meaning |
|---|:---:|---|
| RESOLVED | 3 | F2P all pass + no P2P regressions |
| CLOSE | 2 | Partial F2P, no P2P regressions |
| WRONG LOCATION | 3 | No F2P, no P2P regressions (patch in wrong place) |
| BROKEN P2P | 4 | Patch caused regressions in existing tests |
| MODEL PATCH FAIL | 3 | Empty patch or unapplicable (multi-file mismatch) |
| INSTALL FAIL | 8 | M4 can't build the venv (old Python / native deps); needs 3090 Docker |

The 90.5% patch-engagement → 30% resolved gap is the model's "writing
in the style of" behavior: patches are in the right file, syntactically
valid, often regression-free, but miss the semantic requirement. The
3 resolved instances (django-11001, django-11039, pylint-5859) are all
in repos with heavy qwen36 training exposure; the misses skew toward
niche libraries (xarray, sphinx, seaborn).

For the full picture on the 8 INSTALL FAIL instances (astropy×6,
matplotlib, scikit-learn), ship `exports/qwen36-predictions.jsonl` to
the 3090 stack's `score_docker.py`.

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
