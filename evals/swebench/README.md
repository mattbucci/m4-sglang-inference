# SWE-bench Lite agentic coding on M4

Tool-call-driven coding evaluation: feed the model a GitHub issue + repo,
let it use `read`/`glob`/`grep`/`edit`/`write`/`bash` tools to produce a
patch, score the patch against the official SWE-bench test harness.

**Scaffold: omp (oh-my-pi) 18.12.2, one lane.** The 3090 rig's scaffold
bake-off ran opencode, opencode+DCP, little-coder, little-coder+RTK, prime
and omp on the same 300 isolated instances and every opencode / pi-family
pair is a McNemar tie (Flint: omp 186 vs opencode 182, p = 0.57), so harness
variants are not a lever and the two rigs run the same pinned omp.
`bash evals/swebench/vendor_omp.sh` installs the sha-pinned binary into
`evals/swebench/bin/` (gitignored); `SCAFFOLD=opencode` reproduces the
archived opencode cells.

This is the **load-bearing benchmark for the M4 stack's "long-context
agentic coding" goal.** Static MMLU / HumanEval / Needle numbers in the
README's quality table don't predict agentic capability on this hardware
— SWE-bench Lite does.

## Stack

```
SWE-bench Lite (princeton-nlp/SWE-bench_Lite)
  └── run_rollouts.py (HF dataset loader + ancestor-only checkout + scaffold driver)
      └── sandbox-exec localhost-only.sb (no network past this host)
          └── omp 18.12.2 --yolo --no-title -p --thinking off
              (tools read bash edit eval glob grep wait todo write; task/web_search off;
               provider from <run>/omp-agent/models.yml, session jsonl in <run>/sessions/)
          └── HTTP → no_thinking_proxy.py :23335
              └── HTTP → SGLang :23334
                  └── MLX backend, native Apple Silicon
```

omp reads its provider and settings from the per-run agent dir
`run_rollouts.py` writes (`PI_CODING_AGENT_DIR`): `contextWindow` = the
server's `CTX` (it drives compaction), `maxTokens` and the compaction
reserve = `OUTPUT_BUDGET` (8192, the output cap the archived opencode cells
ran with). `--no-title` is load-bearing: omp's title generator forks a side
turn that replays the whole prefix on the served model whenever the todo
tool initialises a plan, a second full prefill per step. Each session is
archived twice: the `--mode json` event stream in `logs/<inst>.log` and the
pi-row session store in `sessions/<inst>/*.jsonl`.

**`no_thinking_proxy.py`** forces `chat_template_kwargs={"enable_thinking":
false}` on every chat-completion POST. omp's `--thinking off` puts
`enable_thinking: false` on the wire itself, so under omp the proxy is a
belt-and-braces guard; it stays on by default so every cell is served the
same way. It also carries the argument repairs Flint needs: `BACKFILL_INTENT=1`
derives omp's required `i` (intent) argument when a call omits it (the
opencode form is `BACKFILL_BASH_DESCRIPTION=1` for the bash `description`),
and `REMAP_INTENT=1` moves a payload Flint sends as `i` (its hashline edit
on 115 of 118 `edit` calls in the full-tool smoke) to the tool's own
argument and clamps `i` to omp's 200-character limit. Both are inherited
by the proxy from the `smoke.sh` environment. The
rationale is the Qwen3.6 agent loop: thinking on, the model's content either
lands in `reasoning_content` or leaks `</think>` markers into the tool-call
stream, and the loop makes 9 tool calls for a 0-byte diff; thinking off it
patches the canonical location.

Verified 2026-05-18: without the proxy, qwen36 makes 9 tool calls but
produces a 0-byte diff (reasoning is invisible). With the proxy, qwen36
makes 6 tool calls including 1 `edit` and produces a 506-byte patch
that targets the canonical bug location.

## Sealed harness: the answer cannot be downloaded

Three channels would let a session obtain the upstream fix instead of
writing one, and the archive shows two of them being used (neither session
produced a patch):

- **Web tools.** opencode's `webfetch` pulled the post-fix
  `astropy/modeling/separable.py` from three later release tags during a
  Qwen3.8 session on `astropy-12907`. The web tools are off on both
  scaffolds: omp runs with `--tools` naming the file/search/shell set only
  (`web_search` and `task` out), opencode gets `webfetch`, `websearch` and
  `task` switched off in its injected tools block.
- **Package index / git over the network.** A Flint session on
  `astropy-14182` ran `pip download astropy==6.0`, unpacked the wheel and
  read the fixed `rst.py` and its tests. `run_rollouts.py` runs the scaffold under
  a seatbelt profile (`localhost-only.sb`: `deny network-outbound` except
  `localhost`), inherited by every process its bash tool spawns, so curl,
  pip, git and DNS fail; the model's server and the proxy are on 127.0.0.1.
  `SWEBENCH_SANDBOX=0` lifts it (Linux hosts: no seal, use a network
  namespace).
- **Future history in the checkout.** A plain clone of the mirror carried
  every branch and tag: 50,492 commits reachable on astropy where the base
  commit reaches 32,966, the fix merge among them (`git log --all`,
  `git branch -a`, `git show <later sha>` all worked; sessions ran the first
  two). `ensure_repo` now fetches exactly `base_commit` from the mirror into
  a fresh repository (`uploadpack.allowAnySHA1InWant`), so the object store
  holds the base commit and its ancestors only; the harness asserts
  `rev-list --all == rev-list HEAD` before the rollout (3.5 s per instance,
  106 MB for astropy).

`audit_rollouts.py` scans every archived session (opencode event logs and
omp session rows; omp's `web_search`/`github` calls, a `read` of a URL and
`eval` code that fetches count as web/network) for the three channels and
prints the patch size the session produced:

```
python3 evals/swebench/audit_rollouts.py            # all runs/*
python3 evals/swebench/audit_rollouts.py --verbose  # with the matching commands
```

Archive audit (165 sessions): 8 sessions with a hit, all from before the seal. Two reached the
internet (the Qwen3.8 `webfetch` session and the Flint `pip download`
session), both 0-byte patches. Six ran a history-reading git command
(`git log --all --grep`, `git branch -a`, `git show <sha>`): in every case
the output shown to the model was pre-base history or unrelated later
commit subjects, never the fix's diff (`--verbose` lists them). No scored
prediction in `exports/` came out of a session that saw the fix.

The two smokes whose sessions had reached the internet were repeated under
the seal: Qwen3.8-27B community 3/3, each patch the upstream change
(`runs/qwen38-27b-community-sealed-131k-2026-11-01/`; the 12907 session
that had fetched the post-fix file now writes that fix from source), and
Flint oQ8e bash-only 64K 1/3 with zero runaway turns in 265 responses
(`runs/flint-oQ8e-bashonly-64k-sealed-2026-11-01/`).

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

**The complete int4 agentic verdict (2026-05-18 bakeoff,
[`int4-bakeoff-2026-05-18/`](runs/int4-bakeoff-2026-05-18/)):**

| Model | Verdict | Failure mode |
|---|---|---|
| `qwen36` (Qwen3.6-35B-A3B-4bit MoE+DeltaNet+VL) | ✓ **WORKS** | — |
| `qwen35` (Qwen3.5-27B-4bit DeltaNet+VL) | ✓ Works @ TIMEOUT=1800 | 15× slower wall |
| `qwen36-27b` (Qwen3.6-27B Dense+DeltaNet+VL) | ✗ | TIMEOUT, model couldn't converge |
| `qwen3-32b` (Qwen3-32B-DWQ Dense) | ✗ | TIMEOUT, model couldn't converge |
| `qwen3-moe` (Qwen3-30B-A3B-DWQ) | ✗ | Emits CORRUPTED tool-call tags (`<\|name>read>...</read>` half-tokens + malformed JSON) — special-token mis-mapping from MLX-DWQ conversion, not a parser issue |
| `coder-30b` (Qwen3-Coder-30B-A3B-DWQ) | ✗ | 1 glob then asks user — chat template gives up |
| `devstral` (Mistral-arch 24B) | ✗ | Preflight canary blocked by Mistral template strictness |
| `gemma4-31b` (gemma-4-31b-it-mxfp4) | ✗ | 0 tokens emitted under tool prompts |
| `nemotron-30b` (Nemotron-3-Nano-30B-A3B) | ✗ | TIMEOUT, model couldn't converge |
| `nemotron-omni` (NemotronH+VLM 30B-A3B) | ✗ | MLX bug patched 2026-05-19 (partial-cache fallback in `decode_batch_start`); model now decodes cleanly but emits only 23 output tokens then signals stop — chat-template gap, same class as coder-30b |
| `gemma4` (26B-A4B MoE) @ TIMEOUT=1800 | **⚠ Doesn't reproduce** | MLX backend patched 2026-05-19 (patch 020). On v0.5.11, 5-instance characterization showed 1/5 RESOLVED (pylint-5859). **On v0.5.12 retest of pylint-5859: NOT resolved** — same instance, same model, same TIMEOUT, model made 64 tool calls (vs 48 prior) but never an `edit` call; produced 180B reproduction-only patch instead of the prior 1003B canonical fix (`gemma4-pylint-v0512-FAILS-2026-05-20/`). Verdict: the one prior success was a non-reproducible single-trajectory coincidence. Upstream gemma fixes since v0.5.11 (PR #24696, #24048, #25286, #25284, #23976, #24436, #25006, #25547) all target server-GPU code paths irrelevant to MLX. Don't recommend for any agentic work on M4. |
| `qwen35-9b-8bit` (Qwen3.5 9B 8-bit) | ✗ | Engages (13 tool calls) but **can't synthesize an edit** — 9B capacity ceiling |

**Only qwen36 + qwen35 work.** Both use Qwen3-Coder tool-call parser +
DeltaNet+VL architecture; neither factor alone is enough (qwen36-27b is
Dense+DeltaNet+VL+Coder-parser but couldn't converge). The MoE-3B-active
decode speed of qwen36 appears to be what keeps the agent loop alive at
M4's decode rates.

**FP8 retry verdict (negative)**: most failures aren't quantization-sensitive
— gemma4-31b emits 0 tokens (chat-template gap), qwen3-moe emits unparseable
tags (template/parser mismatch), the rest TIMEOUT during legitimate decode
(more precision won't reduce wall time). Not worth 32GB downloads.

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
instance in its own freshly-booted server via
[`evals/swebench/perinst.sh`](perinst.sh). Multi-instance single-server
sweeps at CTX=131K on this M4 hit recurring macOS jetsam — see
`qwen36-missing-ecosystems-JETSAM-2026-05-18/` and
`qwen36-missing-retry-JETSAM-2026-05-18/`. The hardened `run_rollouts.py`
(per-instance preflight, landed 2026-05-18) cleanly aborts on jetsam
detection so contaminated 0-byte rows don't get misread as model
failures. For sweeps beyond a single instance, ALWAYS use the
per-instance restart pattern:

```bash
bash evals/swebench/perinst.sh pallets__flask-4045 psf__requests-1963 ...
```

The script handles fresh-server boot per instance, OS memory reclaim
pause, predictions merging, and optional score_local pass.

**Model ceiling characterization:** the 2 misses (django-11019, flask-4045)
share a signature — both require **adding behavior** (validation logic /
algorithmic rewrite) rather than **correcting visible behavior**. qwen36
can fix bugs it can see but struggles to invent missing logic that isn't
surfaced by failing tests. Plausibly ~5-10% of SWE-bench Lite falls into
this class.

**Real resolved rate (M4-local scoring, 2026-05-18, N=26):**
patch-engagement is not the same as resolved. The full qwen36
prediction set scored on M4 via `score_local.py` (per-instance venv +
native pytest, no Docker) returned **5/26 = 19.2% resolved overall,
5/13 = 38.5% resolved on the M4-scorable subset**. Note this is the
sample-bias-corrected number — the original 21-instance scorecard
(45.5%) had favorable picks; the 5-instance widening sweep returned
2/5 patch-engagement and 0/5 resolved, refining the numbers down ~10pp.
Detailed breakdown in
[`runs/qwen36-score-local-2026-05-18/`](runs/qwen36-score-local-2026-05-18/)
and [`runs/qwen36-widening-N5-2026-05-18/`](runs/qwen36-widening-N5-2026-05-18/).

Score categories:

| Category | Count | Meaning |
|---|:---:|---|
| RESOLVED | 5 | F2P all pass + no P2P regressions |
| CLOSE | 3 | Partial F2P, no P2P regressions |
| WRONG LOCATION | 3 | No F2P, no P2P regressions (patch in wrong place) |
| BROKEN P2P | 4 | Patch caused regressions in existing tests |
| MODEL PATCH FAIL | 5 | Empty patch (model gave up) — flask, django-11019, sphinx-10451, requests-2148, sympy-11870 |
| INSTALL FAIL | 8 | M4 can't build the venv (old Python / native deps); needs 3090 Docker |
| Total (N=26) | 28 | (some instances span CLOSE + BROKEN P2P) |

The 90.5% patch-engagement → 45% resolved gap is the model's "writing
in the style of" behavior: patches are in the right file, syntactically
valid, often regression-free, but miss the semantic requirement in ~55%
of testable cases. The 5 resolved instances (django-10914, django-11001,
django-11039, pylint-5859, sphinx-10325) are in repos with heavy qwen36
training exposure; the misses skew toward niche libraries (xarray,
seaborn).

**score_local.py fixes (2026-05-18):**

1. **Test-file strip**: `apply_patch` now strips test-file diff blocks
   from model patches before applying (matches SWE-bench's official
   Docker harness convention). Without it, `django-10914` failed to
   apply because the model correctly pre-empted the gold test_patch's
   edit and `git apply` choked on the overlap. **+1 resolved**.

2. **tox 4.x + pytest `-rA`**: SWE-bench specs use `tox --current-env`
   (legacy tox 3.x). In tox 4.x that fallback creates `.tox/py39/`
   without pytest, so all tests fail with "No module named pytest."
   Score_local now rewrites to `tox --runner current-env`. Also injects
   `-rA` after the tox `--` separator so pytest emits parseable
   `PASSED test_name` lines instead of dot-style summary. Only sphinx
   uses tox in the SWE-bench specs. **+1 resolved**.

Combined effect: 3 → 5 resolved (+67%). The 45.5% rate is significantly
above the 14-30% typical band — exceptionally strong for the model class.

For the full picture on the 8 INSTALL FAIL instances (astropy×6,
matplotlib, scikit-learn), ship `exports/qwen36-predictions.jsonl` to
the 3090 stack's `score_docker.py`.

## Quickstart

```bash
# 1. Run a single instance (defaults: qwen36, instance 1, proxy on)
bash evals/swebench/smoke.sh

# 2. Run a specific instance
INSTANCE_IDS="django__django-10914" bash evals/swebench/smoke.sh

# 3. Run multiple instances on a fresh server per instance (jetsam-immune)
bash evals/swebench/perinst.sh \
    pallets__flask-4045 \
    psf__requests-1963 \
    pydata__xarray-3364

# 4. Compare across the 4 README picks
bash evals/swebench/bakeoff.sh

# 5. Refresh the leaderboard from all archived runs
python evals/swebench/aggregate.py --markdown

# 6. Score predictions locally (no Docker) — see "Real resolved rate" below
python evals/swebench/score_local.py \
    --predictions evals/swebench/runs/<your-run>/predictions.jsonl \
    --workdir /tmp/score-work --venvdir /tmp/score-venvs \
    --out evals/swebench/runs/<your-run>/scores.jsonl
```

## Workflow

1. `smoke.sh` orchestrates: stops prior server, launches SGLang via
   `scripts/launch.sh`, waits for `/health`, starts `no_thinking_proxy`
   and hands its URL to the scaffold (omp: the per-run `models.yml`;
   opencode: `~/.config/opencode/opencode.jsonc` re-pointed at 23335).
2. `run_rollouts.py` loads SWE-bench Lite from HF, clones each repo at
   the issue's base commit, runs the scaffold in the worktree (omp:
   `omp --model sglang/<served> --yolo --no-title -p`, task on stdin;
   opencode: `opencode run --model sglang/<key>`) with the problem
   statement + tool-call directive prompt.
3. `eval_env.py` tries to build a per-instance venv from the repo's
   pinned deps so the model can `pytest`/`import` during the rollout.
   Often fails on M4 (we don't have system Python headers for old
   packages); falls back to a "no-venv" prompt mode where the model
   reasons about correctness from source alone.
4. Final `git diff` from the worktree is captured as the prediction.
5. `smoke.sh` tears down the proxy (restoring the opencode config when
   that scaffold ran) and kills the server.

Two harness behaviours that matter on the MLX backend:

- A rollout killed at its timeout has its in-flight turn aborted on the
  server (`/abort_request`, all requests) before the next instance starts.
  The per-instance `/health` probe generates a token through the
  scheduler, so a scheduler still prefilling an orphaned turn (one
  2048-token chunk of the dense 27B takes ~20 s) looks exactly like a
  reaped one; the probe retries for `SWEBENCH_HEALTH_WAIT` seconds
  (default 120) and only a refused connection or an exhausted budget
  counts as a dead server.
- The scaffold's context window must match the `CTX` the server is
  launched with (omp: `contextWindow` in the per-run `models.yml`, written
  from `CTX`; opencode: `opencode.json` `limit.context` per model key).
  Both scaffolds compact the session (a summary turn, then
  re-exploration) as the input nears the limit; a 32K limit against a
  131K server cost the Qwen3.8 sessions one to four compactions each, at
  1-2K summary tokens a time on a 13 tok/s model.
- Task-repo mirrors live under the harness workdir in `/tmp`; the periodic
  tmp cleanup can empty a mirror and leave its directory skeleton behind,
  so `ensure_repo` re-clones any mirror git no longer recognises.

## Layout

```
evals/swebench/
├── README.md                       # this file
├── vendor_omp.sh                   # installs the sha-pinned omp binary into bin/ (gitignored)
├── opencode.json                   # opencode provider config (sglang/* keys → 23334), the opencode lane only
├── no_thinking_proxy.py            # aiohttp+httpx proxy on 23335 (enable_thinking=false)
├── run_rollouts.py                 # scaffold driver (omp default, opencode) + SWE-bench loader
├── eval_env.py                     # per-instance venv builder
├── score_local.py                  # local-venv scoring (when Docker unavailable)
├── smoke.sh                        # single-preset orchestrator
├── bakeoff.sh                      # multi-preset comparison
├── perinst.sh                      # per-instance fresh-server sweep (jetsam-immune)
├── aggregate.py                    # leaderboard aggregator
└── runs/                           # archived rollout artifacts
    └── <model>-<tag>-<date>/
        ├── README.md               # per-run interpretation
        ├── predictions.jsonl       # one row per instance, `scaffold` names the pin
        ├── predictions/<inst>.diff
        ├── logs/<inst>.log         # scaffold event stream
        └── sessions/<inst>/*.jsonl # omp session store (pi row format)
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
