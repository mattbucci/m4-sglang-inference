# Cross-stack predictions exports

This directory holds deduplicated SWE-bench Lite prediction JSONL files
ready to ship to the 3090 stack for Docker-harness scoring.

The M4 box can't run the official SWE-bench Docker harness (no Docker
images for Apple Silicon for many of the older Python interpreters that
SWE-bench instances pin). The 3090 stack has the Docker setup and the
score_docker.py harness. The handoff:

```
M4 (this box)                    3090 stack
─────────────                    ──────────
omp → SGLang+MLX rollouts
   ↓
aggregate.py --export
   ↓
predictions.jsonl    ────────►   score_docker.py --predictions <path>
                                    ↓
                                 scores-docker-summary.json  ──►  back to M4
                                 (this dir: qwen36-docker-summary.json)
```

## Files

- `{coder-30b,nemotron-30b,qwen3-moe,qwen38-27b}-omp-26inst-predictions.jsonl` — the same 26
  instances under omp (`runs/<cell>-26inst-omp-2026-11-13/`): 25, 10, 11 and 24 non-empty
  patches; local scorer 1, 1, 1 and 6 of 26; awaiting Docker scoring on the 3090.

- `qwen36-p023-predictions.jsonl` — the same 26 instances rerun on
  SGLang v0.5.23 + patches 002–023 (`runs/qwen36-26inst-p023-2026-10-31/`):
  **26/26 non-empty patches**; awaiting Docker scoring on the 3090.

- `qwopus38-coder-flash-26inst-predictions.jsonl` — the same 26 instances on
  the Qwopus3.8-27B-Coder-Flash MLX 4-bit build through the sealed harness
  (`runs/qwopus38-27b-coder-flash-26inst-sealed-2026-11-02/`, 23/26 non-empty,
  model name `sglang/qwopus38-27b-coder-flash`); awaiting Docker scoring.
- `qwen38-27b-community-26inst-predictions.jsonl` — the same 26 instances on
  stock `mlx-community/Qwen3.8-27B-4bit` through the sealed harness
  (`runs/qwen38-27b-community-26inst-sealed-2026-11-02/`, 22/26 non-empty,
  model name `sglang/qwen38-27b-community`); awaiting Docker scoring.
- `qwen36-omp-26inst-predictions.jsonl` — the same 26 instances on qwen36
  through the omp scaffold and the sealed harness
  (`runs/qwen36-26inst-omp-2026-11-08/`, 26/26 non-empty, 6/26 on the local
  scorer, model name `sglang/qwen36-omp`); the opencode-vs-omp pair for Docker scoring.
- `qwen36-omp-final-26inst-predictions.jsonl` — the same 26 instances on qwen36
  under omp with the fused-decode preset defaults (`runs/qwen36-omp-final-26inst-2026-11-12/`,
  model name `sglang/qwen36-omp-final`); awaits Docker scoring.
- `qwen36-omp-p033-26inst-predictions.jsonl` — the same 26 instances on qwen36
  under omp on the 25-patch tree (`runs/qwen36-26inst-omp-p033-2026-11-11/`,
  26/26 non-empty, 5/26 on the local scorer, model name
  `sglang/qwen36-omp-p033`); a second rollout of the omp cell for Docker scoring.
- `flint-oQ8e-omp-26inst-predictions.jsonl` — the same 26 instances on Flint
  oQ8e, bash-only on omp through the sealed harness
  (`runs/flint-oQ8e-26inst-omp-bashonly-remap-sealed-2026-11-08/`, 5/26
  non-empty, 0/26 on the local scorer, model name `sglang/flint-oQ8e-omp`).
- `qwen36-predictions.jsonl` — deduplicated qwen36 results across all
  archived runs in `evals/swebench/runs/`. **26 instances, 21 non-empty
  patches (80.8%)** (export at commit a5a52d8). The 5 empty:
  `django__django-11019`, `pallets__flask-4045`, `psf__requests-2148`,
  `sphinx-doc__sphinx-10451`, `sympy__sympy-11870`.
- `qwen36-docker-summary.json` — the official Docker verdict from the
  3090 rig (2026-07-19; see Results below).

## Regenerate

```bash
python evals/swebench/aggregate.py \
    --export evals/swebench/exports/qwen36-predictions.jsonl \
    --model-filter sglang/qwen36
```

The aggregator prefers records with non-empty patches when deduplicating
across runs that hit the same instance multiple times — without that, the
alphabetic last-write-wins would prefer 0-byte diagnostic runs (e.g.
`qwen36-thinking-*`) over real post-proxy runs.

## Score on the 3090

```bash
# On the 3090 stack — the predictions file must live in a DEDICATED subdir:
# score_docker.py derives run_id from the parent dir name and writes
# scores-docker/ + scores-docker-summary.json next to the file.
# (There is no --output flag.)
mkdir -p evals/swebench/m4-imports/qwen36-m4
cp <m4-export> evals/swebench/m4-imports/qwen36-m4/predictions.jsonl
python evals/swebench/score_docker.py \
    --predictions evals/swebench/m4-imports/qwen36-m4/predictions.jsonl \
    --max-workers 1 --timeout 1800
```

## Results (scored 2026-07-19 on the 3090 rig)

**Official Docker rate: 9/26 resolved = 34.6%** (9/21 = 42.9% of non-empty;
12 unresolved, 5 empty, **0 errors** — all 21 non-empty patches ran, including
the 8 old-Python instances M4 cannot build).

- **All 5 M4-local resolves re-resolved under Docker** (no flips —
  `score_local.py` was accurate on its subset): django-10914, django-11001,
  django-11039, pylint-5859, sphinx-10325.
- **4 additional resolves came from the M4-untestable class**:
  astropy-12907, astropy-14995, pytest-11143, scikit-learn-10297.
- Against the pre-registered expectation band ("60-80% strong, >50% confirms
  the qwen36 recommendation"): **34.6% lands below the confirmation bar.**
  The number is consistent with the ~30B-class SWE-bench Lite band; the
  earlier 93.75%/80.8% patch-engagement figures measured plausibility, not
  correctness. qwen36 remains the only model that completes the agentic loop
  on M4 — the recommendation stands on that basis, with pass-rate expectations
  now calibrated by an official harness number.

Receipts on the 3090 rig: `evals/swebench/m4-imports/qwen36-m4/`
(predictions copy, `scores-docker-summary.json`, archived official report).
