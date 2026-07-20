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
opencode → SGLang+MLX rollouts
   ↓
aggregate.py --export
   ↓
predictions.jsonl    ────────►   score_docker.py --predictions <path>
                                    ↓
                                 scores-docker-summary.json  ──►  back to M4
                                 (this dir: qwen36-docker-summary.json)
```

## Files

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
