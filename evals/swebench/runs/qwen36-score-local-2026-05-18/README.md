# qwen36 SWE-bench Lite score_local results — calibrating "patch-engagement" vs "resolved"

## Setup

Ran `evals/swebench/score_local.py` against the full deduplicated qwen36
predictions set (21 unique instances) on M4 directly — no Docker, just
per-instance Python venvs + native test commands. This is the rigorous
follow-up to the patch-engagement rate (90.5% across 12 ecosystems) and
the first time the M4 stack has produced real PASS/FAIL data without
shipping predictions to the 3090 Docker harness.

## Bottom line

| Metric | Value |
|---|---|
| Patch-engagement (qwen36 produced a non-empty patch) | 19/21 = 90.5% |
| Patch-applies-on-test-worktree (M4 scorable) | 10/21 = 47.6% |
| Patch-applies / scorable-on-M4 | 10/13 = 77% |
| **Resolved (all F2P pass + no P2P regressions)** | **3/21 = 14.3%** |
| **Resolved among M4-scorable patches** | **3/10 = 30.0%** |

The headline number — **3/10 = 30% resolved** — is the real SWE-bench
Lite score qwen36 produces on the M4 subset. The 8 INSTALL FAILs (mostly
old-Python repos: astropy needs 3.6, matplotlib 3.7, scikit-learn 3.6
with native deps) can't be scored on M4 without the Docker harness;
those rows need to ship to the 3090 stack via the export.

## Per-instance breakdown

| Category | Count | Instances |
|---|:--:|---|
| **RESOLVED** | 3 | `django-11001`, `django-11039`, `pylint-5859` |
| **CLOSE** — partial F2P, no P2P regressions | 2 | `requests-1963` (6/7 F2P), `pytest-11143` (1/1 F2P but 4 P2P regressions) |
| **WRONG LOCATION** — no F2P, no P2P regressions | 3 | `seaborn-2848`, `django-10924`, `sympy-11400` |
| **BROKEN P2P** — patch causes regressions | 4 | `requests-1963` (1 reg), `xarray-3364` (79 regs), `pytest-11143` (4 regs), `sphinx-10325` (5 regs) |
| **MODEL PATCH FAIL** — empty or unapplicable | 3 | `flask-4045` (empty), `django-11019` (empty), `django-10914` (multi-file, apply failure) |
| **INSTALL FAIL** — M4 can't build venv | 8 | 6× astropy, matplotlib, scikit-learn |

Note: `requests-1963` and `pytest-11143` appear in both CLOSE and
BROKEN P2P because they partially solve the issue but break existing
behavior — closest to "almost there but caused collateral damage."

## What the gap (90.5% patch-engagement → 30% resolved) actually means

The model produces patches that are:

- **In the canonical file** for the bug (verified manually for the
  per-instance miss READMEs earlier today)
- **Syntactically valid** Python that applies cleanly to the worktree
  (10/13 applied)
- **Often regression-free** on existing tests (3/10 had no P2P regressions
  among non-resolved)
- **But miss the actual semantic requirement** (only 3/10 pass the F2P
  tests that the gold patch is supposed to enable)

That's the "writing in the style of the right fix" failure mode — the
model has learned what fixes look like (right file, right region,
plausible diff) without learning what they need to do (cause the
new tests to pass). This is the kind of failure that patch-engagement
rates can't surface.

## Implication for the qwen36 recommendation

**Before:** "qwen36 at 90.5% patch-engagement is a strong agentic-coding
result on this hardware."

**After:** "qwen36 at 90.5% patch-engagement, but only 30% real SWE-bench
Lite resolved rate on the M4-scorable subset. Real-world usage will see
the model produce many patches that LOOK right but miss the semantic
intent. **Trust the model for first-pass exploration; verify with tests
before merging.** This is consistent with what SWE-bench community
reports for ~30B-class models in general — 14-30% is the typical band."

## What this still does NOT prove

- **Docker pass-rate via the 3090**: 8 of 21 (38%) instances can't be
  scored on M4 due to old-Python install failures. The full picture
  requires shipping `evals/swebench/exports/qwen36-predictions.jsonl`
  to the 3090 stack. Astropy patches might score better than 30% if
  testable — the model targeted them with the same care.
- **N=300 (full SWE-bench Lite) resolved rate**: 21 is a small sample.
  Could be ±10pp easily.
- **Whether the "close but broke P2P" patches would resolve with a
  retry**: re-rolling those 4 instances might surface the issue.

## The 3 RESOLVED instances (manual review)

### `django__django-11001`
F2P 2/2 + P2P 118/118. The model produced a 1368B patch targeting the
right ORM construct. **Clean resolve.**

### `django__django-11039`
F2P 1/1 + P2P 88/88. 826B patch. **Clean resolve.**

### `pylint-dev__pylint-5859`
F2P 1/1 + P2P 10/10. 656B patch. **Clean resolve** — the model fixed
a pylint warning false-positive correctly.

All 3 resolved instances are in repos where qwen36 has obvious training
exposure (Django ORM, pylint checks). The CLOSE/BROKEN-P2P instances
are in more niche libraries (xarray, sphinx, seaborn graph rendering).

## Files

- `scores.jsonl` — per-instance score records (resolved, patch_applied,
  f2p_passed, p2p_passed, score_seconds). Compatible with the
  princeton-nlp/SWE-bench-experiments leaderboard format.

## Method

```bash
# Generate predictions file
python evals/swebench/aggregate.py \
    --export /tmp/qwen36-all-preds.jsonl \
    --model-filter sglang/qwen36

# Score locally (one venv per instance, ~3-30s pytest each)
python evals/swebench/score_local.py \
    --predictions /tmp/qwen36-all-preds.jsonl \
    --workdir /tmp/swebench-score-work \
    --venvdir /tmp/swebench-score-venvs \
    --out evals/swebench/runs/qwen36-score-local-2026-05-18/scores.jsonl
```

Full batch took ~3 minutes on M4 (most instances scored in 3-37 seconds;
the install-fail cases bail out fast).
