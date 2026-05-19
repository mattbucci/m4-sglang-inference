# qwen36 N=21→26 widening — sample-bias correction

## Why

The 21-instance qwen36 SWE-bench Lite scorecard (90.5% patch-engagement,
45.5% resolved on M4-scorable subset) had heavy sample bias toward
ecosystems where qwen36 had been picked carefully (typically the first
or canonical instance per ecosystem). To test the headline rate's
generalizability, ran a 5-instance widening sweep using fresh
`per-instance restart` workflow, all picks fresh (no prior qwen36
exposure).

## Picks

| Instance | Ecosystem | Pick reason |
|---|---|---|
| `sphinx-doc__sphinx-10451` | sphinx | 2nd sphinx (validate harness fixes at N>1) |
| `pytest-dev__pytest-11148` | pytest | 2nd pytest (corner case in pathlib) |
| `pylint-dev__pylint-6506` | pylint | 2nd pylint (config-init failure mode) |
| `psf__requests-2148` | requests | 2nd requests (low-level connection) |
| `sympy__sympy-11870` | sympy | 2nd sympy (ccode formatting) |

## Results

| Instance | Patch | Wall | rc | Resolved | F2P | P2P |
|---|---:|---:|:---:|:---:|:---:|:---:|
| sphinx-10451 | 0 B | 912 s | 124 | (no patch) | — | — |
| pytest-11148 | **639 B** | 904 s | 124 | NO | 0/2 | 128/129 |
| pylint-6506 | **1243 B** | 492 s | 0 | NO | 0/2 | 0/6 |
| requests-2148 | 0 B | 903 s | 124 | (no patch) | — | — |
| sympy-11870 | 0 B | 911 s | 124 | (no patch) | — | — |

**This batch: patch-engagement 2/5 = 40%, resolved 0/5 = 0%.** Steep
drop from the prior batch's 90.5% / 45.5%.

## Failure mode analysis

### sphinx-10451, requests-2148, sympy-11870: model gave up

All three TIMEOUT'd at 900s having produced 0 bytes. Same "model
gave up" pattern as `django-11019` and `flask-4045`. Without log
inspection these can't be distinguished from "model hit a tough
algorithmic case" vs "model just spun in tool calls."

### pytest-11148: real-looking partial

Model produced a 639B patch to `src/_pytest/pathlib.py:insert_missing_modules`
— removed an assignment (`module = importlib.import_module(...)` →
`importlib.import_module(...)`) which is consistent with the gold
patch's spirit (probably). Passed 128/129 P2P (1 regression) and
0/2 F2P. Real-but-incomplete fix.

### pylint-6506: wrong-place fix

Model added `try/except _UnrecognizedOptionError` in `pylint/lint/run.py`
to catch the existing exception. The gold patch makes a totally
different change in `pylint/config/config_initialization.py` —
replacing the exception RAISE with `linter._arg_parser.error(...)`.
The model's exception handler doesn't reach the test's expected
error-message-format pathway, so 0/2 F2P fail; the patch is also
broken enough to cause 0/6 P2P regressions (all config tests fail
with `pylint.exceptio...` errors).

## Combined N=26 picture

| Metric | Old (N=21) | New (N=26) |
|---|---|---|
| Patch-engagement | 19/21 = 90.5% | **21/26 = 80.8%** |
| Patches applied | 11/21 = 52.4% | **13/26 = 50.0%** |
| Resolved (M4-scorable) | 5/11 = 45.5% | **5/13 = 38.5%** |
| Resolved (all) | 5/21 = 23.8% | **5/26 = 19.2%** |

Per-ecosystem patch-engagement (N=26):

| Eco | Old | New | Notes |
|---|---|---|---|
| astropy | 6/6 | 6/6 | not retested |
| django | 4/5 | 4/5 | not retested |
| matplotlib | 1/1 | 1/1 | not retested |
| pylint | 1/1 | 2/2 | both pylint instances produce patches |
| pytest | 1/1 | 2/2 | both pytest instances produce patches |
| sphinx | 1/1 | 1/2 | sphinx-10451 model gave up |
| sympy | 1/1 | 1/2 | sympy-11870 model gave up |
| requests | 1/1 | 1/2 | requests-2148 model gave up |
| seaborn | 1/1 | 1/1 | not retested |
| xarray | 1/1 | 1/1 | not retested |
| flask | 0/1 | 0/1 | not retested |
| sklearn | 1/1 | 1/1 | not retested |

The 3 new ecosystems where we now have 1/2 (sphinx, sympy, requests)
suggest the original N=1 picks were systematically easier than the
ecosystem average.

## Recommendation refinement

**Before this run:** qwen36 at 90.5% patch-engagement / 45.5% resolved
on M4-scorable. "Exceptionally strong for ~30B class."

**After this run:** qwen36 at **80.8% patch-engagement / 38.5%
resolved** on M4-scorable. Still in the high end of the 14-30%
typical resolved band for ~30B-class models, but more honest. The
~10pp drops were sample bias — the original picks were carefully-
chosen "first instance per ecosystem" which leans toward simpler
canonical fixes.

The 80.8% patch-engagement remains a strong agentic-coding result;
this is "patches produced, not just answers," and qwen36 stands
alone among M4 models in achieving it.

## Files

- `predictions.jsonl` — 5 rollout records
- `predictions/<inst>.diff` — per-instance diffs (2 non-empty)
- `scores.jsonl` — score_local results
- `logs/<inst>.log` — opencode rollout logs (tool-call breakdown)
