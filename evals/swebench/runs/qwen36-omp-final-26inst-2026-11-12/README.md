# qwen36, 26-instance sealed SWE-bench Lite cell on omp, fused-decode defaults — 26/26 non-empty, local scorer 4/26, decode 72 tok/s against 62

The scorecard's 26 instances on the preset default `mlx-community/Qwen3.6-35B-A3B-4bit`, same command as
`../qwen36-26inst-omp-p033-2026-11-11/` (the reference cell below): `PRESET=qwen36 TIMEOUT=900 bash
evals/swebench/smoke.sh`, omp 18.12.2, thinking off, sealed. The server is SGLang v0.5.23 with the patch
series through 041. Against the reference cell's server the preset now runs the fused MoE router, the fused
DeltaNet front, grouped decode linears and five-part submission by default (the launch log shows all four
"enabled" lines). One server served all 26 in sequence: no traceback, 81 min of rollout wall against 89.

## Result

26/26 non-empty patches, every one touching a file the reference cell's patch touches. No instance reached
the 900 s cap (longest 717 s). Per-instance wall median 110 s against 148 s, 4847 s in total against 5320 s.

`score_local.py` resolves **4/26** against 5/26: the reference cell's five without `sphinx-10325`
(FAIL_TO_PASS 0/1 here). One instance on 26 is inside rollout-to-rollout variation; the lane decodes
sampled. The official number needs the Docker harness on the 3090:
`../../exports/qwen36-omp-final-26inst-predictions.jsonl` (model name `sglang/qwen36-omp-final`).

### Served decode rate

Decode rate per assistant turn from omp's timings, median per bin, turns of 128 or more output tokens:

| Prompt depth | Reference cell | This cell | Change |
|---|---:|---:|---:|
| under 8K | 75.0 tok/s (9 turns) | 90.8 tok/s (9 turns) | +21 % |
| 8K to 16K | 71.2 tok/s (51) | 86.3 tok/s (53) | +21 % |
| 16K to 32K | 67.9 tok/s (184) | 80.6 tok/s (156) | +19 % |
| 32K to 64K | 58.9 tok/s (175) | 68.2 tok/s (177) | +16 % |
| 64K and over | 50.8 tok/s (42) | 58.4 tok/s (58) | +15 % |

Pooled over every turn: 233,760 output tokens in 3254 s of decode wall (71.8 tok/s) against 227,209 in
3653 s (62.2 tok/s), +15 %. Time to first token: median 0.63 s against 0.66 s, p90 2.29 s against 2.5 s,
1304 s in total against 1393 s. 1124 assistant turns against 1092; tool calls `read` 381, `bash` 334,
`edit` 218, `grep` 161, `glob` 46, `todo` 32. Largest prompt 88K tokens.

### Per instance

"Archived cell" columns are the reference cell.

| Instance | Wall | Patch | Files | Archived cell (wall, patch) | Same file | Local score | Archived local score |
| `astropy__astropy-12907` | 64 s | 504 B | `astropy/modeling/separable.py` | 289 s, 1662 B | yes | install fail (M4) | install fail (M4) |
| `astropy__astropy-14182` | 43 s | 622 B | `astropy/io/ascii/rst.py` | 110 s, 928 B | yes | install fail (M4) | install fail (M4) |
| `astropy__astropy-14365` | 35 s | 617 B | `astropy/io/ascii/qdp.py` | 55 s, 617 B | yes | install fail (M4) | install fail (M4) |
| `astropy__astropy-14995` | 45 s | 626 B | `astropy/nddata/mixins/ndarithmetic.py` | 21 s, 626 B | yes | install fail (M4) | install fail (M4) |
| `astropy__astropy-6938` | 37 s | 533 B | `astropy/io/fits/fitsrec.py` | 20 s, 514 B | yes | install fail (M4) | install fail (M4) |
| `astropy__astropy-7746` | 49 s | 1077 B | `astropy/wcs/wcs.py` | 79 s, 1008 B | yes | install fail (M4) | install fail (M4) |
| `django__django-10914` | 174 s | 3184 B | `django/conf/global_settings.py`, `docs/howto/deployment/checklist.txt` + 2 more | 200 s, 3908 B | yes | **resolved** | **resolved** |
| `django__django-10924` | 142 s | 1794 B | `django/db/models/fields/__init__.py`, `django/forms/fields.py` | 263 s, 2295 B | yes | applied; F2P 0/1, P2P 1/1 | applied; F2P 0/1, P2P 0/1 |
| `django__django-11001` | 53 s | 1296 B | `django/db/models/sql/compiler.py` | 90 s, 1761 B | yes | **resolved** | **resolved** |
| `django__django-11019` | 472 s | 5134 B | `django/forms/widgets.py` | 907 s, 7745 B | yes | applied; F2P 5/16, P2P 58/58 | applied; F2P 5/16, P2P 58/58 |
| `django__django-11039` | 50 s | 810 B | `django/core/management/commands/sqlmigrate.py` | 48 s, 658 B | yes | **resolved** | **resolved** |
| `matplotlib__matplotlib-18869` | 113 s | 1433 B | `lib/matplotlib/__init__.py` | 164 s, 545 B | yes | install fail (M4) | install fail (M4) |
| `mwaskom__seaborn-2848` | 183 s | 1519 B | `seaborn/axisgrid.py` | 704 s, 1606 B | yes | applied; F2P 0/1, P2P 50/50 | applied; F2P 0/1, P2P 50/50 |
| `pallets__flask-4045` | 53 s | 496 B | `src/flask/blueprints.py` | 47 s, 491 B | yes | applied; F2P 0/2, P2P 50/50 | applied; F2P 0/2, P2P 50/50 |
| `psf__requests-1963` | 64 s | 795 B | `requests/sessions.py` | 130 s, 760 B | yes | applied; F2P 6/7, P2P 112/112 | applied; F2P 6/7, P2P 112/112 |
| `psf__requests-2148` | 107 s | 1271 B | `requests/models.py` | 224 s, 1744 B | yes | applied; F2P 9/10, P2P 118/118 | applied; F2P 9/10, P2P 118/118 |
| `pydata__xarray-3364` | 510 s | 6213 B | `xarray/core/concat.py` | 297 s, 4025 B | yes | applied; F2P 0/2, P2P 110/110 | applied; F2P 0/2, P2P 110/110 |
| `pylint-dev__pylint-5859` | 606 s | 664 B | `pylint/checkers/misc.py` | 185 s, 650 B | yes | **resolved** | **resolved** |
| `pylint-dev__pylint-6506` | 264 s | 1285 B | `pylint/lint/run.py` | 165 s, 1199 B | yes | applied; F2P 0/2, P2P 0/6 | applied; F2P 0/2, P2P 0/6 |
| `pytest-dev__pytest-11143` | 64 s | 830 B | `src/_pytest/assertion/rewrite.py` | 38 s, 582 B | yes | applied; F2P 1/1, P2P 110/114 | applied; F2P 1/1, P2P 110/114 |
| `pytest-dev__pytest-11148` | 510 s | 1297 B | `src/_pytest/pathlib.py` | 179 s, 1241 B | yes | applied; F2P 2/2, P2P 128/129 | applied; F2P 2/2, P2P 128/129 |
| `scikit-learn__scikit-learn-10297` | 83 s | 1765 B | `sklearn/linear_model/ridge.py`, `sklearn/linear_model/tests/test_ridge.py` | 90 s, 930 B | yes | install fail (M4) | install fail (M4) |
| `sphinx-doc__sphinx-10325` | 126 s | 3937 B | `doc/usage/extensions/autodoc.rst`, `sphinx/ext/autodoc/__init__.py` | 133 s, 2414 B | yes | applied; F2P 0/1, P2P 5/5 | **resolved** |
| `sphinx-doc__sphinx-10451` | 144 s | 1887 B | `sphinx/ext/autodoc/typehints.py` | 242 s, 2107 B | yes | applied; F2P 0/2, P2P 80/80 | applied; F2P 0/2, P2P 80/80 |
| `sympy__sympy-11400` | 137 s | 1292 B | `sympy/core/basic.py`, `sympy/plotting/plot.py` + 1 more | 87 s, 538 B | yes | applied; F2P 0/2, P2P 29/29 | applied; F2P 0/2, P2P 29/29 |
| `sympy__sympy-11870` | 717 s | 3797 B | `sympy/assumptions/sathandlers.py`, `sympy/core/basic.py` + 3 more | 552 s, 2685 B | yes | applied; F2P 0/1, P2P 57/58 | applied; F2P 0/1, P2P 58/58 |

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`, `scores.jsonl`, `sessions/<inst>/*.jsonl`
- `logs/`, `launch.log`, `proxy.log`, `rollout.log`, `score_local.log`, `omp-agent/` (local only)
