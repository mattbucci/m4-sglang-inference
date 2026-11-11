# qwen36, 26-instance sealed SWE-bench Lite cell on omp, 25-patch tree — 26/26 non-empty, local scorer 5/26, decode 62 tok/s against 52

The scorecard's 26 instances on the preset default
`mlx-community/Qwen3.6-35B-A3B-4bit` with the same command as the archived
omp cell (`../qwen36-26inst-omp-2026-11-08/`): `scripts/launch.sh qwen36`
(CTX=131072, turboquant, chunk 2048, mem-fraction 0.5, radix cache on), omp
(oh-my-pi) 18.12.2, tool set `read bash edit eval glob grep wait todo write`,
thinking off, context window 131072, output budget 8192, 900 s per instance,
sealed (localhost-only sandbox, ancestor-only checkout). The server is SGLang
v0.5.23 + 25 patches. Against the archived cell's server that adds the
pipelined decode loop for hybrid presets under the radix cache (patch 028, on
by default), the prefix-hit position numbering (patch 033) and the
per-request position delta (patch 034). The opt-in decode flags (patches
029-032) and batched decode (patch 035) are off for this preset. One server
served all 26 in sequence: no jetsam, OOM guard zero warnings, 89 min of
rollout wall against 99.

## Result

26/26 non-empty patches, every one touching a file the archived cell's patch
touches. One instance ran to the cap: `django-11019` stopped at 907 s with a
7745 B patch already on disk (660 s in the archived cell). Per-instance wall
median 148 s against 187 s, 5320 s in total against 5912 s. Patches are
larger again (median 1104 B against 943 B).

`score_local.py` (per-instance venv, native pytest) resolves **5/26** (5 of
the 18 instances the box can build) against 6/26 for the archived cell. The
five are the archived cell's six without `django-10924`. On that instance
this rollout also rewrote the model field's `__init__` and dropped its
`super().__init__()` call, so the test module errors (FAIL_TO_PASS 0/1,
PASS_TO_PASS 0/1); the archived patch was the two-line forms change alone.
Among the misses, `pytest-11148` now passes both FAIL_TO_PASS tests,
`requests-2148` and `sympy-11400` keep PASS_TO_PASS clean, and `flask-4045`
loses the one FAIL_TO_PASS test it had. One instance of difference on 26 is
inside rollout-to-rollout variation: the two cells are separate greedy
rollouts whose turns diverge early. The official number needs the Docker
harness on the 3090: `../../exports/qwen36-omp-p033-26inst-predictions.jsonl`
(model name `sglang/qwen36-omp-p033`).

### Served decode rate

Decode rate per assistant turn from omp's own timings (output tokens over
duration minus time to first token), median per bin, turns of 128 or more
output tokens:

| Prompt depth | Archived cell | This cell | Change |
|---|---:|---:|---:|
| under 8K | 58.8 tok/s (12 turns) | 75.0 tok/s (9 turns) | +28 % |
| 8K to 16K | 57.1 tok/s (62) | 71.2 tok/s (51) | +25 % |
| 16K to 32K | 53.9 tok/s (191) | 67.9 tok/s (184) | +26 % |
| 32K to 64K | 48.3 tok/s (178) | 58.9 tok/s (175) | +22 % |
| 64K and over | 43.5 tok/s (8) | 50.8 tok/s (42) | +17 % |

Pooled over every turn: 227,209 output tokens in 3653 s of decode wall
(62.2 tok/s) against 198,717 in 3833 s (51.8 tok/s), +20 %. The same gain
shows in every bin by output length (32-64 tokens: 75.5 against 60.7;
1024 and over: 61.7 against 48.8), so it is not a change in the mix of
turns. It is the pipelined decode loop (patch 028) on an agentic workload;
the raw bench put it at +22 to +25 % for this preset
(`benchmarks/perf/2026-11-10/`). Time to first token is unchanged: median
0.66 s against 0.67 s, p90 2.5 s against 2.1 s, 1393 s in total against
1552 s.

### Per instance

| Instance | Wall | Patch | Files | Archived cell (wall, patch) | Same file | Local score | Archived local score |
|---|---:|---:|---|---|---|---|---|
| `astropy__astropy-12907` | 289 s | 1662 B | `astropy/modeling/separable.py` | 197 s, 1369 B | yes | install fail (M4) | install fail (M4) |
| `astropy__astropy-14182` | 110 s | 928 B | `astropy/io/ascii/rst.py` | 148 s, 936 B | yes | install fail (M4) | install fail (M4) |
| `astropy__astropy-14365` | 55 s | 617 B | `astropy/io/ascii/qdp.py` | 70 s, 617 B | yes | install fail (M4) | install fail (M4) |
| `astropy__astropy-14995` | 21 s | 626 B | `astropy/nddata/mixins/ndarithmetic.py` | 44 s, 626 B | yes | install fail (M4) | install fail (M4) |
| `astropy__astropy-6938` | 20 s | 514 B | `astropy/io/fits/fitsrec.py` | 89 s, 517 B | yes | install fail (M4) | install fail (M4) |
| `astropy__astropy-7746` | 79 s | 1008 B | `astropy/wcs/wcs.py` | 192 s, 1310 B | yes | install fail (M4) | install fail (M4) |
| `django__django-10914` | 200 s | 3908 B | `django/conf/global_settings.py`, `docs/howto/deployment/checklist.txt` + 2 more | 300 s, 3229 B | yes | **resolved** | **resolved** |
| `django__django-10924` | 263 s | 2295 B | `django/db/models/fields/__init__.py`, `django/forms/fields.py` | 102 s, 608 B | yes | applied; F2P 0/1, P2P 0/1 | **resolved** |
| `django__django-11001` | 90 s | 1761 B | `django/db/models/sql/compiler.py` | 112 s, 1296 B | yes | **resolved** | **resolved** |
| `django__django-11019` | 907 s | 7745 B | `django/forms/widgets.py` | 660 s, 5237 B | yes | applied; F2P 5/16, P2P 58/58 | applied; F2P 5/16, P2P 58/58 |
| `django__django-11039` | 48 s | 658 B | `django/core/management/commands/sqlmigrate.py` | 168 s, 2102 B | yes | **resolved** | **resolved** |
| `matplotlib__matplotlib-18869` | 164 s | 545 B | `lib/matplotlib/__init__.py` | 374 s, 719 B | yes | install fail (M4) | install fail (M4) |
| `mwaskom__seaborn-2848` | 704 s | 1606 B | `seaborn/axisgrid.py` | 265 s, 932 B | yes | applied; F2P 0/1, P2P 50/50 | applied; F2P 0/1, P2P 50/50 |
| `pallets__flask-4045` | 47 s | 491 B | `src/flask/blueprints.py` | 64 s, 513 B | yes | applied; F2P 0/2, P2P 50/50 | applied; F2P 1/2, P2P 50/50 |
| `psf__requests-1963` | 130 s | 760 B | `requests/sessions.py` | 152 s, 830 B | yes | applied; F2P 6/7, P2P 112/112 | applied; F2P 6/7, P2P 112/112 |
| `psf__requests-2148` | 224 s | 1744 B | `requests/models.py` | 442 s, 1395 B | yes | applied; F2P 9/10, P2P 118/118 | applied; F2P 9/10, P2P 114/118 |
| `pydata__xarray-3364` | 297 s | 4025 B | `xarray/core/concat.py` | 467 s, 3662 B | yes | applied; F2P 0/2, P2P 110/110 | applied; F2P 0/2, P2P 110/110 |
| `pylint-dev__pylint-5859` | 185 s | 650 B | `pylint/checkers/misc.py` | 158 s, 658 B | yes | **resolved** | **resolved** |
| `pylint-dev__pylint-6506` | 165 s | 1199 B | `pylint/lint/run.py` | 422 s, 1198 B | yes | applied; F2P 0/2, P2P 0/6 | applied; F2P 0/2, P2P 0/6 |
| `pytest-dev__pytest-11143` | 38 s | 582 B | `src/_pytest/assertion/rewrite.py` | 55 s, 557 B | yes | applied; F2P 1/1, P2P 110/114 | applied; F2P 1/1, P2P 110/114 |
| `pytest-dev__pytest-11148` | 179 s | 1241 B | `src/_pytest/pathlib.py` | 240 s, 1022 B | yes | applied; F2P 2/2, P2P 128/129 | applied; F2P 1/2, P2P 124/129 |
| `scikit-learn__scikit-learn-10297` | 90 s | 930 B | `sklearn/linear_model/ridge.py` | 55 s, 930 B | yes | install fail (M4) | install fail (M4) |
| `sphinx-doc__sphinx-10325` | 133 s | 2414 B | `doc/usage/extensions/autodoc.rst`, `sphinx/ext/autodoc/__init__.py` | 259 s, 1352 B | yes | **resolved** | **resolved** |
| `sphinx-doc__sphinx-10451` | 242 s | 2107 B | `sphinx/ext/autodoc/typehints.py` | 274 s, 2581 B | yes | applied; F2P 0/2, P2P 80/80 | applied; F2P 0/2, P2P 80/80 |
| `sympy__sympy-11400` | 87 s | 538 B | `sympy/printing/ccode.py` | 182 s, 950 B | yes | applied; F2P 0/2, P2P 29/29 | applied; F2P 0/2, P2P 0/29 |
| `sympy__sympy-11870` | 552 s | 2685 B | `sympy/functions/elementary/exponential.py`, `sympy/simplify/fu.py` + 1 more | 420 s, 849 B | yes | applied; F2P 0/1, P2P 58/58 | applied; F2P 0/1, P2P 58/58 |

- 1092 assistant turns across the 26 sessions against 1286 (median 35 per
  instance against 41), 227,209 output tokens against 198,717. Tool calls:
  `read` 401, `edit` 305, `bash` 216, `grep` 137, `todo` 47, `glob` 37,
  `write` 1, `eval` 0 (the archived cell: `read` 475, `bash` 367, `edit`
  247). Largest prompt 89K tokens (`seaborn-2848`). The session store shows
  196 of the 1144 tool calls without omp's `i` argument (135 of 1332 in the
  archived cell); omp ran them, and no tool result reports a missing
  argument. The proxy's `BACKFILL_INTENT` was off.
- Fewer, longer turns and more edits per bash call than the archived cell.
  With one rollout per tree this cannot be attributed to the position fix
  (patch 033): the archived cell numbered each follow-up turn's new tokens
  from zero, this one numbers them from the prefix length, and both produce
  26/26 patches on the same files.
- `audit_rollouts.py` flags 5 of the 26 sessions, all for `pip install`
  attempts (`pytest`, `pyerfa`) that fail inside the sandbox (no DNS); no
  web or history hit. 21 of the 26 venvs failed to build on the box (old
  Python pins), so those sessions ran on the bash prompt without a venv.

## Files

- `predictions.jsonl` (rows carry `scaffold: omp/18.12.2`), `predictions/<inst>.diff`
- `scores.jsonl` — `score_local.py` verdicts; `score_local.log` (local only)
- `sessions/<inst>/*.jsonl` — omp's pi-row session store (archived)
- `logs/<inst>.log` — omp `--mode json` event stream (local only)
- `launch.log`, `proxy.log`, `rollout.log`, `omp-agent/` (local only)
