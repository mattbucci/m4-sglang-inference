# qwen36, 26-instance sealed SWE-bench Lite cell on the omp scaffold — 26/26 non-empty, 25/26 on the file the opencode cell patched

The scorecard's 26 instances (`../qwen36-26inst-p023-2026-10-31/`) on the
preset default `mlx-community/Qwen3.6-35B-A3B-4bit` (`scripts/launch.sh
qwen36`: CTX=131072, turboquant, chunk 2048, mem-fraction 0.5, radix cache on)
on SGLang v0.5.23 + the 17 patches, driven by omp (oh-my-pi) 18.12.2 instead
of opencode: `omp --model sglang/qwen36 --yolo --no-title -p --mode json`,
tool set `read bash edit eval glob grep wait todo write` (`web_search` and
`task` off), thinking off (`--thinking off` plus the proxy's
`enable_thinking=false`), context window 131072, output budget 8192, 900 s
per instance. Sealed: omp and its bash children in the localhost-only
sandbox, ancestor-only checkout. One server served all 26 in sequence: no
jetsam, OOM guard zero warnings, 99 min of rollout wall. This is the
validation cell for the switch to omp as the single scaffold lane.

## Result

26/26 non-empty patches, 0 timeouts, 25/26 touching a file the opencode cell's
patch touches. The exception is `pylint-6506`, where omp patches
`pylint/lint/run.py` (the upstream fix's file) and the opencode cell patched
`pylint/__init__.py`. Per-instance wall median 187 s, max 660 s
(`django-11019`), 5912 s in total against the opencode cell's 5153 s. Patches
are larger (median 943 B against 721 B): `django-10914` also edits
three docs files, `django-11039` also adds a test. `score_local.py` (per-instance venv, native pytest) resolves **6/26**
(6 of the 18 instances the box can build: the four django instances,
`pylint-5859`, `sphinx-10325`), against 5/26 for the opencode lane's
predictions on the same scorer; every one of the 18 scorable patches
applies, and 9 of the 12 misses pass part of their FAIL_TO_PASS set or keep
PASS_TO_PASS clean. The official number needs the Docker harness on the
3090: `../../exports/qwen36-omp-26inst-predictions.jsonl` (model name
`sglang/qwen36-omp`).

| Instance | Wall | Patch | Files | opencode cell (wall, patch) | Same file | Local score |
|---|---:|---:|---|---|---|---|
| `astropy__astropy-12907` | 197 s | 1369 B | `astropy/modeling/separable.py` | 55 s, 506 B | yes | install fail (M4) |
| `astropy__astropy-14182` | 148 s | 936 B | `astropy/io/ascii/rst.py` | 62 s, 579 B | yes | install fail (M4) |
| `astropy__astropy-14365` | 70 s | 617 B | `astropy/io/ascii/qdp.py` | 61 s, 476 B | yes | install fail (M4) |
| `astropy__astropy-14995` | 44 s | 626 B | `astropy/nddata/mixins/ndarithmetic.py` | 77 s, 1187 B | yes | install fail (M4) |
| `astropy__astropy-6938` | 89 s | 517 B | `astropy/io/fits/fitsrec.py` | 33 s, 516 B | yes | install fail (M4) |
| `astropy__astropy-7746` | 192 s | 1310 B | `astropy/wcs/wcs.py` | 173 s, 1042 B | yes | install fail (M4) |
| `django__django-10914` | 300 s | 3229 B | `django/conf/global_settings.py` + 3 docs files | 261 s, 3521 B | yes | **resolved** |
| `django__django-10924` | 102 s | 608 B | `django/forms/fields.py` | 84 s, 622 B | yes | **resolved** |
| `django__django-11001` | 112 s | 1296 B | `django/db/models/sql/compiler.py` | 201 s, 754 B | yes | **resolved** |
| `django__django-11019` | 660 s | 5237 B | `django/forms/widgets.py` | 357 s, 726 B | yes | applied; F2P 5/16, P2P 58/58 |
| `django__django-11039` | 168 s | 2102 B | `django/core/management/commands/sqlmigrate.py`, `tests/migrations/test_commands.py` | 63 s, 716 B | yes | **resolved** |
| `matplotlib__matplotlib-18869` | 374 s | 719 B | `lib/matplotlib/__init__.py` | 163 s, 1956 B | yes | install fail (M4) |
| `mwaskom__seaborn-2848` | 265 s | 932 B | `seaborn/axisgrid.py` | 460 s, 2842 B | yes | applied; F2P 0/1, P2P 50/50 |
| `pallets__flask-4045` | 64 s | 513 B | `src/flask/blueprints.py` | 144 s, 488 B | yes | applied; F2P 1/2, P2P 50/50 |
| `psf__requests-1963` | 152 s | 830 B | `requests/sessions.py` | 162 s, 412 B | yes | applied; F2P 6/7, P2P 112/112 |
| `psf__requests-2148` | 442 s | 1395 B | `requests/models.py` | 152 s, 1045 B | yes | applied; F2P 9/10, P2P 114/118 |
| `pydata__xarray-3364` | 467 s | 3662 B | `xarray/core/concat.py` | 338 s, 3291 B | yes | applied; F2P 0/2, P2P 110/110 |
| `pylint-dev__pylint-5859` | 158 s | 658 B | `pylint/checkers/misc.py` | 185 s, 672 B | yes | **resolved** |
| `pylint-dev__pylint-6506` | 422 s | 1198 B | `pylint/lint/run.py` | 83 s, 676 B | no (`pylint/__init__.py`) | applied; F2P 0/2, P2P 0/6 |
| `pytest-dev__pytest-11143` | 55 s | 557 B | `src/_pytest/assertion/rewrite.py` | 123 s, 485 B | yes | applied; F2P 1/1, P2P 110/114 |
| `pytest-dev__pytest-11148` | 240 s | 1022 B | `src/_pytest/pathlib.py` | 168 s, 501 B | yes | applied; F2P 1/2, P2P 124/129 |
| `scikit-learn__scikit-learn-10297` | 55 s | 930 B | `sklearn/linear_model/ridge.py` | 310 s, 932 B | yes | install fail (M4) |
| `sphinx-doc__sphinx-10325` | 259 s | 1352 B | `sphinx/ext/autodoc/__init__.py` | 354 s, 1515 B | yes | **resolved** |
| `sphinx-doc__sphinx-10451` | 274 s | 2581 B | `sphinx/ext/autodoc/typehints.py` | 235 s, 2721 B | yes | applied; F2P 0/2, P2P 80/80 |
| `sympy__sympy-11400` | 182 s | 950 B | `sympy/printing/ccode.py` | 103 s, 341 B | yes | applied; F2P 0/2, P2P 0/29 |
| `sympy__sympy-11870` | 420 s | 849 B | `sympy/simplify/trigsimp.py` | 748 s, 830 B | yes | applied; F2P 0/1, P2P 58/58 |

- 1286 assistant turns across the 26 sessions; tool calls: `read` 475,
  `bash` 367, `edit` 247, `grep` 136, `glob` 54, `todo` 53, `eval` 0,
  `write` 0. Largest prompt 68K tokens (`django-11019`). Every tool call
  carried omp's required `i` argument on its own (the proxy's
  `BACKFILL_INTENT` was off and would not have fired).
- `audit_rollouts.py` flags 5 of the 26 sessions, all for `pip install`
  attempts (`pytest`, `astropy`, `httpbin-core`) that fail inside the
  sandbox (no DNS); no web or history hit. 20 of the 26 venvs failed to
  build on the box (old Python pins), so those sessions ran on the bash
  prompt without a venv, as in the opencode cell.

## Files

- `predictions.jsonl` (rows carry `scaffold: omp/18.12.2`), `predictions/<inst>.diff`
- `scores.jsonl` — `score_local.py` verdicts; `score_local.log` (local only)
- `sessions/<inst>/*.jsonl` — omp's pi-row session store (archived)
- `logs/<inst>.log` — omp `--mode json` event stream (local only)
- `launch.log`, `proxy.log`, `rollout.log`, `omp-agent/` (local only)
