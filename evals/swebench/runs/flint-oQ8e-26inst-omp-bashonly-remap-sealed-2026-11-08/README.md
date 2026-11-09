# flint (oQ8e), 26-instance sealed SWE-bench Lite cell on omp, bash-only — 5/26 non-empty, every instance at the 1800 s cap, 0/26 on the local scorer

The scorecard's 26 instances (`../qwen36-26inst-omp-2026-11-08/` on the same
harness) on Flint oQ8e served as `flint` (CTX=65536 with a 65536-slot pool,
chunk 1024, 2 GB MLX cache, model sampling, radix cache on, mem-fraction
0.7) on SGLang v0.5.23 + the 17 patches, driven by omp 18.12.2 with every
tool but `bash` disabled and the bash prompt style, the proxy running
`BACKFILL_INTENT=1 REMAP_INTENT=1` (derives omp's `i` when missing, clamps
it to 200 characters), output budget 4096, 1800 s per instance, sealed
(localhost-only sandbox, ancestor-only checkout). 20 of the 26 venvs failed
to build on the box, so those sessions ran on the bash prompt without one,
as in the other cells. One server served all 26 in sequence: no jetsam, OOM
guard zero warnings, 13.0 h wall. The configuration was chosen by the three
smokes `../flint-oQ8e-omp-64k-sealed-2026-11-08/`,
`../flint-oQ8e-omp-bashonly-64k-sealed-2026-11-08/` and
`../flint-oQ8e-omp-remap-64k-sealed-2026-11-08/`.

## Result

5/26 non-empty patches, each on the file qwen36's patch touches on the same
harness; 21 empties. All 26 instances ran to the 1800 s cap: Flint never
ends its turn under omp (4439 of 4443 responses are tool calls, the other 4
hit the 4096-token output cap), so the cap is the cell's wall. `score_local.py`
(per-instance venv, native pytest) resolves 0/26: three patches apply and
fail their FAIL_TO_PASS tests, two are on astropy instances the box cannot
build. qwen36 on the same harness: 26/26 non-empty in 99 min.

| Instance | Wall | Patch | Files | Turns | Local score |
|---|---:|---:|---|---:|---|
| `astropy__astropy-12907` | cap | 3648 B | `modeling/separable.py` + `core.py`, `mappings.py`, `projections.py` | 151 | install fail (M4) |
| `astropy__astropy-14182` | cap | 0 B | | 163 | |
| `astropy__astropy-14365` | cap | 0 B | | 214 | |
| `astropy__astropy-14995` | cap | 817 B | `nddata/mixins/ndarithmetic.py` | 145 | install fail (M4) |
| `astropy__astropy-6938` | cap | 0 B | | 234 | |
| `astropy__astropy-7746` | cap | 0 B | | 127 | |
| `django__django-10914` | cap | 0 B | | 125 | |
| `django__django-10924` | cap | 0 B | | 125 | |
| `django__django-11001` | cap | 0 B | | 165 | |
| `django__django-11019` | cap | 3227 B | `django/forms/widgets.py` | 147 | applied; F2P 0/16, P2P 10/58 |
| `django__django-11039` | cap | 0 B | | 224 | |
| `matplotlib__matplotlib-18869` | cap | 0 B | | 150 | |
| `mwaskom__seaborn-2848` | cap | 0 B | | 90 | |
| `pallets__flask-4045` | cap | 648 B | `src/flask/blueprints.py` | 225 | applied; F2P 0/2, P2P 41/50 |
| `psf__requests-1963` | cap | 0 B | | 166 | |
| `psf__requests-2148` | cap | 0 B | | 298 | |
| `pydata__xarray-3364` | cap | 0 B | | 138 | |
| `pylint-dev__pylint-5859` | cap | 0 B | | 264 | |
| `pylint-dev__pylint-6506` | cap | 0 B | | 144 | |
| `pytest-dev__pytest-11143` | cap | 0 B | | 196 | |
| `pytest-dev__pytest-11148` | cap | 0 B | | 158 | |
| `scikit-learn__scikit-learn-10297` | cap | 0 B | | 292 | |
| `sphinx-doc__sphinx-10325` | cap | 3647 B | `sphinx/ext/autodoc/__init__.py` | 152 | applied; F2P 0/1, P2P 5/5 |
| `sphinx-doc__sphinx-10451` | cap | 0 B | | 137 | |
| `sympy__sympy-11400` | cap | 0 B | | 114 | |
| `sympy__sympy-11870` | cap | 0 B | | 99 | |

- **Where the budget goes.** 4471 bash calls, 90–298 per instance, 47–58K
  tokens of context at the cap. On 14 of the 21 empty instances Flint
  never attempted an edit: it reads the code in `sed -n` and `grep` windows
  for the whole half hour. Where it did write (`sed -i`, heredocs: 121 write
  commands in the cell), it wrote to mistyped paths
  (`astropy__astropy__6938/`, `django__django__`), to scratch files it then
  removed (`requests-2148`, 57 writes, `rm -f check_tmp.py`), or to the right
  file followed by an edit back (`astropy-14365`, the instance it patches in
  every smoke). Test runs are concentrated: 196 pytest invocations on
  `pytest-11143`, 157 on `pytest-11148`, none on 17 instances.
- **The harness repairs held.** 0 calls rejected by omp; 705 of the 4471
  intents were clamped to 200 characters; every call carried `command`.
- **Runaway turns: 4 in 4443** (each 4096 output tokens: `astropy-12907`,
  `astropy-6938` ×2, `matplotlib-18869`), down from 9 in 170 on the stack
  before patch 022.
- `audit_rollouts.py` flags 4 sessions: `git show <base sha>` and `git log
  --all`/`git branch -a` in the ancestor-only checkout (nothing past the base
  commit to read) and one `pip install pytest` that fails inside the
  sandbox. No session saw the fix.
- The three patches that apply are regressions or no-ops: `django-11019`
  rewrites the `Media` merge and breaks 48 of 58 passing tests,
  `flask-4045` adds a blueprint-name check that fails the two new tests and
  9 existing ones, `sphinx-10325` edits `inherited_members` handling without
  passing the new test.

## Files

- `predictions.jsonl` (rows carry `scaffold: omp/18.12.2`), `predictions/<inst>.diff`
- `scores.jsonl` — `score_local.py` verdicts; `score_local.log` (local only)
- `sessions/<inst>/*.jsonl` — omp's pi-row session store (archived)
- `logs/`, `launch.log`, `proxy.log`, `rollout.log`, `omp-agent/` (local only)
