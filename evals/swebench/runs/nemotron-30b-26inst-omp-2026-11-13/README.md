# nemotron-30b, 26-instance sealed SWE-bench Lite cell on omp: 10/26 non-empty, local scorer 1/26

The qwen36 scorecard's 26 instances (`../qwen36-omp-final-26inst-2026-11-12/`) on `mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit`
served by the `nemotron-30b` preset, SGLang v0.5.23 with the patch series through 041, omp 18.12.2,
thinking off through the proxy, sealed harness, 900 s per instance, context window 131072:
`PRESET=nemotron-30b MODEL_KEY=nemotron-30b TIMEOUT=900 INSTANCE_IDS="<the 26>" bash evals/swebench/smoke.sh`. One server served all 26 in sequence with no traceback.

## Result

- Non-empty patches: **10/26**; 7 touch a file the qwen36 omp cell's patch touches.
- At the 900 s cap: 2 instances. Rollout wall: 126 min.
- `score_local.py`: **1/26** resolved (pylint-dev__pylint-5859). The scorer could not
  build the environment for 3 instances, so the count is a floor. The official number needs
  the Docker harness on the 3090: `../../exports/nemotron-30b-omp-26inst-predictions.jsonl`.

## Per instance

| Instance | Rollout s | Patch bytes | Local scorer |
|---|---:|---:|---|
| astropy__astropy-12907 | 303 | 0 | empty patch |
| astropy__astropy-14182 | 17 | 0 | empty patch |
| astropy__astropy-14365 | 372 | 0 | empty patch |
| astropy__astropy-14995 | 174 | 0 | empty patch |
| astropy__astropy-6938 | 17 | 0 | empty patch |
| astropy__astropy-7746 | 192 | 0 | empty patch |
| django__django-10914 | 220 | 22843 | f2p 0/1, p2p 0/98 |
| django__django-10924 | 247 | 0 | empty patch |
| django__django-11001 | 242 | 0 | empty patch |
| django__django-11019 | 77 | 38632 | f2p 0/16, p2p 0/58 |
| django__django-11039 | 108 | 0 | empty patch |
| matplotlib__matplotlib-18869 | 174 | 0 | empty patch |
| mwaskom__seaborn-2848 | 366 | 1788 | f2p 0/1, p2p 50/50 |
| pallets__flask-4045 | 165 | 0 | empty patch |
| psf__requests-1963 | 315 | 66357 | install_fail |
| psf__requests-2148 | 274 | 50678 | install_fail |
| pydata__xarray-3364 | 440 | 18633 | f2p 0/2, p2p 33/110 |
| pylint-dev__pylint-5859 | 412 | 789 | resolved |
| pylint-dev__pylint-6506 | 284 | 10452 | f2p 0/2, p2p 0/6 |
| pytest-dev__pytest-11143 | 169 | 9432 | f2p 0/1, p2p 110/114 |
| pytest-dev__pytest-11148 | 99 | 0 | empty patch |
| scikit-learn__scikit-learn-10297 | 804 | 112110 | install_fail |
| sphinx-doc__sphinx-10325 | 22 | 0 | empty patch |
| sphinx-doc__sphinx-10451 | 248 | 0 | empty patch |
| sympy__sympy-11400 | 905 | 0 | empty patch |
| sympy__sympy-11870 | 905 | 0 | empty patch |
