# qwen3-moe, 26-instance sealed SWE-bench Lite cell on omp: 11/26 non-empty, local scorer 1/26

The qwen36 scorecard's 26 instances (`../qwen36-omp-final-26inst-2026-11-12/`) on `mlx-community/Qwen3-30B-A3B-4bit-DWQ`
served by the `qwen3-moe` preset, SGLang v0.5.23 with the patch series through 041, omp 18.12.2,
thinking off through the proxy, sealed harness, 900 s per instance, context window 131072:
`PRESET=qwen3-moe MODEL_KEY=qwen3-moe TIMEOUT=900 INSTANCE_IDS="<the 26>" bash evals/swebench/smoke.sh`. One server served all 26 in sequence with no traceback.

## Result

- Non-empty patches: **11/26**; 9 touch a file the qwen36 omp cell's patch touches.
- At the 900 s cap: 2 instances. Rollout wall: 58 min.
- `score_local.py`: **1/26** resolved (django__django-11039). The scorer could not
  build the environment for 4 instances, so the count is a floor. The official number needs
  the Docker harness on the 3090: `../../exports/qwen3-moe-omp-26inst-predictions.jsonl`.

## Per instance

| Instance | Rollout s | Patch bytes | Local scorer |
|---|---:|---:|---|
| astropy__astropy-12907 | 123 | 0 | empty patch |
| astropy__astropy-14182 | 76 | 1147 | install_fail |
| astropy__astropy-14365 | 34 | 0 | empty patch |
| astropy__astropy-14995 | 66 | 0 | empty patch |
| astropy__astropy-6938 | 9 | 0 | empty patch |
| astropy__astropy-7746 | 38 | 0 | empty patch |
| django__django-10914 | 13 | 0 | empty patch |
| django__django-10924 | 54 | 2572 | f2p 0/1, p2p 0/1 |
| django__django-11001 | 207 | 5294 | f2p 0/2, p2p 0/118 |
| django__django-11019 | 38 | 0 | empty patch |
| django__django-11039 | 907 | 122407 | resolved |
| matplotlib__matplotlib-18869 | 54 | 618 | install_fail |
| mwaskom__seaborn-2848 | 902 | 0 | empty patch |
| pallets__flask-4045 | 94 | 0 | empty patch |
| psf__requests-1963 | 51 | 916 | install_fail |
| psf__requests-2148 | 17 | 929 | install_fail |
| pydata__xarray-3364 | 172 | 8497 | f2p 0/2, p2p 0/110 |
| pylint-dev__pylint-5859 | 33 | 0 | empty patch |
| pylint-dev__pylint-6506 | 27 | 734 | f2p 0/2, p2p 0/6 |
| pytest-dev__pytest-11143 | 73 | 0 | empty patch |
| pytest-dev__pytest-11148 | 241 | 0 | empty patch |
| scikit-learn__scikit-learn-10297 | 23 | 0 | empty patch |
| sphinx-doc__sphinx-10325 | 94 | 742 | f2p 0/1, p2p 0/5 |
| sphinx-doc__sphinx-10451 | 43 | 0 | empty patch |
| sympy__sympy-11400 | 25 | 0 | empty patch |
| sympy__sympy-11870 | 73 | 703 | f2p 0/1, p2p 0/58 |
