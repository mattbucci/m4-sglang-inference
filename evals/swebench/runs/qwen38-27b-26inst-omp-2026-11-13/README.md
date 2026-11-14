# qwen38-27b, 26-instance sealed SWE-bench Lite cell on omp: 24/26 non-empty, local scorer 6/26

The qwen36 scorecard's 26 instances (`../qwen36-omp-final-26inst-2026-11-12/`) on `mlx-community/Qwen3.8-27B-4bit`
served by the `qwen36-27b` preset, SGLang v0.5.23 with the patch series through 041, omp 18.12.2,
thinking off through the proxy, sealed harness, 1800 s per instance, context window 131072:
`MODEL=mlx-community/Qwen3.8-27B-4bit PRESET=qwen36-27b MODEL_KEY=qwen38-27b TIMEOUT=1800 INSTANCE_IDS="<the 26>" bash evals/swebench/smoke.sh`. One server served all 26 in sequence with no traceback.

## Result

- Non-empty patches: **24/26**; 20 touch a file the qwen36 omp cell's patch touches.
- At the 1800 s cap: 11 instances. Rollout wall: 476 min.
- `score_local.py`: **6/26** resolved (django__django-10914, django__django-10924, django__django-11001, django__django-11039, pylint-dev__pylint-5859, sympy__sympy-11400). The scorer could not
  build the environment for 7 instances, so the count is a floor. The official number needs
  the Docker harness on the 3090: `../../exports/qwen38-27b-omp-26inst-predictions.jsonl`.

## Per instance

| Instance | Rollout s | Patch bytes | Local scorer |
|---|---:|---:|---|
| astropy__astropy-12907 | 1805 | 0 | empty patch |
| astropy__astropy-14182 | 1214 | 874 | install_fail |
| astropy__astropy-14365 | 1806 | 3318 | install_fail |
| astropy__astropy-14995 | 278 | 1184 | install_fail |
| astropy__astropy-6938 | 351 | 567 | install_fail |
| astropy__astropy-7746 | 1804 | 5830 | install_fail |
| django__django-10914 | 456 | 2823 | resolved |
| django__django-10924 | 380 | 622 | resolved |
| django__django-11001 | 193 | 543 | resolved |
| django__django-11019 | 1807 | 1411 | f2p 2/16, p2p 52/58 |
| django__django-11039 | 168 | 658 | resolved |
| matplotlib__matplotlib-18869 | 1808 | 3556 | install_fail |
| mwaskom__seaborn-2848 | 1802 | 2673 | f2p 0/1, p2p 50/50 |
| pallets__flask-4045 | 354 | 554 | f2p 1/2, p2p 50/50 |
| psf__requests-1963 | 1803 | 3289 | f2p 6/7, p2p 112/112 |
| psf__requests-2148 | 1469 | 2858 | f2p 9/10, p2p 118/118 |
| pydata__xarray-3364 | 1804 | 18306 | f2p 0/2, p2p 40/110 |
| pylint-dev__pylint-5859 | 635 | 846 | resolved |
| pylint-dev__pylint-6506 | 1804 | 2725 | f2p 0/2, p2p 0/6 |
| pytest-dev__pytest-11143 | 289 | 959 | f2p 1/1, p2p 110/114 |
| pytest-dev__pytest-11148 | 1803 | 4169 | f2p 1/2, p2p 127/129 |
| scikit-learn__scikit-learn-10297 | 261 | 1478 | install_fail |
| sphinx-doc__sphinx-10325 | 1804 | 0 | empty patch |
| sphinx-doc__sphinx-10451 | 880 | 2187 | f2p 0/2, p2p 80/80 |
| sympy__sympy-11400 | 873 | 1041 | resolved |
| sympy__sympy-11870 | 910 | 1544 | f2p 0/1, p2p 58/58 |
