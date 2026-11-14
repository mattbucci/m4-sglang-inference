# coder-30b, 26-instance sealed SWE-bench Lite cell on omp: 25/26 non-empty, local scorer 1/26

The qwen36 scorecard's 26 instances (`../qwen36-omp-final-26inst-2026-11-12/`) on `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ`
served by the `coder-30b` preset, SGLang v0.5.23 with the patch series through 041, omp 18.12.2,
thinking off through the proxy, sealed harness, 900 s per instance, context window 131072:
`PRESET=coder-30b MODEL_KEY=coder-30b TIMEOUT=900 INSTANCE_IDS="<the 26>" bash evals/swebench/smoke.sh`. One server served all 26 in sequence with no traceback.

## Result

- Non-empty patches: **25/26**; 24 touch a file the qwen36 omp cell's patch touches.
- At the 900 s cap: 4 instances. Rollout wall: 147 min.
- `score_local.py`: **1/26** resolved (django__django-11039). The scorer could not
  build the environment for 8 instances, so the count is a floor. The official number needs
  the Docker harness on the 3090: `../../exports/coder-30b-omp-26inst-predictions.jsonl`.

## Per instance

| Instance | Rollout s | Patch bytes | Local scorer |
|---|---:|---:|---|
| astropy__astropy-12907 | 261 | 15718 | install_fail |
| astropy__astropy-14182 | 151 | 728 | install_fail |
| astropy__astropy-14365 | 114 | 2811 | install_fail |
| astropy__astropy-14995 | 151 | 626 | install_fail |
| astropy__astropy-6938 | 42 | 514 | install_fail |
| astropy__astropy-7746 | 131 | 964 | install_fail |
| django__django-10914 | 918 | 1684 | f2p 1/1, p2p 97/98 |
| django__django-10924 | 222 | 1778 | f2p 0/1, p2p 0/1 |
| django__django-11001 | 238 | 3312 | f2p 0/2, p2p 0/118 |
| django__django-11019 | 183 | 2490 | f2p 2/16, p2p 58/58 |
| django__django-11039 | 224 | 2355 | resolved |
| matplotlib__matplotlib-18869 | 251 | 2371 | install_fail |
| mwaskom__seaborn-2848 | 329 | 1965 | f2p 0/1, p2p 50/50 |
| pallets__flask-4045 | 903 | 40305 | f2p 0/2, p2p 0/50 |
| psf__requests-1963 | 132 | 4545 | f2p 6/7, p2p 112/112 |
| psf__requests-2148 | 392 | 617 | f2p 8/10, p2p 118/118 |
| pydata__xarray-3364 | 546 | 0 | empty patch |
| pylint-dev__pylint-5859 | 360 | 638 | f2p 1/1, p2p 9/10 |
| pylint-dev__pylint-6506 | 172 | 867 | f2p 0/2, p2p 0/6 |
| pytest-dev__pytest-11143 | 116 | 2035 | f2p 1/1, p2p 110/114 |
| pytest-dev__pytest-11148 | 156 | 832 | f2p 0/2, p2p 122/129 |
| scikit-learn__scikit-learn-10297 | 912 | 2868 | install_fail |
| sphinx-doc__sphinx-10325 | 329 | 125325 | f2p 0/1, p2p 0/5 |
| sphinx-doc__sphinx-10451 | 386 | 6112 | f2p 0/2, p2p 79/80 |
| sympy__sympy-11400 | 277 | 339 | f2p 0/2, p2p 29/29 |
| sympy__sympy-11870 | 905 | 5189 | f2p 0/1, p2p 0/58 |
