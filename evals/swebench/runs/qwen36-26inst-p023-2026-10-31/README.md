# qwen36 26-instance SWE-bench Lite sweep on patches 021–023 — 26/26 non-empty patches

The 26-instance set of the scorecard (`../../exports/qwen36-predictions.jsonl`,
21/26 non-empty on the previous stack pin, 9/26 resolved on the 3090's Docker
harness) rerun on SGLang v0.5.23 + patches 002–023 with the preset default
`mlx-community/Qwen3.6-35B-A3B-4bit`: `scripts/launch.sh qwen36` as the
harness launches it (`CTX=131072`, turboquant, chunk 2048, `MEM_FRAC=0.5`,
radix cache on), opencode 1.15.4 through `no_thinking_proxy.py`, 900 s per
instance, venvs where they build. One server served all 26 instances in
sequence (a quoting slip handed the whole list to one `smoke.sh`): no
jetsam, no server error, OOM guard zero warnings, 87 minutes wall.

## Result

26/26 non-empty patches (the previous stack: 21/26). The five instances that
gave up before all produced a patch. Per-instance wall median 162 s,
max 748 s (`sympy__sympy-11870`). Resolution needs the Docker
harness on the 3090: `../../exports/qwen36-p023-predictions.jsonl` is the
file to score; the Docker column below is the previous stack's scoring.

| Instance | Docker (previous stack) | Previous patch | This run | Wall | Note |
|---|---|---:|---:|---:|---|
| `astropy__astropy-12907` | resolved | 506 B | 506 B | 55 s | byte-identical |
| `astropy__astropy-14182` | unresolved | 579 B | 579 B | 62 s | byte-identical |
| `astropy__astropy-14365` | unresolved | 476 B | 476 B | 61 s | byte-identical |
| `astropy__astropy-14995` | resolved | 628 B | 1187 B | 77 s | different patch |
| `astropy__astropy-6938` | unresolved | 516 B | 516 B | 33 s | byte-identical |
| `astropy__astropy-7746` | unresolved | 567 B | 1042 B | 173 s | different patch |
| `django__django-10914` | resolved | 2563 B | 3521 B | 261 s | different patch |
| `django__django-10924` | unresolved | 1714 B | 622 B | 84 s | different patch |
| `django__django-11001` | resolved | 1368 B | 754 B | 201 s | different patch |
| `django__django-11019` | empty_patch | 0 B | 726 B | 357 s | new patch (empty before) |
| `django__django-11039` | resolved | 826 B | 716 B | 63 s | different patch |
| `matplotlib__matplotlib-18869` | unresolved | 758 B | 1956 B | 163 s | different patch |
| `mwaskom__seaborn-2848` | unresolved | 710 B | 2842 B | 460 s | different patch |
| `pallets__flask-4045` | empty_patch | 0 B | 488 B | 144 s | new patch (empty before) |
| `psf__requests-1963` | unresolved | 521 B | 412 B | 162 s | different patch |
| `psf__requests-2148` | empty_patch | 0 B | 1045 B | 152 s | new patch (empty before) |
| `pydata__xarray-3364` | unresolved | 1178 B | 3291 B | 338 s | different patch |
| `pylint-dev__pylint-5859` | resolved | 656 B | 672 B | 185 s | different patch |
| `pylint-dev__pylint-6506` | unresolved | 1243 B | 676 B | 83 s | different patch |
| `pytest-dev__pytest-11143` | resolved | 557 B | 485 B | 123 s | different patch |
| `pytest-dev__pytest-11148` | unresolved | 639 B | 501 B | 168 s | different patch |
| `scikit-learn__scikit-learn-10297` | resolved | 932 B | 932 B | 310 s | byte-identical |
| `sphinx-doc__sphinx-10325` | resolved | 1038 B | 1515 B | 354 s | different patch |
| `sphinx-doc__sphinx-10451` | empty_patch | 0 B | 2721 B | 235 s | new patch (empty before) |
| `sympy__sympy-11400` | unresolved | 704 B | 341 B | 103 s | different patch |
| `sympy__sympy-11870` | empty_patch | 0 B | 830 B | 748 s | new patch (empty before) |

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`, `logs/<inst>.log`
- `launch.log`, `proxy.log`, `rollout.log`, `smoke.run.log`
