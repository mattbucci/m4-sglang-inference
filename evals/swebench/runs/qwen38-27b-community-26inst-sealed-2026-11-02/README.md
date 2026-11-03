# Qwen3.8-27B community 4-bit, 26-instance sealed SWE-bench Lite cell — 22/26 non-empty, 21/26 on the canonical file

The qwen36 scorecard's 26 instances (`../qwen36-26inst-p023-2026-10-31/`) on
`mlx-community/Qwen3.8-27B-4bit` served by the `qwen36-27b` preset (`MODEL=`
override, CTX=131072, `CHUNKED=2048`, mem-fraction 0.5, radix on, thinking
always on) on SGLang v0.5.23 + the 17 patches, through the sealed harness
(`webfetch`/`websearch`/`task` off, opencode and its bash children in the
localhost-only sandbox, ancestor-only checkout), opencode `limit.context`
131072, 1800 s per instance. The dense comparator for
`../qwopus38-27b-coder-flash-26inst-sealed-2026-11-02/` (same instances,
same settings, same day). One server served all 26 in sequence: no jetsam,
OOM guard zero warnings, 8.8 h wall.

## Result

22/26 non-empty patches, 21/26 touching the file the qwen36 cell's patch
touches. 12/26 hit the 1800 s cap; 8 of those still left a diff. The four
empties (`seaborn-2848`, `xarray-3364`, `pylint-6506`, `sphinx-10451`) are
timeouts. Against the Qwopus coding fine-tune the two dense builds trade
instances: stock patches `django-10924` and `pytest-11148` where Qwopus
timed out empty; Qwopus patches `xarray-3364`, `pylint-6506` and
`sphinx-10451` where stock did. Resolution needs the Docker harness on the
3090: `../../exports/qwen38-27b-community-26inst-predictions.jsonl` (model
name `sglang/qwen38-27b-community`).

| Instance | Wall | Patch | Target | Qwopus (same instance) | qwen36 cell |
|---|---:|---:|---|---|---:|
| `astropy__astropy-12907` | 717 s | 504 B | canonical file | 504 B, canonical file | 506 B |
| `astropy__astropy-14182` | 1811 s (cap) | 552 B | canonical file | 2352 B, canonical file | 579 B |
| `astropy__astropy-14365` | 279 s | 617 B | canonical file | 1641 B, canonical file | 476 B |
| `astropy__astropy-14995` | 509 s | 1418 B | canonical file | 626 B, canonical file | 1187 B |
| `astropy__astropy-6938` | 178 s | 570 B | canonical file | 558 B, canonical file | 516 B |
| `astropy__astropy-7746` | 560 s | 1832 B | canonical file | 1135 B, canonical file | 1042 B |
| `django__django-10914` | 374 s | 625 B | canonical file | 625 B, canonical file | 3521 B |
| `django__django-10924` | 861 s | 2107 B | canonical file | 0 B, empty | 622 B |
| `django__django-11001` | 1807 s (cap) | 1930 B | canonical file | 4825 B, canonical file | 754 B |
| `django__django-11019` | 1808 s (cap) | 3238 B | canonical file | 3474 B, canonical file | 726 B |
| `django__django-11039` | 1808 s (cap) | 2628 B | canonical file | 658 B, canonical file | 716 B |
| `matplotlib__matplotlib-18869` | 638 s | 696 B | canonical file | 2600 B, canonical file | 1956 B |
| `mwaskom__seaborn-2848` | 1804 s (cap) | 0 B | empty | 0 B, empty | 2842 B |
| `pallets__flask-4045` | 1152 s | 979 B | canonical file | 2133 B, canonical file | 488 B |
| `psf__requests-1963` | 985 s | 457 B | canonical file | 517 B, canonical file | 412 B |
| `psf__requests-2148` | 870 s | 2450 B | canonical file | 1228 B, canonical file | 1045 B |
| `pydata__xarray-3364` | 1803 s (cap) | 0 B | empty | 1885 B, canonical file | 3291 B |
| `pylint-dev__pylint-5859` | 1046 s | 1254 B | canonical file | 1112 B, canonical file | 672 B |
| `pylint-dev__pylint-6506` | 1804 s (cap) | 0 B | empty | 796 B, canonical file | 676 B |
| `pytest-dev__pytest-11143` | 1557 s | 557 B | canonical file | 713 B, canonical file | 485 B |
| `pytest-dev__pytest-11148` | 1804 s (cap) | 1042 B | canonical file | 0 B, empty | 501 B |
| `scikit-learn__scikit-learn-10297` | 211 s | 930 B | canonical file | 1564 B, canonical file | 932 B |
| `sphinx-doc__sphinx-10325` | 1805 s (cap) | 1709 B | canonical file | 9398 B, canonical file | 1515 B |
| `sphinx-doc__sphinx-10451` | 1806 s (cap) | 0 B | empty | 3853 B, canonical file | 2721 B |
| `sympy__sympy-11400` | 1805 s (cap) | 379 B | canonical file | 560 B, canonical file | 341 B |
| `sympy__sympy-11870` | 1806 s (cap) | 1081 B | other file (`sympy/functions/elementary/exponential.py`) | 3582 B, other file (`sinc_research.md`) | 830 B |

- `audit_rollouts.py` flags 16 of the 26 sessions: history reads
  (`git log --all`, `git show <sha>`) that can only reach ancestors of the
  base commit, and network attempts that fail inside the sandbox. The
  `flask-4045` session tried eight fetches, including the PR diff and the
  fix commit from GitHub, and every one returned `HTTP:000 size:0`;
  `seaborn-2848`, `pylint-6506` and `sphinx-10325` got `URLError: nodename
  nor servname provided` on every attempt. No session saw the fix.

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `logs/<inst>.log` — opencode JSON event stream (local only)
- `launch.log`, `proxy.log`, `rollout.log` (local only)
