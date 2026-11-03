# Qwopus3.8-27B-Coder-Flash (in-house MLX 4-bit), 26-instance sealed SWE-bench Lite cell — 23/26 non-empty, 22/26 on the canonical file

The qwen36 scorecard's 26 instances (`../qwen36-26inst-p023-2026-10-31/`) on
`~/AI/models/Qwopus3.8-27B-Coder-Flash-4bit-inhouse` (published as
`mattbucci/Qwopus3.8-27B-Coder-Flash-MLX-4bit`) served by the `qwen36-27b`
preset (`MODEL=` override, CTX=131072, `CHUNKED=2048`, mem-fraction 0.5,
radix on, thinking always on) on SGLang v0.5.23 + the 17 patches, through the
sealed harness (`webfetch`/`websearch`/`task` off, opencode and its bash
children in the localhost-only sandbox, ancestor-only checkout), opencode
`limit.context` 131072, 1800 s per instance (the qwen36 cell ran at 900 s;
the dense 27B decodes at 13 tok/s with thinking on). One server served all
26 in sequence: no jetsam, OOM guard zero warnings, 10.1 h wall.

## Result

23/26 non-empty patches, 22/26 touching the file the qwen36 cell's
patch touches (its patches target the canonical issue file on every
instance). 13/26 hit the 1800 s cap; 10 of those still left a diff. The
three empties (`django-10924`, `seaborn-2848`, `pytest-11148`) are timeouts
after 77–123 steps of exploration; `sympy-11870` wrote a research note
instead of code. Resolution needs the Docker harness on the 3090:
`../../exports/qwopus38-coder-flash-26inst-predictions.jsonl` is the file to
score (model name `sglang/qwopus38-27b-coder-flash`).

| Instance | Wall | Patch | Target | qwen36 cell |
|---|---:|---:|---|---:|
| `astropy__astropy-12907` | 1557 s | 504 B | canonical file | 506 B |
| `astropy__astropy-14182` | 1361 s | 2352 B | canonical file | 579 B |
| `astropy__astropy-14365` | 540 s | 1641 B | canonical file | 476 B |
| `astropy__astropy-14995` | 381 s | 626 B | canonical file | 1187 B |
| `astropy__astropy-6938` | 308 s | 558 B | canonical file | 516 B |
| `astropy__astropy-7746` | 1240 s | 1135 B | canonical file | 1042 B |
| `django__django-10914` | 1657 s | 625 B | canonical file | 3521 B |
| `django__django-10924` | 1808 s (cap) | 0 B | empty | 622 B |
| `django__django-11001` | 1807 s (cap) | 4825 B | canonical file | 754 B |
| `django__django-11019` | 1807 s (cap) | 3474 B | canonical file | 726 B |
| `django__django-11039` | 647 s | 658 B | canonical file | 716 B |
| `matplotlib__matplotlib-18869` | 1842 s (cap) | 2600 B | canonical file | 1956 B |
| `mwaskom__seaborn-2848` | 1806 s (cap) | 0 B | empty | 2842 B |
| `pallets__flask-4045` | 736 s | 2133 B | canonical file | 488 B |
| `psf__requests-1963` | 1804 s (cap) | 517 B | canonical file | 412 B |
| `psf__requests-2148` | 781 s | 1228 B | canonical file | 1045 B |
| `pydata__xarray-3364` | 1807 s (cap) | 1885 B | canonical file | 3291 B |
| `pylint-dev__pylint-5859` | 1574 s | 1112 B | canonical file | 672 B |
| `pylint-dev__pylint-6506` | 1804 s (cap) | 796 B | canonical file | 676 B |
| `pytest-dev__pytest-11143` | 1351 s | 713 B | canonical file | 485 B |
| `pytest-dev__pytest-11148` | 1804 s (cap) | 0 B | empty | 501 B |
| `scikit-learn__scikit-learn-10297` | 522 s | 1564 B | canonical file | 932 B |
| `sphinx-doc__sphinx-10325` | 1814 s (cap) | 9398 B | canonical file | 1515 B |
| `sphinx-doc__sphinx-10451` | 1808 s (cap) | 3853 B | canonical file | 2721 B |
| `sympy__sympy-11400` | 1818 s (cap) | 560 B | canonical file | 341 B |
| `sympy__sympy-11870` | 1807 s (cap) | 3582 B | other file (`sinc_research.md`) | 830 B |

- Per-instance stats (steps, tool mix, max input) are in the opencode event
  streams under `logs/` (local only); the median session is 70 steps with
  bash dominant, 50–80K max input.
- `audit_rollouts.py` flags 8 sessions: history reads (`git log --all`,
  `git show <sha>`) that can only reach ancestors of the base commit, because
  the checkout fetches exactly that commit; and network attempts
  (`urllib.request.urlopen` to raw.githubusercontent, a `curl` to the GitHub
  API) that fail inside the sandbox (`URLError: nodename nor servname
  provided`). No session saw the fix.

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `logs/<inst>.log` — opencode JSON event stream (local only)
- `launch.log`, `proxy.log`, `rollout.log` (local only)
