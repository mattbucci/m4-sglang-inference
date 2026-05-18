# qwen36 scale test — 6/7 on SWE-bench Lite instances 4-10

## Result

| Instance | Wall | Diff | Notes |
|----------|:----:|:----:|-------|
| `astropy__astropy-14995` | 157 s | 628 B | astropy edit |
| `astropy__astropy-6938` | 73 s | 516 B | fastest run so far |
| `astropy__astropy-7746` | 235 s | 567 B | astropy edit |
| `django__django-10924` | 859 s | 1714 B | Django multi-file |
| `django__django-11001` | 772 s | 1368 B | Django substantial |
| `django__django-11019` | 45 s | **0** | **convergence miss** |
| `django__django-11039` | 349 s | 826 B | Django edit |

6/7 produce patches. The miss (`django-11019`) is a convergence
failure — 2 globs + 1 read + 1 text turn, then the model exited after
45 seconds. Not a structural break (other Django instances in the
same batch produced 800-1700 byte patches).

## Cumulative scorecard (2026-05-18, with proxy)

| Run | Instances | Patches | Cumulative |
|-----|:---------:|:-------:|:----------:|
| Initial smoke | 1 | 1/1 | 1/1 |
| 3-instance reproducibility | 3 (overlap) | 3/3 | 3/3 unique |
| Cross-repo (django/matplotlib/sympy) | 3 | 3/3 | 6/6 |
| Harder ecosystems (pylint/sklearn/sphinx) | 3 | 3/3 | 9/9 |
| **Scale test (4 astropy/django + harder hold)** | **7** | **6/7** | **15/16** |

**15/16 = 93.75% patch-engagement rate across 16 unique SWE-bench Lite
instances spanning 7 ecosystems** (astropy, Django, matplotlib, sympy,
pylint, sphinx, scikit-learn).

Average wall time across all runs: ~280 s/instance. The fastest was
astropy__astropy-6938 at 73 s (model knew the answer immediately,
1 read + 1 edit). The slowest was django__django-10924 at 859 s
(close to the 900 s TIMEOUT — multi-file Django changes are the
upper bound).

## What this DOES validate

- qwen36 + `no_thinking_proxy` is a robust agentic-coding stack on M4
- The 100% earlier rate (9/9) was not selection bias — it holds in
  the mid-90s at N=16
- Failure mode when it happens is "model converged early", not
  infrastructure crash

## What this still doesn't prove

- Docker pass-rate (cross-stack scoring needed; use
  `aggregate.py --export ... --model-filter sglang/qwen36` to ship
  predictions to the 3090)
- N=100 or N=300 win rate (would take 10-25 hours of M4 wall time)
- Performance on instances requiring 5+ file changes
