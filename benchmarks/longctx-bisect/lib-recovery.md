# Old MLX lib versions — recovery evidence

No pip freeze of the pre-rebuild venvs exists and the pip wheel cache is
empty; versions below are reconstructed from the repo's recorded version
table (git history) + PyPI upload timestamps. All are installable from PyPI
(arm64/sdist present).

## Timeline

| Stack point | sglang | mlx | mlx-lm | mlx-vlm | 32K prefill growth |
|---|---|---|---|---|---|
| 128K validation (commit `8957971` era) | v0.5.11 + patches 002-008 | **0.31.1** (README version table of that era) | 0.31.2 | 0.4.4 (0.5.0 hand-bumped later, pre-rebuild) | ~0 (128K fit, guard never warned below 10 GB) |
| v0.5.12 era (regression receipts) | v0.5.12 + 14 patches | **0.31.2** (newest on PyPI at the rebuild; 0.32.0 did not exist yet) | 0.31.3 | 0.5.0 | 0.19-0.22 MB/token (32K ceiling) |
| current | v0.5.15.post1 + 6 patches | **0.32.0** | 0.31.3 | 0.6.5 | ~0.6-0.7 MB/token (phase0: 32K killed at mem-frac 0.5/CTX 140K) |

## PyPI upload dates (fetched via /pypi/<pkg>/json)

| Package | Version | Uploaded |
|---|---|---|
| mlx | 0.31.1 | 2026-03-12 |
| mlx | 0.31.2 | 2026-04-22 |
| mlx | 0.32.0 | 2026-07-07 |
| mlx-lm | 0.31.2 | 2026-04-07 |
| mlx-lm | 0.31.3 | 2026-04-22 |
| mlx-vlm | 0.4.4 | 2026-04-04 |
| mlx-vlm | 0.5.0 | 2026-05-06 |
| mlx-vlm | 0.6.5 | 2026-07-16 |

## Implications for the arms

- Every historical stack step changed BOTH the sglang tree and mlx-core, but
  the growth rate worsened monotonically alongside mlx 0.31.1 → 0.31.2 →
  0.32.0.
- mlx-lm has been 0.31.3 across the v0.5.12 era and today — effectively
  eliminated as the recent-worsening axis (only 0.31.2→0.31.3 spans the
  original boundary).
- Cheapest single-axis probe (no arm builds, no disk): downgrade mlx alone in
  the CURRENT venv (0.32.0 → 0.31.2, then → 0.31.1 if compatible) and repeat
  the 32K growth measurement on unchanged v0.5.15.post1 code.
