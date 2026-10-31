# qwen36 in-house 4-bit (hazard classes BF16) 3-instance smoke — 3/3, byte-identical to the incumbent's patches

Spec 07 gate 5: the in-house build (`scripts/quantize/convert_qwen36_inhouse.py`,
`MODEL=~/AI/models/Qwen3.6-35B-A3B-4bit-inhouse`) served as `qwen36` with
the preset's flags on SGLang v0.5.23 + patches 002–023, same harness as
`../qwen36-v0523-p023-2026-10-31/` (opencode 1.15.4 through
`no_thinking_proxy.py`, `task` tool disabled, no venv, 1800 s per instance).
OOM guard: zero warnings.

## Result

| # | Instance | Wall | Diff | Steps | Tools | Max input | Target file |
|---|----------|-----:|-----:|------:|-------|----------:|-------------|
| 1 | `astropy__astropy-12907` | 50 s | 506 B | 6 | edit 1, glob 3, read 3 | 18.5K tok | `astropy/modeling/separable.py` |
| 2 | `astropy__astropy-14182` | 70 s | 624 B | 6 | bash 1, edit 1, glob 2, read 4 | 27.9K tok | `astropy/io/ascii/rst.py` |
| 3 | `astropy__astropy-14365` | 63 s | 476 B | 7 | edit 2, glob 1, grep 1, read 3 | 25.0K tok | `astropy/io/ascii/qdp.py` |

All three diffs are byte-identical to the incumbent
`mlx-community/Qwen3.6-35B-A3B-4bit` run on the same stack; 50–70 s per
instance against its 51–70 s.

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `logs/<inst>.log` — opencode JSON event stream
- `launch.log`, `proxy.log`, `rollout.log`
