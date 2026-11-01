# Qwen3.8-27B community 4-bit 3-instance smoke — 2/3 non-empty patches

`mlx-community/Qwen3.8-27B-4bit` served by the `qwen36-27b` preset
(`MODEL=` override, `CHUNKED=2048`, radix on, thinking always on — the 3.8
chat template ignores `enable_thinking`) on SGLang v0.5.23 + the patch set,
same harness as `../qwen36-v0523-p023-2026-10-31/` (opencode through
`no_thinking_proxy.py`, `task` tool disabled, no venv, 1800 s per instance).
OOM guard: no memory warnings (one startup notice).

## Result

| # | Instance | Wall | Diff | Steps | Tools | Max input | Target file |
|---|----------|-----:|-----:|------:|-------|----------:|-------------|
| 1 | `astropy__astropy-12907` | 1800 s (timeout) | empty | 26 | bash 9, read 9, webfetch 6, grep 2 | 37.2K tok | — (explored, no edit) |
| 2 | `astropy__astropy-14182` | 828 s | 873 B | 17 | read 8, bash 5, grep 2, edit 1 | 29.0K tok | `astropy/io/ascii/rst.py` |
| 3 | `astropy__astropy-14365` | 394 s | 499 B | 8 | edit 4, read 3 | 24.6K tok | `astropy/io/ascii/qdp.py` |

Both patches land on the same target files as the qwen36 runs. The dense
27B decodes at ~14 tok/s, so a thinking-on rollout costs 7–14 min per
instance against qwen36's ~1 min, and the 12907 rollout runs out its 30 min
budget while still reading; the in-house 4-bit build (`../qwen38-27b-inhouse-2026-10-31/`)
behaves the same way.

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `logs/<inst>.log` — opencode JSON event stream
- `launch.log`, `proxy.log`, `rollout.log`
