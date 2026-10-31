# flint (oQ8e), default tool set, prefix cache, 64K, patches 022–023 — 1/3, zero runaway turns

Same configuration as `../flint-oQ8e-prefix-cache-64k-2026-10-31/` (all
opencode tools, bash `description` backfill, `limit.output` 4096; oQ8e at
CTX=65536 with a 65536-slot pool, chunk 1024, 2 GB MLX cache, model
sampling, radix cache on; every request captured, 187 turns) on SGLang
v0.5.23 + patches 002–023. No crash; OOM guard: zero warnings.

## Result

| # | Instance | Wall | Diff | Steps | Tools | Runaway turns | Max input | Median turn | Longest turn |
|---|----------|-----:|-----:|------:|-------|--------------:|----------:|------------:|-------------:|
| 1 | `astropy__astropy-12907` | 285 s | 0 B | 63 | bash 62, read 1 | 0 | 61.3K tok | 1.7 s | 3 s |
| 2 | `astropy__astropy-14182` | 264 s | 0 B | 53 | bash 54 | 0 | 60.3K tok | 1.9 s | 4 s |
| 3 | `astropy__astropy-14365` | 379 s | 477 B | 66 | bash 38, read 23, edit 2, grep 2, glob 1 | 0 | 52.1K tok | 2.7 s | 12 s |

- 0 of 187 responses hit the output cap. This configuration before patch
  022 had two runaways in 37 turns and no patch.
- Instance 3's patch wraps `_command_re` in an inline `(?i:…)` group in
  `astropy/io/ascii/qdp.py`; the session ended with a final text turn, the
  only instance in the Flint runs to stop on its own.
- Instances 1–2 used bash almost exclusively (one `read` in 116 turns) and
  ended at the server's context limit without an edit, as in the bash-only
  run (`../flint-oQ8e-bashonly-64k-p023-2026-10-31/`).

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `logs/<inst>.log` — opencode JSON event stream
- `capture/` — every chat-completion request and streamed response (local only)
- `launch.log`, `proxy.log`, `capture_proxy.log`, `rollout.log`
