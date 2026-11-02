# flint (oQ8e), bash-only tool set, prefix cache, 64K, sealed harness — 1/3, no runaway turns

Same configuration as `../flint-oQ8e-bashonly-64k-2026-10-31/` (oQ8e served as
`flint`, CTX=65536 with a 65536-slot pool, chunk 1024, 2 GB MLX cache, model
sampling, radix cache on, opencode tool set reduced to `bash` with the
`description` backfill, `limit.context` 65536 / output 4096) on the current
stack (SGLang v0.5.23 + the 16 patches, 022 and 026 included), run under the
sealed harness: `webfetch`/`websearch`/`task` off, opencode and its bash
children in the localhost-only sandbox, ancestor-only checkout. No venv
(install failed, bash prompt), 1800 s per instance. OOM guard: no warnings.

## Result

| # | Instance | Wall | Diff | Steps | bash | Runaway turns | Max input | Ending |
|---|----------|-----:|-----:|------:|-----:|--------------:|----------:|--------|
| 1 | `astropy__astropy-12907` | 187 s | 0 B | 34 | 33 | 0 | 38.4K tok | model ended its turn with an analysis, no edit |
| 2 | `astropy__astropy-14182` | 929 s | 0 B | 212 | 212 | 0 | 61.4K tok | context cap (62,259 + 4,096 > 65,536, server 400) |
| 3 | `astropy__astropy-14365` | 126 s | 493 B | 20 | 19 | 0 | 20.4K tok | patch, model declared the fix verified |

- Instance 3's patch makes `_command_re` in `astropy/io/ascii/qdp.py`
  case-insensitive (character classes; the earlier Flint run used `(?i)`),
  the same change qwen36 makes (`../qwen36-v0523-2026-10-31/`).
- Instance 2 spent its 212 turns on `find`/`grep`/`sed` over the ascii
  writers and hit the context cap without an edit, as in the earlier run.
- Instance 1 stopped on its own after 34 turns: a long written analysis of
  `_cstack` ending with the intent to look up the upstream fix, and no
  edit. The seal leaves no channel for that lookup.
- Every bash call was accepted (the backfill supplied `description`).
- **Runaway turns: 0 in 265 responses** (longest response 2,175 tokens
  against the 4,096 cap). The earlier run on the stack before patch 022 had
  9 in 170; the pool-backed cache fix holds in a real rollout.
- `audit_rollouts.py`: no web, network or history hit in the three sessions.

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `logs/<inst>.log` — opencode JSON event stream (local only)
- `launch.log`, `proxy.log`, `rollout.log` (local only)
