# flint (oQ8e), bash-only tool set, prefix cache, 64K, patches 022–023 — 1/3, zero runaway turns

Same configuration as `../flint-oQ8e-bashonly-64k-2026-10-31/` (oQ8e,
CTX=65536, a 65536-slot pool, chunk 1024, 2 GB MLX cache, model sampling,
radix cache on; opencode reduced to `bash` with the `description`
backfill, `limit.output` 4096; every request captured, 334 turns) on
SGLang v0.5.23 + patches 002–023. No crash; OOM guard: zero warnings.

## Result

| # | Instance | Wall | Diff | Steps | bash | Runaway turns | Max input | Median turn | Longest turn |
|---|----------|-----:|-----:|------:|-----:|--------------:|----------:|------------:|-------------:|
| 1 | `astropy__astropy-12907` | 493 s | 0 B | 103 | 103 | 0 | 61.1K tok | 1.7 s | 104 s (one 4,036-token bash call) |
| 2 | `astropy__astropy-14182` | 334 s | 0 B | 84 | 84 | 0 | 61.1K tok | 1.9 s | 7 s |
| 3 | `astropy__astropy-14365` | 580 s | 619 B | 141 | 144 | 0 | 61.3K tok | 1.9 s | 15 s |

- 0 of 334 responses hit the output cap; every tool call closed. The same
  configuration before patch 022 had 9 runaways in 170 turns.
- Instance 3's patch compiles `_type_re` with `re.IGNORECASE` in
  `astropy/io/ascii/qdp.py` (the earlier runs and qwen36 put `(?i)` on
  `_command_re`); the session kept exploring after the edit.
- All three sessions ended at the server's context limit (`Requested token
  count exceeds the model's maximum context length of 65536`): opencode
  sent requests past `limit.context` instead of compacting, and instances
  1–2 reached the cap after 84–103 turns of `find`/`grep`/`sed` without an
  edit.
- 1.7–1.9 s median per turn: every turn is a prefix hit; the one long turn
  is a single bash call that ran to 4,036 output tokens and closed.

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `logs/<inst>.log` — opencode JSON event stream
- `capture/` — every chat-completion request and streamed response (local only)
- `launch.log`, `proxy.log`, `capture_proxy.log`, `rollout.log`
