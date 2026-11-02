# Qwen3.8-27B community 4-bit, sealed harness, opencode limit 32K — 2/3 non-empty patches

`mlx-community/Qwen3.8-27B-4bit` served by the `qwen36-27b` preset
(`MODEL=` override, CTX=131072, `CHUNKED=2048`, mem-fraction 0.5, radix on,
thinking always on) on SGLang v0.5.23 + the 16 patches, through the sealed
harness: `webfetch`/`websearch`/`task` off, opencode and its bash children
in the localhost-only sandbox, ancestor-only checkout. No venv (install
failed, read-edit prompt), 1800 s per instance. OOM guard: no warnings.

This run went out with opencode's `limit.context` for the `qwen36-27b` key
at 32768 against the 131K server, so opencode compacted the sessions as
their input neared ~29K (a 1-2K-token summary turn at ~13 tok/s, then
re-exploration): 0, 4 and 2 compactions on the three instances. The
sibling run `../qwen38-27b-community-sealed-131k-2026-11-01/` repeats the
smoke with the limit matching the server.

## Result

| # | Instance | Wall | Diff | Steps | Tools | Compactions | Max input | Target file |
|---|----------|-----:|-----:|------:|-------|------------:|----------:|-------------|
| 1 | `astropy__astropy-12907` | 448 s | 504 B | 18 | bash 8, read 6, edit 2, grep 1 | 0 | 25.8K tok | `astropy/modeling/separable.py` |
| 2 | `astropy__astropy-14182` | 1811 s (timeout) | empty | 36 | read 18, grep 12, bash 4 | 4 | 33.6K tok | — (still reading `rst.py`/`fixedwidth.py` at the kill) |
| 3 | `astropy__astropy-14365` | 1309 s | 477 B | 26 | bash 13, grep 6, edit 5, read 3 | 2 | 30.5K tok | `astropy/io/ascii/qdp.py` |

- Instance 1's patch is the upstream fix itself: `_cstack` copies the
  right operand's matrix (`= right`) instead of filling ones. The
  unsealed session on this instance had fetched the post-fix file through
  `webfetch` and produced nothing; this one wrote it from source.
- Instance 3's patch makes `_command_re` case-insensitive with an inline
  `(?i:...)` group, the same change qwen36 makes.
- Instance 3 ran separately after the sweep: the sweep's health probe had
  read the server as dead while it was still prefilling the turn of the
  just-killed instance 2 (`run_rollouts.py` now aborts that turn and
  retries a busy probe). Its own `launch.log`/`rollout.log`/guard log are
  under `rerun-14365/`; the same server configuration.
- `audit_rollouts.py`: no web, network or history hit in the three sessions.
- Server gates after the sweep on the same launch: `validate_capabilities.py`
  3/3 (thinking terminated at 245 tokens), `probe_codegen.py` 8/8 STRONG.

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `logs/<inst>.log` — opencode JSON event stream (local only)
- `launch.log`, `proxy.log`, `rollout.log`, `rerun-14365/` (local only)
