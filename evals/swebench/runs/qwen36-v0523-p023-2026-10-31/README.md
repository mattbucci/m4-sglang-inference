# qwen36 3-instance smoke on patches 022–023 — 3/3, the same three fixes

Re-run of `../qwen36-v0523-2026-10-31/` on SGLang v0.5.23 + patches
002–023 (radix cache on): `CTX=131072 EXTRA_ARGS="--kv-cache-dtype
turboquant --chunked-prefill-size 2048 --mem-fraction-static 0.5"`,
opencode 1.15.4 through `no_thinking_proxy.py`, `task` tool disabled, no
venv, 1800 s per instance. OOM guard: zero warnings.

## Result

| # | Instance | Wall | Diff | Steps | Tools | Max input | Change |
|---|----------|-----:|-----:|------:|-------|----------:|--------|
| 1 | `astropy__astropy-12907` | 70 s | 506 B | 9 | bash 1, edit 1, glob 2, grep 1, read 4 | 22.4K tok | `_cstack`: `= 1` → `= right` (byte-identical to the earlier run) |
| 2 | `astropy__astropy-14182` | 69 s | 624 B | 6 | edit 1, glob 2, read 4 | 27.6K tok | `RST.__init__(header_rows=None)` defaulting to `["name"]` (the earlier run passed it through unchanged) |
| 3 | `astropy__astropy-14365` | 51 s | 476 B | 5 | edit 1, glob 2, read 2 | 22.6K tok | `(?i)` on `_command_re` (the earlier run also added a doctest line) |

3/3 patches in 51–70 s, 1–2 s median per turn with every follow-up turn a
prefix hit; the two patches that differ from the earlier run differ under
model sampling, not in the fix.

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `logs/<inst>.log` — opencode JSON event stream
- `launch.log`, `proxy.log`, `rollout.log`
