# flint (oQ8e), bash-only tool set, radix cache off, 64K — 1/3, every instance at the 1800 s timeout

The radix-off twin of `../flint-oQ8e-bashonly-64k-2026-10-31/`: same
server settings with `EXTRA_ARGS="--disable-radix-cache"` (the recompute
path: every turn re-prefills its whole context), same bash-only opencode
configuration, bash backfill, `limit.output` 4096, every request captured
(135 turns). OOM guard: zero warnings.

## Result

| # | Instance | Wall | Diff | Steps | bash | Runaway turns | Max input | Median turn |
|---|----------|-----:|-----:|------:|-----:|--------------:|----------:|------------:|
| 1 | `astropy__astropy-12907` | 1805 s | 0 B | 38 | 38 | 0 | 44.5K tok | 48 s |
| 2 | `astropy__astropy-14182` | 1808 s | 0 B | 38 | 47 | 0 | 43.4K tok | 44 s |
| 3 | `astropy__astropy-14365` | 1808 s | 754 B | 54 | 55 | 0 | 29.8K tok | 30 s |

- 0 of 133 responses hit the output cap; every tool call closed.
- Instance 3's patch makes `_command_re` case-insensitive with character
  classes (`[Rr][Ee][Aa][Dd] [TtSs][Ee][Rr][Rr]`) and adds a doctest line;
  the instance was still running when the 1800 s budget ended.
- 30–48 s per turn is the re-prefill of 30–45K tokens at ~800 tok/s; the
  radix-on twin spent 1–2 s per turn on the same instances.

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `logs/<inst>.log` — opencode JSON event stream
- `capture/` — every chat-completion request and streamed response (local only)
- `launch.log`, `proxy.log`, `capture_proxy.log`, `rollout.log`
