# flint (oQ8e), bash-only tool set, prefix cache, 64K — 1/3, nine runaway turns (stack before patch 022)

Same server as `../flint-oQ8e-prefix-cache-64k-2026-10-31/` (oQ8e, CTX=65536,
a 65536-slot pool, chunk 1024, 2 GB MLX cache, model sampling, radix cache
on; SGLang v0.5.23 + patches 002–021, i.e. before patch 022) with the
opencode tool set reduced to `bash` (`DISABLE_TOOLS` for the other nine,
`PROMPT_STYLE=bash`: "your only tool is bash"), the bash `description`
backfill, `limit.output` 4096, and every request captured
(`capture/reqNNN.*`, 181 turns). No crash; the OOM guard warned at 10.9 GB
free and never fired.

## Result

| # | Instance | Wall | Diff | Steps | bash | Runaway turns (context) | Max input | Median turn |
|---|----------|-----:|-----:|------:|-----:|-------------------------|----------:|------------:|
| 1 | `astropy__astropy-12907` | 858 s | 0 B | 52 | 59 | 4 (16.6K, 35.3K, 50.1K, 61.4K) | 61.4K tok | 2 s |
| 2 | `astropy__astropy-14182` | 362 s | 0 B | 53 | 66 | 1 (9.9K) | 61.0K tok | 2 s |
| 3 | `astropy__astropy-14365` | 693 s | 476 B | 71 | 77 | 4 (25.0K, 35.5K, 46.1K, 55.7K) | 55.7K tok | 1 s |

- Instance 3's patch is the `(?i)` fix to `_command_re` in
  `astropy/io/ascii/qdp.py` — the same change qwen36 makes
  (`../qwen36-v0523-2026-10-31/`).
- Instances 1 and 2 ended at the context cap (`limit.context` 65536 minus
  the 4,096-token output reserve) without an edit; instance 2 spent its
  53 turns on `find`/`grep`/`sed` over the repo.
- Every bash call was accepted (the backfill supplied `description`);
  turns were 1–2 s median because each one was a prefix hit.

## Runaway turns: 9 of 170 responses

Nine responses hit the 4,096-token output cap with an unterminated
`<tool_call>` (opencode: invalid tool input). Across the radix-on harness
runs that is 13 runaways in ~254 turns; across the radix-off runs
(`../flint-oQ8e-2026-10-31/`, `../flint-oQ8e-bash-description-rule-2026-10-31/`,
`../flint-oQ8e-bashonly-64k-radixoff-2026-10-31/`) it is 0 in ~395. That
split is what pointed at the hit path rather than the model's sampling: a
prefix hit's final chunk is computed as two forwards on the pool-backed
attention cache (patch 020's trunk-then-last-token split), and the upstream
cache re-gathered only the pool prefix on the second call, so the first
sampled token of such a turn saw the prefix plus itself and the sync that
followed wrote stale buffer rows into the pool for the new tokens, which
every later turn then gathered. Patch 022 fixes the cache; the earlier
replay table (`../flint-oQ8e-prefix-cache-64k-2026-10-31/`) missed it
because a replay of a captured turn is a full-prefix hit with no new
tokens.

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `logs/<inst>.log` — opencode JSON event stream
- `capture/` — every chat-completion request and streamed response (local only)
- `launch.log`, `proxy.log`, `capture_proxy.log`, `rollout.log`
