# flint (oQ8e) with the prefix cache and the bash backfill — 0/1, two runaway turns, then Metal OOM

First SWE-bench rollout of the `flint` preset with patch 021 (radix cache
on) and the proxy backfilling opencode's required bash `description`.
Same harness as `../flint-oQ8e-2026-10-31/` otherwise: oQ8e at
CTX=131072 with `--max-total-tokens 131072` (a 3.7 GB pool), chunk 1024,
2 GB MLX cache, model sampling, 1800 s per instance, no venv. The server
died before the second instance (`rollout_returncode -2`, `server_dead`).

## Result

| # | Instance | Wall | Diff | Steps | bash | read | invalid | Max input | Median turn |
|---|----------|-----:|-----:|------:|-----:|-----:|--------:|----------:|------------:|
| 1 | `astropy__astropy-12907` | 1805 s | 0 B | 47 | 44 (0 rejected) | 3 | 2 | 61.3K tok | 2 s |
| 2 | `astropy__astropy-14182` | — | — | — | — | — | — | — | server dead |

## What worked

- **Prefix cache:** 45 of the 47 turns took 5.7 s or less end to end; the
  server log shows turns such as `#new-token: 783, #cached-token: 23833`.
  The previous run spent 24–52 s per turn re-prefilling the same context.
- **bash backfill:** all 44 bash calls were accepted and executed
  (`find`/`grep`/`sed`/`awk` over the repo); the earlier runs had every
  bash call rejected.

## What failed

- **Two runaway turns.** At 39.3K and 61.3K tokens of context the model
  opened `<tool_call>bash<arg_key>command</arg_key><arg_value>sed -n` and
  then wrote prose ("Let me run more targeted greps…") instead of the
  command and the closing tags, until the 16,384-token output limit
  (`finish=length`): 436 s and 487 s, 923 s of the 1805 s budget. opencode
  received the unterminated arguments as an invalid tool input
  (`JSON parsing failed`). Whether this is model behaviour at long context
  under sampling or a cache-path artefact is not established by this run;
  the next run records every request (capture proxy) so a runaway turn can
  be replayed with the cache on and off.
- **Metal out-of-memory before instance 2.** The first instance ended as a
  77K-token request (61K context + the 16K runaway output): its
  per-request KV (28 KB/token on the 14 full layers, 2.2 GB) plus the
  3.7 GB pool sit at the 51.8 GB Metal working set next to the 42.3 GB of
  weights, and the next instance's cold prefill
  (`kIOGPUCommandBufferCallbackErrorOutOfMemory`, kernel-available 9.0 GB)
  tipped it over. With the radix cache on, a 131K-token request does not
  fit this checkpoint on 64 GB; the harness now bounds Flint at 64K
  (CTX=65536, a 65536-slot pool, opencode `limit.context` 65536 so the
  agent compacts instead of growing).

## Files

- `predictions.jsonl`, `predictions/astropy__astropy-12907.diff` (0 B)
- `logs/astropy__astropy-12907.log` — opencode JSON event stream
- `launch.log` (the Metal OOM traceback), `proxy.log`, `rollout.log`
