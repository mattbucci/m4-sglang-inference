# flint (oQ8e) with a bash-description rule in the task prompt — 0/1, rule ignored

Re-run of `../flint-oQ8e-2026-10-31/` with one line appended to the task
prompt (`PROMPT_EXTRA`), telling the model that opencode's `bash` tool
requires both `command` and `description`. Same server configuration
(oQ8e, CTX=131072, chunk 1024, 2 GB MLX cache, radix cache off, model
sampling), opencode 1.15.4, 1800 s per instance. Stopped after the first
instance; the second was aborted at its first turns.

## Result

| # | Instance | Wall | Diff | Steps | bash | read | Max input | Median turn |
|---|----------|-----:|-----:|------:|-----:|-----:|----------:|------------:|
| 1 | `astropy__astropy-12907` | 1805 s | 0 B | 72 | 71 (all rejected) | 2 | 19.6K tok | 24 s |

Every bash call again omitted `description` (`SchemaError(Missing key at
["description"])`, 71 identical retries). opencode's session database
confirms the rule was in the user message. The placement sweep that
followed (`../flint-oQ8e-2026-10-31/README.md`) found no prompt placement
that works under the real task prompt; the harness now backfills the
argument in the proxy instead.

## Files

- `predictions.jsonl`, `predictions/astropy__astropy-12907.diff` (0 B)
- `logs/astropy__astropy-12907.log` — opencode JSON event stream
- `launch.log`, `proxy.log`, `rollout.log`
