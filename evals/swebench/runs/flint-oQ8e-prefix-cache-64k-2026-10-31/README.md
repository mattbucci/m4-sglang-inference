# flint (oQ8e), prefix cache, bash backfill, 64K context bound — 0/3, two runaway turns (stack before patch 022)

Same configuration as `../flint-oQ8e-prefix-cache-2026-10-31/` with the
context bounded at 64K (`CTX=65536`, a 65536-slot pool, opencode
`limit.context` 65536; output limit still 16384) and every request
recorded by a capture proxy between the harness proxy and the server
(`capture/reqNNN.request.json` + `.response.sse`, 37 turns). No crash, no
OOM-guard warning.

## Result

| # | Instance | Wall | Diff | Steps | Tools | Max input | Median turn | Longest turn |
|---|----------|-----:|-----:|------:|-------|----------:|------------:|-------------:|
| 1 | `astropy__astropy-12907` | 497 s | 0 B | 6 | bash 4, read 1, invalid 1 | 17.7K tok | 1 s | 451 s (16,384 tokens, `length`) |
| 2 | `astropy__astropy-14182` | 121 s | 0 B | 8 | read 9, grep 1 | 47.4K tok | 8 s | 3 s |
| 3 | `astropy__astropy-14365` | 508 s | 0 B | 17 | read 11, grep 3, glob 1, edit 2, invalid 1 | 30.1K tok | 2 s | 415 s (16,384 tokens, `length`) |

- Instances 1 and 3 each ended in a **runaway turn**: the model opened a
  tool call (`bash` at 17.7K tokens of context, `edit` at 30.1K) and then
  wrote prose inside the argument value until the 16,384-token output
  limit — in instance 3 a JSON-style tool call pasted into the XML
  argument, then "Let me just Do that edit now:>" repeated to the limit.
  opencode received it as an invalid tool input and the session ended.
- Instance 3's two real `edit` calls were rejected by opencode before that
  (one without `filePath`, one without `oldString`), so the diff is empty.
- Instance 2 stopped after eight read/grep turns at 47.4K tokens: with
  `limit.context` 65536 and `limit.output` 16384, opencode has ~49K of
  input room, and the session ended there without an edit.

## Replaying the runaway turns (`capture/req007`, `capture/req036`)

The two requests that produced the runaways, replayed verbatim against the
server (model sampling unless noted, `max_tokens` 2048, classified as a
closed tool call / final text / `length`):

| Arm | req036 (30.1K tokens) | req007 (17.7K tokens) |
|-----|----------------------:|----------------------:|
| radix on, prefix hits | 6/6 tool call | 6/6 tool call |
| radix on, cold (flush before each) | 6/6 tool call | — |
| radix off server | 6/6 tool call | 4/4 tool call |
| greedy, hit / cold | 1/1 / 1/1 tool call | — |
| temperature 0.6, hits | 6/6 tool call | — |

0 runaways in 36 replays; every replay of req036 is a well-formed `edit`
call with `filePath`. The runaways are therefore per-turn stochastic
events under the model's sampling (two in ~70 radix-on turns across the
two prefix-cache runs), not a property of the cached-prefix path. The
harness now caps Flint's per-turn output at 4,096 tokens (`limit.output`
in the opencode config) so a runaway costs ~90 s instead of ~450 s.

The replays never exercised the defect: a replay of a captured turn is a
full-prefix hit with no new tokens, while a live turn's new tokens run
through the split final chunk of a prefix hit, where the upstream
pool-backed cache dropped the first forward's tokens (patch 022). The
bash-only pair exposed it (`../flint-oQ8e-bashonly-64k-2026-10-31/`: 9
runaways in 170 radix-on turns against 0 in 133 with the cache off), and
the rerun of this configuration on patches 022–023
(`../flint-oQ8e-prefix-cache-64k-p023-2026-10-31/`) has 0 runaways in 187
turns.

## Files

- `predictions.jsonl`, `predictions/<inst>.diff` (all 0 B)
- `logs/<inst>.log` — opencode JSON event stream
- `capture/` — every chat-completion request and streamed response (local only)
- `launch.log`, `proxy.log`, `capture_proxy.log`, `rollout.log`
