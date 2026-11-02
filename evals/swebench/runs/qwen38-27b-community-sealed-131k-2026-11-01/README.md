# Qwen3.8-27B community 4-bit, sealed harness, opencode limit 131K — 3/3, each the upstream change

`mlx-community/Qwen3.8-27B-4bit` served by the `qwen36-27b` preset
(`MODEL=` override, CTX=131072, `CHUNKED=2048`, mem-fraction 0.5, radix on,
thinking always on) on SGLang v0.5.23 + the 16 patches, through the sealed
harness: `webfetch`/`websearch`/`task` off, opencode and its bash children
in the localhost-only sandbox, ancestor-only checkout. opencode's
`limit.context` for the `qwen36-27b` key is 131072 (matching the server;
`../qwen38-27b-community-sealed-2026-11-01/` is the same smoke under the
earlier 32K limit). No venv (install failed, read-edit prompt), 1800 s per
instance. OOM guard: no warnings. No compaction in any session.

## Result

| # | Instance | Wall | Diff | Steps | Tools | Max input | Target file |
|---|----------|-----:|-----:|------:|-------|----------:|-------------|
| 1 | `astropy__astropy-12907` | 495 s | 504 B | 18 | read 11, grep 6, bash 5, glob 1, edit 1 | 32.0K tok | `astropy/modeling/separable.py` |
| 2 | `astropy__astropy-14182` | 397 s | 552 B | 24 | grep 8, read 7, bash 6, edit 2 | 31.9K tok | `astropy/io/ascii/rst.py` |
| 3 | `astropy__astropy-14365` | 224 s | 989 B | 11 | edit 4, bash 3, read 2, grep 1 | 24.0K tok | `astropy/io/ascii/qdp.py` |

- 12907: `_cstack` copies the right operand's matrix (`= right`) instead of
  filling ones — the upstream fix.
- 14182: `RST.__init__` takes `header_rows` and passes it to `FixedWidth`
  — the upstream fix.
- 14365: `_line_type_re` compiled with `re.IGNORECASE` and the `NO` mask
  compared case-insensitively — the two lines of the upstream fix (the
  32K-limit run and qwen36 patch the command regex instead).
- 7–8 min per instance at ~13 tok/s decode with thinking on; the 32K-limit
  run had spent 22–30 min on the two instances that compacted.
- `audit_rollouts.py`: no web, network or history hit in the three sessions.

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `logs/<inst>.log` — opencode JSON event stream (local only)
- `launch.log`, `proxy.log`, `rollout.log` (local only)
