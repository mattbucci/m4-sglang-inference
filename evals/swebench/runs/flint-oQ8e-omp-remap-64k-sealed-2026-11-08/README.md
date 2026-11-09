# flint (oQ8e), omp full tool set with the intent remap, 64K, sealed — 1/3 at the 1200 s cap; the remap makes omp's `edit` usable, Flint still prefers bash

Same server as `../flint-oQ8e-omp-64k-sealed-2026-11-08/` (oQ8e as `flint`,
CTX=65536 with a 65536-slot pool, chunk 1024, 2 GB MLX cache, model sampling,
radix cache on, SGLang v0.5.23 + the 17 patches), omp 18.12.2 with its default
tool set minus the web tools, output budget 4096, 1200 s per instance, sealed,
and the proxy running `BACKFILL_INTENT=1 REMAP_INTENT=1`: a payload Flint
sends as `i` moves to the tool's own argument and `i` is clamped to omp's
200-character limit. The configuration probe for the 26-instance cell after
the un-remapped full-tool smoke (0/3, every edit rejected) and bash-only
smoke (1/3, a third of one session rejected for over-long intents).

## Result

| # | Instance | Wall | Diff | Turns | Tools | Rejected by omp | Max input | Ending |
|---|----------|-----:|-----:|------:|-------|----------------:|----------:|--------|
| 1 | `astropy__astropy-12907` | 1206 s (cap) | 0 B | 159 | bash 151, read 8 | 0 | 55.7K tok | exploring `modeling/separable.py` with bash at the cap |
| 2 | `astropy__astropy-14182` | 1206 s (cap) | 0 B | 137 | bash 139, read 4 | 0 | 56.0K tok | exploring the ascii writers with bash at the cap |
| 3 | `astropy__astropy-14365` | 1206 s (cap) | 474 B | 106 | bash 74, read 18, edit 13, grep 2, glob 1 | 0 | 55.7K tok | patch landed through `edit`, then 60 failed regex checks until the cap |

- **The remap works.** Instance 3 made 13 `edit` calls and 8 applied; the
  five failures are Flint's own DSL slips (an angle-bracket anchor, a
  payload line without a hunk header, two hunks on one range). No call in
  the three sessions was rejected for a missing `input` or an over-long `i`
  (one `i` was clamped). The patch is the `(?i)` form of the case-insensitive
  `_command_re` in `astropy/io/ascii/qdp.py`, as in every Flint run.
- **Flint still works through bash.** With the full tool set offered, 364 of
  the 412 calls are `bash`; instances 1 and 2 never tried `edit`, read
  files through `sed -n` and `grep`, and ran to the cap without an edit, as
  in the bash-only smoke. The 64 failed bash calls on instance 3 are a loop
  of `python -c "import re; …"` regex checks that each died on an
  unterminated subpattern.
- Flint never ends its turn: all 402 responses are tool calls, so the
  per-instance cap bounds the cell's wall whatever the tool set.
- Runaway turns: 0 (longest response 292 output tokens).
- `audit_rollouts.py`: no web, network or history hit.

The decision rule (two of three patched) picks bash-only with the clamp for
the 26-instance cell: the tool set changes nothing on the two unpatched
instances, and the bash shape matches Flint's own harnesses.

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `sessions/<inst>/*.jsonl` — omp's pi-row session store (archived)
- `logs/`, `launch.log`, `proxy.log`, `rollout.log`, `omp-agent/` (local only)
