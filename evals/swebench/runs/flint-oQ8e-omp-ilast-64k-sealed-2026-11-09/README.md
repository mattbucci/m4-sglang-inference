# flint (oQ8e), omp full tool set with `i` moved last in every schema, 64K, sealed — 1/3 at the 1200 s cap; no malformed call in 366, no response-side rewrite needed

Same server as `../flint-oQ8e-omp-64k-sealed-2026-11-08/` (oQ8e as `flint`,
CTX=65536 with a 65536-slot pool, chunk 1024, 2 GB MLX cache, model sampling,
radix cache on, SGLang v0.5.23 + the 17 patches), omp 18.12.2 with its default
tool set minus the web tools, output budget 4096, 1200 s per instance, sealed,
and the proxy running `INTENT_LAST=1` with `BACKFILL_INTENT=0` and
`REMAP_INTENT=0`: the proxy moves omp's `i` (intent) key and the other
metadata keys to the end of every tool schema's `properties` and `required`
on the request side, and rewrites nothing in the responses. The request-side
counterpart of `../flint-oQ8e-omp-remap-64k-sealed-2026-11-08/` (same three
instances, same server and cap, response-side backfill + remap + clamp),
after the 3090 found Flint fills `i` with its main argument because omp
declares `i` first.

## Result

| # | Instance | Wall | Diff | Turns | Tools | Calls without `i` | Rejected by omp | Max input | Ending |
|---|----------|-----:|-----:|------:|-------|------------------:|----------------:|----------:|--------|
| 1 | `astropy__astropy-12907` | 1206 s (cap) | 0 B | 124 | bash 107, read 17 | 124 of 124 | 0 | 55.9K tok | reading `modeling/core.py` in `sed -n` windows at the cap |
| 2 | `astropy__astropy-14182` | 1206 s (cap) | 0 B | 114 | bash 106, read 9 | 115 of 115 | 0 | 53.8K tok | reading `io/ascii/core.py` in `sed -n` windows at the cap |
| 3 | `astropy__astropy-14365` | 1206 s (cap) | 617 B | 126 | bash 90, edit 22, read 11, grep 3, glob 1 | 104 of 127 | 0 | 56.2K tok | patch landed through `edit`; `git diff --check` + `py_compile` loop at the cap |

Cell wall 61 min including the 30 s boot.

- **Every call is well-formed.** With `i` last, Flint puts the payload in the
  tool's own key on all 366 calls: `bash` is always `{command}`, `read` is
  `{path}`, and all 22 `edit` calls are `{i, input}` with the hashline DSL in
  `input` (the un-reordered smoke had it in `i` on 115 of 118). No `i`
  exceeds 200 characters (longest 92), omp rejected nothing, and the proxy
  rewrote nothing, so nothing was copied into later calls.
- **omp runs a call without `i`.** Flint drops `i` on 343 of 366 calls
  (every `bash`, `read` and `grep`; it keeps it on `edit` and `glob`) and omp
  executes them without complaint: its only `i` check is the 200-character
  length cap. `BACKFILL_INTENT` is therefore not needed under omp.
- **Edits:** 22 calls on instance 3, 10 applied, 5 parsed-but-no-op (the
  body restated the file), 2 rejected for a stale hash after an intervening
  change, 4 DSL slips (`input: "True"`, a `read` in `input`, a payload with no
  hunk header, a `PUT` body too short), 1 stopped by omp's three-no-ops rule.
  The remap smoke landed 8 of 13. The surviving patch is the
  `re.compile(_type_re, re.IGNORECASE)` form in `astropy/io/ascii/qdp.py`
  (the reference fix); the remap smoke's was the `(?i)` prefix.
- **Flint still works through bash.** 303 of 366 calls are `bash`; instances
  1 and 2 never tried `edit`, read files through `sed -n` and `grep` and ran
  to the cap without an edit, as in every full-tool smoke. The tool-schema
  order changes the shape of the calls, not Flint's strategy.
- Flint never ends its turn: all 364 responses are tool calls (stop reason
  `toolUse`), so the per-instance cap bounds the cell's wall.
- Runaway turns: 0 (longest response 422 output tokens).
- The 7 `Cannot write to closing transport` tracebacks in `proxy.log` are the
  harness killing omp at the cap mid-stream (6 and 8 in the sister smokes).
- `audit_rollouts.py`: no web, network or history hit.

Verdict: the request-side reorder alone removes every malformed call the
response-side repairs were written for, with nothing rewritten; it is the
Flint lane configuration (`smoke.sh` sets `INTENT_LAST=1` for `PRESET=flint`),
with the full tool set offered, and `BACKFILL_INTENT`/`REMAP_INTENT` stay off.
Engagement (1/3, same instance, both unpatched instances at the cap) is
unchanged, so the cell-level number it is expected to move is the
rejected-call and edit-applied rate, not resolve. The 3090's re-roll of its
`flint:omp` cell under its own key-order fix is still queued there; this is
the configuration that will compare with it.

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `sessions/<inst>/*.jsonl` — omp's pi-row session store (archived)
- `logs/`, `launch.log`, `proxy.log`, `rollout.log`, `omp-agent/` (local only)
