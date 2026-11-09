# flint (oQ8e), omp full tool set, 64K, sealed — 0/3, every instance at the 1200 s cap without an edit landing

Flint oQ8e served as `flint` (CTX=65536 with a 65536-slot pool, chunk 1024,
2 GB MLX cache, model sampling, radix cache on) on SGLang v0.5.23 + the 17
patches, driven by omp 18.12.2 with its default tool set minus the web tools
(`read bash edit eval glob grep wait todo write`), context window 65536,
output budget 4096, the proxy backfilling omp's required `i` argument, 1200 s
per instance, sealed (localhost-only sandbox, ancestor-only checkout). No
venv (install failed, bash prompt). OOM guard: no warnings. The configuration
probe that decides the tool shape of the 26-instance Flint cell.

## Result

| # | Instance | Wall | Diff | Turns | Tools | Max input | Ending |
|---|----------|-----:|-----:|------:|-------|----------:|--------|
| 1 | `astropy__astropy-12907` | 1206 s (cap) | 0 B | 69 | read 46, bash 16, grep 9 | 57.4K tok | still reading `modeling/core.py` in 80-line windows at the cap |
| 2 | `astropy__astropy-14182` | 1206 s (cap) | 0 B | 20 | read 39 | 59.0K tok | still reading `io/ascii/core.py` at the cap, 60 s per turn |
| 3 | `astropy__astropy-14365` | 1206 s (cap) | 0 B | 198 | edit 118, bash 74, read 5, glob 2 | 55.7K tok | 118 `edit` calls, every one rejected |

- **Flint puts omp's `edit` DSL in the wrong argument.** 115 of the 118
  `edit` calls carry only `i` (the intent label) and its value is the
  hashline edit itself (`[astropy/io/ascii/qdp.py#4B12]\nPUT 62.=62 …`),
  so omp rejects each with "input must be a string (was missing)"; the three
  calls that did fill `input` sent a placeholder anchor. The edit Flint kept
  retrying is the case-insensitive `_command_re`, the same fix it lands
  through bash in the bash-only runs. Two `bash` calls on instance 1 were
  rejected because Flint wrote an `i` longer than omp's 200-character cap.
- Every `bash` call carried `command`; when Flint omitted `i` the proxy
  supplied it (`BACKFILL_INTENT`), and omp accepted those calls.
- Reads: Flint uses omp's `path:start-end` line ranges, 40–80 lines at a
  time, and instances 1 and 2 spent the whole budget reading toward the
  context cap without a hypothesis.
- Runaway turns: 0 in 287 responses (longest 198 output tokens).
- `audit_rollouts.py`: no web, network or history hit.

The 26-instance cell therefore runs bash-only (`DISABLE_TOOLS` of everything
but `bash`, `PROMPT_STYLE=bash`), the shape Flint's own harnesses use. A
proxy remap for the `i`-only `edit` calls is the lever for a later
full-tool-set lane.

## Files

- `predictions.jsonl` (three empty patches), `predictions/<inst>.diff`
- `sessions/<inst>/*.jsonl` — omp's pi-row session store (archived)
- `logs/`, `launch.log`, `proxy.log`, `rollout.log`, `omp-agent/` (local only)
