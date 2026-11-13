# flint (int4), omp full tool set with `i` moved last, 64K, sealed — 0/3 at the 1200 s cap, no edit attempted

The same three instances, server settings and cap as `../flint-oQ8e-omp-ilast-64k-sealed-2026-11-09/`
(CTX=65536, chunk 1024, 2 GB MLX cache, model sampling, radix cache on, omp 18.12.2, output budget
4096, 1200 s per instance, `INTENT_LAST=1`, no response-side rewrites), with
`MODEL=RecursiveSI/flint-int4` and the preset's fused Flint MoE and five-part submission.

| # | Instance | Wall | Diff | Turns | Tools | Tool errors | Max input | Ending |
|---|----------|-----:|-----:|------:|-------|------------:|----------:|--------|
| 1 | `astropy__astropy-12907` | 1206 s (cap) | 0 B | 106 | bash 106 | 3 | 52.1K tok | reading `modeling/rotations.py` in `sed -n` windows |
| 2 | `astropy__astropy-14182` | 1206 s (cap) | 0 B | 125 | bash 102, read 24 | 5 | 54.3K tok | reading `io/ascii/core.py` |
| 3 | `astropy__astropy-14365` | 1230 s (cap) | 0 B | 112 | bash 101, read 8, grep 3, glob 1 | 5 | 56.4K tok | reading `io/ascii/ui.py` in `sed -n` windows |

0/3 non-empty against 1/3 for oQ8e. int4 never calls `edit` (oQ8e: 22 calls on instance 3, 10
applied) and shows the same read-to-the-cap pattern at a similar turn count, so the faster decode
does not buy engagement. Three instances cannot rank the checkpoints; nothing here argues for
moving the default off oQ8e.

Files: `predictions.jsonl`, `sessions/`; `logs/`, `launch.log`, `proxy.log`, `rollout.log` (local only).
