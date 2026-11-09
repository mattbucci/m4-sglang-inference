# flint (oQ8e), omp bash-only, 64K, sealed — 1/3, all three at the 1200 s cap, a third of one session's calls rejected for an over-long intent

Same server configuration as `../flint-oQ8e-omp-64k-sealed-2026-11-08/` (oQ8e
as `flint`, CTX=65536 with a 65536-slot pool, chunk 1024, 2 GB MLX cache,
model sampling, radix cache on, SGLang v0.5.23 + the 17 patches), omp 18.12.2
with every tool but `bash` disabled and the bash prompt style, the proxy
backfilling omp's `i` argument, output budget 4096, 1200 s per instance,
sealed. No venv (install failed, bash prompt). OOM guard: no warnings.

## Result

| # | Instance | Wall | Diff | Turns | bash calls | Rejected by omp | Max input | Ending |
|---|----------|-----:|-----:|------:|-----------:|----------------:|----------:|--------|
| 1 | `astropy__astropy-12907` | 1206 s (cap) | 0 B | 108 | 108 | 0 | 54.5K tok | exploring `modeling/separable.py` and `core.py` at the cap |
| 2 | `astropy__astropy-14182` | 1206 s (cap) | 0 B | 101 | 103 | 6 | 52.2K tok | exploring the ascii writers at the cap |
| 3 | `astropy__astropy-14365` | 1206 s (cap) | 474 B | 203 | 203 | 67 | 55.6K tok | patch landed through `sed`, then kept verifying until the cap |

- Instance 3's patch is the `(?i)` form of the case-insensitive `_command_re`
  in `astropy/io/ascii/qdp.py`, the same fix as the opencode bash-only runs.
- **Flint never ends its turn under omp.** Every one of the 412 responses is a
  tool call; the opencode bash-only runs ended two of three instances on
  their own in 125–930 s. With `bash` the only tool and no idle stop, each
  instance runs to the cap, so the per-instance cap bounds the cell's wall.
- **Flint learns the backfilled intent and overruns it.** In instance 1 it
  wrote `i` itself on 107 of 108 calls; in instances 2 and 3 it copied the
  proxy's `bash: <command>` form into every call, pasting the whole command,
  and omp rejected 6 and 67 calls for exceeding its 200-character intent
  limit ("the tool was not run"). A third of instance 3's budget went to
  rejected calls. The proxy's `REMAP_INTENT=1` clamps `i` to 200 characters;
  the 26-instance cell runs with it.
- Runaway turns: 0 in 412 responses (longest 638 output tokens against the
  4096 cap).
- `audit_rollouts.py`: one history read in instance 1, `git show` of a commit
  in the checkout's own ancestry (the ancestor-only checkout holds nothing
  past the base commit); no web or network hit.

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `sessions/<inst>/*.jsonl` — omp's pi-row session store (archived)
- `logs/`, `launch.log`, `proxy.log`, `rollout.log`, `omp-agent/` (local only)
