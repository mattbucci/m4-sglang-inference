# qwen36 on the omp scaffold — 3-instance sealed smoke, 3/3 canonical files

First cell on the omp (oh-my-pi 18.12.2) scaffold, the harness default:
`PRESET=qwen36 INSTANCES=3 TIMEOUT=900 bash evals/swebench/smoke.sh` as
shipped (`CTX=131072`, turboquant, chunk 2048, `MEM_FRAC=0.5`, radix cache
on, `no_thinking_proxy` on, omp `--thinking off`, tools
`read bash edit eval glob grep wait todo write`, `task` and `web_search`
off, localhost-only seatbelt, ancestor-only checkout). One server served
all three; OOM guard zero warnings; 10.8 minutes wall including boot.

| Instance | Wall | Patch | File | Note |
|---|---:|---:|---|---|
| `astropy__astropy-12907` | 418 s | 1479 B | `astropy/modeling/separable.py` | canonical file; `_cstack` recurses into `CompoundModel` operands, but the `_cdot` hunk also deletes the two `cleft`/`cright` assignments, so this patch would not pass |
| `astropy__astropy-14182` | 156 s | 874 B | `astropy/io/ascii/rst.py` | `header_rows` threaded through `RST.__init__` and the separator line indexed by its length — the upstream shape |
| `astropy__astropy-14365` | 71 s | 617 B | `astropy/io/ascii/qdp.py` | `re.IGNORECASE` on the line-type regex — the upstream fix |

Sessions: 51 / 39 / 17 assistant turns, 52 / 38 / 17 tool calls (12907: 23
bash, 17 read, 7 edit; 14182: 16 read, 8 edit, 2 write; 14365: 6 read,
5 edit). The opencode cell on the same stack patched 12907 in 55 s with
the 506 B upstream change; omp's session spent its first minutes trying to
build the package (`pip install meson-python …`), which the seal refused
(`Failed to resolve 'pypi.org'`), before reading the source.

Seal: `audit_rollouts.py` flags the one network attempt (pip, DNS failure
inside the sandbox); no web tool, no history read past the base commit.

## Files

- `predictions.jsonl` (rows carry `scaffold: omp/18.12.2`), `predictions/<inst>.diff`
- `sessions/<inst>/*.jsonl` — omp session store (pi row format), the audit's input
- `logs/<inst>.log`, `launch.log`, `proxy.log`, `rollout.log`, `smoke.run.log` (gitignored)
