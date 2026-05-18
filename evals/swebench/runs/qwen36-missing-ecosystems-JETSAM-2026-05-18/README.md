# qwen36 missing-ecosystems sweep — JETSAM CONTAMINATED, do not score

## What this was supposed to be

Close the recommendation gap: qwen36 had been validated on 7 SWE-bench
Lite ecosystems (astropy, django, matplotlib, sympy, pylint, sphinx,
scikit-learn). This sweep picked 1 instance from each of the 5 missing
ecosystems (flask, pytest, requests, seaborn, xarray) to test the
"qwen36 generalizes" claim.

## What actually happened

The SGLang server **jetsam'd at 12:51:51**, ~10 minutes into the
seaborn-2848 rollout. The smoke.sh + run_rollouts.py harness had no
per-instance health check, so opencode kept POST-ing chat completions
to a dead upstream for the remaining 4 instances. Every one of them
hit the 900s wall, captured 0 bytes, and was recorded as a "model
failure."

The seaborn result is the **only real data point** — the model
genuinely made 15 tool calls (1 bash, 1 edit, 1 glob, 4 grep, 8 read)
and produced a 710-byte patch before the server died mid-rollout.

| Instance | Wall | Patch | Tool calls | Real? |
|----------|:---:|:----:|:----------:|:-----:|
| `mwaskom__seaborn-2848` | 908 s | 710 B | 15 | ✓ (worktree captured pre-death) |
| `pallets__flask-4045` | 904 s | 0 B | 0 | ✗ (jetsam — server already dead) |
| `psf__requests-1963` | 903 s | 0 B | 0 | ✗ (jetsam) |
| `pydata__xarray-3364` | 908 s | 0 B | 0 | ✗ (jetsam) |
| `pytest-dev__pytest-11143` | 905 s | 0 B | 0 | ✗ (jetsam) |

## Root cause + fix

The `run_rollouts.py` harness ran one preflight at startup and then
trusted the server to stay alive across all subsequent instances. macOS
jetsam silently reaped the SGLang scheduler mid-sweep — no traceback,
no exit code, just a server that stopped responding to chat completions.
This is the same pattern documented in `project_eval_jetsam_artifact.md`
for the LAB-Bench evals; the hardening in `eval_and_chart.py` (#45)
never propagated to `run_rollouts.py`.

**Fix** (commit landing alongside this archive): per-instance health
check before opencode is launched. If `preflight_canary` fails, the
harness writes a record tagged `server_dead=True`, prints an ABORT
message, and `sys.exit(4)`. No more 900-second timeout burns on a
dead upstream.

## Real recommendation status (unchanged by this contaminated sweep)

- qwen36 on 7 ecosystems: 16/17 = 94.1% — still the figure of record.
- qwen36 on missing ecosystems: **only seaborn is verified (1/1)**; the
  remaining 4 (flask/pytest/requests/xarray) must be re-run on a fresh
  server. Cleanup retry is queued.

## Files

Each instance's record + diff is preserved here for forensics, but
**do NOT include this directory's predictions.jsonl in cross-stack
exports**. The contaminated rows are deceptively identical in shape
to real results.
