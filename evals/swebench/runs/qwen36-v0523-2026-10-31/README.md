# qwen36 3-instance smoke on the v0.5.23 stack — 3/3, same patches as the earlier pin

Re-run of `../qwen36-3instance-2026-05-18/` (SWE-bench Lite instances 1–3)
on the current stack: SGLang v0.5.23 + 11 patches, radix cache on.

## Config

- **Server:** `mlx-community/Qwen3.6-35B-A3B-4bit` served as `qwen36`,
  `CTX=131072 EXTRA_ARGS="--kv-cache-dtype turboquant --chunked-prefill-size 2048 --mem-fraction-static 0.5" bash scripts/launch.sh qwen36`
  (the `--kv-cache-dtype` flag is inert on the MLX backend; KV stays bf16)
- **Agent:** opencode 1.15.4 via `evals/swebench/run_rollouts.py` through
  `no_thinking_proxy.py` (`enable_thinking=false`), `task` tool disabled
- **Instances:** `astropy__astropy-12907`, `-14182`, `-14365`; no venv
  (install failed, read-edit prompt); 1800 s per instance
- **OOM guard:** running, zero warnings

## Result

| # | Instance | Wall | Diff | Steps | Tools | Max input | Target file | Change |
|---|----------|-----:|-----:|------:|-------|----------:|-------------|--------|
| 1 | `astropy__astropy-12907` | 58 s | 506 B | 7 | glob 2, grep 1, read 3, bash 1, edit 1 | 22.8K tok | `astropy/modeling/separable.py` | `_cstack`: `= 1` → `= right` (byte-identical to the earlier run's patch) |
| 2 | `astropy__astropy-14182` | 78 s | 579 B | 9 | glob 2, grep 1, read 4, edit 4 | 28.7K tok | `astropy/io/ascii/rst.py` | `RST.__init__`: add `header_rows` (same change as the earlier run) |
| 3 | `astropy__astropy-14365` | 56 s | 707 B | 6 | glob 2, read 2, edit 2 | 23.2K tok | `astropy/io/ascii/qdp.py` | `_command_re`: add `(?i)` (same fix as the earlier run, plus a doctest line) |

3/3 patches, 6–9 steps each, median inter-step gap 1 s: every follow-up turn
is a prefix hit, so the 23–29K-token contexts never re-prefill. The earlier
run on the previous pin took 116–132 s per instance for the same three
fixes.

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `logs/<inst>.log` — opencode JSON event stream
- `launch.log`, `proxy.log`, `rollout.log`
