# Cross-stack predictions exports

This directory holds deduplicated SWE-bench Lite prediction JSONL files
ready to ship to the 3090 stack for Docker-harness scoring.

The M4 box can't run the official SWE-bench Docker harness (no Docker
images for Apple Silicon for many of the older Python interpreters that
SWE-bench instances pin). The 3090 stack has the Docker setup and the
score_docker.py harness. The intended handoff:

```
M4 (this box)                    3090 stack
─────────────                    ──────────
opencode → SGLang+MLX rollouts
   ↓
aggregate.py --export
   ↓
predictions.jsonl    ────────►   score_docker.py --predictions <path>
                                    ↓
                                 PASS/FAIL per instance
                                    ↓
                                 leaderboard.json    ────────►   back to M4
```

## Files

- `qwen36-predictions.jsonl` — deduplicated qwen36 results across all
  archived runs in `evals/swebench/runs/`. **15 of 16 unique instances
  have non-empty patches** (the one miss: `django__django-11019`,
  topological-sort algorithmic rewrite, see SWE-bench README).

## Regenerate

```bash
python evals/swebench/aggregate.py \
    --export evals/swebench/exports/qwen36-predictions.jsonl \
    --model-filter sglang/qwen36
```

The aggregator prefers records with non-empty patches when deduplicating
across runs that hit the same instance multiple times — without that, the
alphabetic last-write-wins would prefer 0-byte diagnostic runs (e.g.
`qwen36-thinking-*`) over real post-proxy runs.

## Score on the 3090

```bash
# On the 3090 stack:
scp <m4>:~/AI/m4-sglang-inference/evals/swebench/exports/qwen36-predictions.jsonl \
    ~/AI/2x-3090-GA102-300-A1-sglang-inference/evals/swebench/m4-imports/
cd ~/AI/2x-3090-GA102-300-A1-sglang-inference
python evals/swebench/score_docker.py \
    --predictions evals/swebench/m4-imports/qwen36-predictions.jsonl \
    --output evals/swebench/leaderboards/qwen36-m4.json
```

The Docker harness will run the full SWE-bench test suite per instance
and return PASS/FAIL. Send the leaderboard JSON back to M4 to update
the headline rate from "patch-engagement" to "actual pass-rate."

## Expected rate

Patch-engagement rate on M4 is 15/16 = 93.75%. The patches target the
canonical bug locations from the issue reports and match the gold-patch
structure on manual review (see per-run READMEs under
`evals/swebench/runs/<run-name>/README.md`). The Docker pass-rate is
expected to land below patch-engagement — some patches will be on the
right file but wrong line, hallucinate type signatures, etc. A 60-80%
real pass-rate would be a strong result; anything above 50% confirms
the qwen36 recommendation.
