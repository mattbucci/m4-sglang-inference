# Qwen3.8-27B in-house 4-bit (hazard classes BF16) 3-instance smoke — 2/3 non-empty patches

`~/AI/models/Qwen3.8-27B-4bit-inhouse` (the spec-07 converter on
`Qwen/Qwen3.8-27B`: DeltaNet `in_proj_a/b` and the vision tower BF16) served
by the `qwen36-27b` preset with `MODEL=`, same flags and harness as
`../qwen38-27b-community-2026-10-31/`. OOM guard: no memory warnings.

## Result

| # | Instance | Wall | Diff | Steps | Tools | Max input | Target file |
|---|----------|-----:|-----:|------:|-------|----------:|-------------|
| 1 | `astropy__astropy-12907` | 1800 s (timeout) | empty | 34 | read 19, grep 12, bash 6 | 29.3K tok | — (explored, no edit) |
| 2 | `astropy__astropy-14182` | 1800 s (timeout) | 531 B | 54 | bash 25, grep 18, read 10, write 3, edit 2 | 30.2K tok | `astropy/io/ascii/rst.py` (patch captured at the timeout) |
| 3 | `astropy__astropy-14365` | 415 s | 619 B | 13 | read 5, grep 3, edit 2, bash 2 | 28.1K tok | `astropy/io/ascii/qdp.py` |

Same engagement as the community build (2/3, same target files); the
longer 14182 rollout spent its budget writing stubs and running tests after
the edit. Fidelity and reference-distance receipts for the A/B are in
`benchmarks/quality/fidelity/`.

## Files

- `predictions.jsonl`, `predictions/<inst>.diff`
- `logs/<inst>.log` — opencode JSON event stream
- `launch.log`, `proxy.log`, `rollout.log`
