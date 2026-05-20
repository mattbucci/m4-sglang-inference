# qwen36 v0.5.12 rebase regression — PASS

## Context

User direction: "A new version of sglang is out let's rebuild a new venv
prune any upstreamed patches, reapply our fixes and regression test
using our evals."

SGLang `v0.5.12` (commit `127b9e328`) was tagged on 2026-05-20.
Diff vs the prior `v0.5.11` pin in MLX paths:

| File | Lines changed | What |
|---|---|---|
| `mlx/model_runner.py` | +69 / -2 | Add on-the-fly `mlx_q4` / `mlx_q8` quantization (PR #24907) |
| `mlx/tp_worker.py` | +1 | Pass `quantization=` through to `MlxModelRunner` |
| `utils/hf_transformers/processor.py` | +3 | Add `resolve_runai_obj_uri` indirection |
| (new) `layers/quantization/mlx.py` | +74 | Quantization preset config |
| (new) `tests/.../test_quantization.py` | +190 | Unit test for the new feature |
| `docs/.../apple_metal.mdx` | +24 | Doc the new feature |
| `server_args.py` | +4 | CLI plumbing for `--quantization` |

**No upstreamed patches** — none of our 19 patches (002-020) were merged
into v0.5.12. The new v0.5.12 code is orthogonal to what we patch (it
adds quantization features around the same `__init__` / `_load_model`
that our patches modify, but the changes don't overlap).

## Rebase work

The line-number shift from v0.5.12's +69 in model_runner.py broke all
19 individual patch files' hunk contexts. Rebase steps:

1. Apply patches 002, 003 cleanly (no overlap with v0.5.12 changes)
2. Patch 004 has 2 conflicts in `model_runner.py` and `tp_worker.py` —
   both at locations where v0.5.12 added new constructor params. Both
   resolved by combining changes (additive, not contradictory).
3. Patches 005-012 apply cleanly.
4. Patch 013 (`mlx-vlm-pixel-values.patch`) has a malformed hunk header
   (count mismatch); manually copy from stash + re-merge v0.5.12 deltas.
5. Patches 014, 015, 016, 017, 018 in nano_nemotron_vl.py and
   hf_transformers/processor.py — copied from stash (no v0.5.12 changes
   in nano_nemotron_vl.py; the 3-line v0.5.12 change to processor.py
   re-applied on top).
6. Patches 019, 020 in model_runner.py — copied from stash + re-applied
   v0.5.12's quantization-preset additions on top.

Final state: 19 patches' worth of changes against v0.5.12, captured as
a single cumulative patch `021-v0512-rebase-cumulative.patch` (117 KB,
touches 19 files).

## Regression result

```
PRESET=qwen36 INSTANCE_IDS=astropy__astropy-12907 TIMEOUT=600
```

| Metric | Value |
|---|---|
| Patch bytes | **506 B** (bit-identical to baseline) |
| Wall | 185 s (within noise of 125 s baseline) |
| Return code | 0 |
| Verdict | **PASS** |

The model produces the canonical 506-byte fix to
`astropy/modeling/separable.py` — same patch as on every prior v0.5.11
regression check. Greedy MLX decode + same input → bit-identical output.

## Setup changes

- `scripts/setup.sh`: `SGLANG_COMMIT="v0.5.12"` (was v0.5.11)
- `scripts/setup.sh`: apply only `021-v0512-rebase-cumulative.patch`
  instead of looping over 002-020 (the individual patches no longer
  apply cleanly to v0.5.12)
- `patches/README.md`: updated banner + per-patch table now notes
  historical-documentation status

## Files

- `run.log` — smoke.sh stdout (Smoke done rc=0)
- `predictions.jsonl` — the 506-byte prediction
- `astropy__astropy-12907.diff` — the canonical fix
