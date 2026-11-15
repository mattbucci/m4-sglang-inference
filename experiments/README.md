# Experiments — M4 Pro MLX (m4)

Each doc is a self-contained spec: method, probes, and a disposition per experiment. Open
work is not queued here; it lives in [GitHub issues](https://github.com/mattbucci/m4-sglang-inference/issues) (`area:` and `P1`–`P3` labels).

| ID | Spec | Disposition |
|---|---|---|
| M4-E | [In-house qwen36 MLX 4-bit with exclusions](07-qwen36-inhouse-mlx4bit-exclusions.md) | Ties the community 4-bit build on teacher-forced NLL and KL to the 8-bit reference, so it is opt-in via `MODEL=` and the community build is the default (`docs/quality.md`). |
| M4-P | [MLX decode micro-experiments](11-mlx-decode-micro-experiments.md) | Patches 029 to 035 and 040 and `MLX_MAX_MB_PER_BUFFER=200`; fused decode launches are on in `qwen36` (`patches/README.md`). Open levers: issues #25, #27, #28, #29. |
| M4-F | [Flint against the published M4 Pro engines](12-flint-published-parity.md) | Patches 036 to 039; decode is 78 to 84 % of Kapitsa through 128K and ahead of rec at 128K (`docs/flint.md`, `docs/performance-ceiling.md`). Open lever: issue #26. |

The box serves one model at a time, so any spec that needs a server needs an exclusive
serving window.

## Cross-rig contracts

- **Docker scoring** — this rig commits predictions to `evals/swebench/exports/` and to the
  3090's drop-dir `evals/swebench/m4-imports/`; the 3090 runs `score_docker.py` under its
  score lock and commits the summary back (`evals/swebench/exports/qwen36-docker-summary.json`).
  Unscored exports: issue #21.
- **Baselines** — `benchmarks/baselines.json` is schema 2 on the depth-verified instrument;
  re-arm with `BASELINE=save scripts/bench/bench_regression.sh <preset>` after any preset
  default change.
- **Cross-team notes** — generic fixes go to the sister repos as notes, never as upstream
  pull requests (issue #35).
