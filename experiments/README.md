# Experiments — M4 Pro MLX (m4)

Vetted execution queue. Each doc is a self-contained spec: a fresh agent starts
at Method step 1. The table is the execution order; statuses here are
authoritative.

| Order | ID | Spec | Status | Why now / gates |
|:-:|---|---|---|---|
| 1 | disk-triage | (no spec — ~1h task) | done | 213 GiB free of 926 GiB; the HF cache is 120 GB (hub) + 39 GB (content store). M4-E and bisect arms are no longer disk-gated. |
| 2 | re-measure | (no spec — qwen36 SWE-bench re-run) | ready | The SWE-bench cell is from the earlier stack pin (dedicated opencode-harness session). Depth/concurrency re-measure is done: eight presets characterized at genuine depth and armed in `benchmarks/baselines.json` (schema v2); remaining legacy artifacts stay flagged in `benchmarks/LEGACY-DEPTH-SUSPECT.md` (coder-next infeasible, gemma4* blocked at boot, non-DWQ dupes pending disk triage). |
| 3 | M4-E | [In-house qwen36 MLX 4-bit with exclusions](07-qwen36-inhouse-mlx4bit-exclusions.md) | ready (disk cleared) | Needs ~70+ GB (67 GB BF16 + conversion scratch). Motivation is the DWQ-quality precedent; run its probe A/B after M4-B lands. |
| 4 | extend-tax-profile | (no spec — LOW priority) | ready | The 32K append-1 turn tax measured 5.4x decode (117.6 ms; `benchmarks/session-endurance/VERDICT.md`) — marginally over the 5x threshold. Profile whether the cost sits in pool-gather or contiguous-cache rebuild before any fix; absolute latency (~120 ms/turn) is already imperceptible, hence LOW. |
| 5 | flint-agentic | (no spec — opencode-harness session) | ready | Flint is the first preset trained specifically for coding agents (Terminal-Bench 2.1 54.9% / SWE-bench Verified 74.2% at bf16 per its card). Run the SWE-bench Lite harness (`evals/swebench/`) on `flint` (oQ8e) with the model's own sampling (`--sampling-defaults model`, no thinking proxy needed — the `flint` parsers split `reasoning_content`) and compare with the qwen36 cell. Also sweep oQ8e vs oQ6e vs oQ4e on the depth bench + the SWE-bench cell to decide whether a smaller variant should be the long-context default. |
| 6 | flint-prefix-cache | (no spec yet — patch-level work) | ready | Flint runs with the radix cache off: upstream stores only full-attention KV in the MLX pool and recomputes the whole prefix on a sliding-window model, so every opencode turn re-prefills its 15–30K-token context (~20–45 s at the measured 630–930 tok/s) — the dominant cost of an agentic trajectory. A sliding layer only ever attends to its last 513 tokens, so an *exact* prefix hit needs just that window per layer: snapshot the 35 `WindowedAttentionKVCache` buffers at prefix boundaries through the auxiliary-state pool (the DeltaNet snapshot path, patch 008) and gate reuse on exact-prefix matches only (partial-band rebuilds stay inexact, as upstream notes). Gate: greedy output token-identical with and without the hit (the qwen36 standard), then the extend-tax bench. |
| — | upstream-candidates | (no spec — repo-only) | ready | Offer upstream / sister repos the generic fixes: declared-tokenizer-class guard, `wrap_as_pixtral` mistral3 + top-level `spatial_merge_size`, pixtral MULTI_IMAGES splitter, mamba-radix `no_buffer` resolution on MLX, device-aware FutureMap stash, **the MLX buffer-cache cap** (unbounded-cache prefill growth affects every MLX serving stack). |

Scheduling constraint: the box serves one model at a time — M4-E and any
probe sweep need exclusive serving windows; the repo-only items don't.

## Cross-rig edges

| From | To | What moves |
|---|---|---|
| M4-D | 3090-B | Executable Docker-scoring callout in the 3090 README (corrected score_docker.py command, runtime bound, result-return path). Shared drop-dir: `evals/swebench/m4-imports/qwen36-m4/` (3090-B's name wins as the executor). |
| 3090-B | M4-D | `scores-docker-summary.json` (first full-26 official resolve rate) committed back to this repo (`exports/qwen36-docker-summary.json` + README headline update), replacing the 5/13-local-subset cell. |
| beyond-128k | 3090 / R9700 | Buffer-cache-cap method note — delivered in the sister READMEs' cross-team notes (grep `M4→R9700 (2026-07-20)` / `M4→3090 (2026-07-20)`). |
| agentic-endurance-and-extend-tax | 3090 / R9700 | Turn-tax comparison — delivered in the same cross-team notes (85x does not reproduce on MLX: 2.6x@8K, 5.4x@32K). |
| flint-agentic | 3090 / R9700 | Flint availability note: Recursive's own SGLang kit (`flint_sglang` wheel, SGLang 0.5.16) targets one ≥141 GB GPU per replica in bf16, so the 2×24 GB rigs need a quantized variant (the Hub lists NVFP4/AWQ-class conversions) plus the kit's FlintConfig / parsers — patch 019 re-implements those as plain SGLang registrations (config registry, reasoning + tool-call parsers), which port as-is; only the MLX loader/attention pieces are Apple-specific. Neither sister repo carries Flint work yet (both are on v0.5.15). |

## Fleet sync points involving this rig

- **docker-scoring-handoff** — participants: M4-D, 3090-B; trigger: the callout
  is committed with the resolved drop-dir; action: 3090-B runs score_docker.py
  under their score lock and commits the summary back here.
- **baseline-schema-v2-to-sister-rearm** — participants: 3090-D, R9700, M4;
  trigger: 3090-D commits schema v2 and arms `benchmarks/baselines.json`;
  action: this rig re-arms its baselines on its own depth-verified instrument
  (~32K with oom_guard).
