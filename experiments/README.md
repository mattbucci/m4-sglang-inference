# Experiments — M4 Pro MLX (m4)

Vetted execution queue. Each doc is a self-contained spec: a fresh agent starts
at Method step 1. The table is the execution order; statuses here are
authoritative.

| Order | ID | Spec | Status | Why now / gates |
|:-:|---|---|---|---|
| 1 | disk-triage | (no spec — ~1h task) | done | 213 GiB free of 926 GiB; the HF cache is 120 GB (hub) + 39 GB (content store). M4-E and bisect arms are no longer disk-gated. |
| 2 | re-measure | (no spec — qwen36 SWE-bench re-run) | done (smoke) | The 3-instance smoke on the current stack reproduces the earlier cell: 3/3 patches on the same target files in 56–78 s per instance (`evals/swebench/runs/qwen36-v0523-2026-10-31/`); the 26-instance scorecard itself was not re-run. Depth/concurrency re-measure is done: eight presets characterized at genuine depth and armed in `benchmarks/baselines.json` (schema v2); remaining legacy artifacts stay flagged in `benchmarks/LEGACY-DEPTH-SUSPECT.md` (coder-next infeasible, gemma4* blocked at boot, non-DWQ dupes pending disk triage). |
| 3 | M4-E | [In-house qwen36 MLX 4-bit with exclusions](07-qwen36-inhouse-mlx4bit-exclusions.md) | ready (disk cleared) | Needs ~70+ GB (67 GB BF16 + conversion scratch). Motivation is the DWQ-quality precedent; run its probe A/B after M4-B lands. |
| 4 | extend-tax-profile | (no spec — LOW priority) | ready | The 32K append-1 turn tax measured 5.4x decode (117.6 ms; `benchmarks/session-endurance/VERDICT.md`) — marginally over the 5x threshold. Profile whether the cost sits in pool-gather or contiguous-cache rebuild before any fix; absolute latency (~120 ms/turn) is already imperceptible, hence LOW. |
| 5 | flint-agentic | (no spec — opencode-harness session) | ready | Flint is the first preset trained specifically for coding agents (Terminal-Bench 2.1 54.9% / SWE-bench Verified 74.2% at bf16 per its card). Run the SWE-bench Lite harness (`evals/swebench/`) on `flint` (oQ8e) with the model's own sampling (`--sampling-defaults model`, no thinking proxy needed — the `flint` parsers split `reasoning_content`) and compare with the qwen36 cell. Also sweep oQ8e vs oQ6e vs oQ4e on the depth bench + the SWE-bench cell to decide whether a smaller variant should be the long-context default. |
| 6 | flint-prefix-cache | (no spec — patch 021) | done | Patch 021 snapshots the 35 `WindowedAttentionKVCache`s through the auxiliary-state pool at the boundaries the radix tree names and restores them on an exact hit; the 14 full layers gather from the shared pool, which the runner now allocates for sliding-window models. Measured on oQ8e (`benchmarks/flint/2026-10-31/prefix-cache/`): turn 2 of a 13.4K-token conversation starts in 0.6 s instead of 19 s; a full-prefix repeat in 0.2 s; validator 3/3, codegen STRONG, repeat test 4/4 with the cache on. Memory: the pool costs ~2 GB of headroom at 65,536 slots (cold 13K prefill 16.0 vs 18.6 GB kernel-available); the first build leaked ~6 GB through never-evaluated pool scatters, fixed by evaluating them with each step. Determinism: hits repeat exactly and cold runs repeat exactly, but a hit is not bit-identical to a cold run (the first new chunk splits at the 64-token checkpoint grid and starts at the cached length, moving bf16 logits ≤ 1 nat on tail tokens, top token unchanged) — the same holds on qwen36's pre-existing hybrid path, so the earlier “token-identical on hits” wording was corrected rather than gated on. Follow-up: for models whose snapshot state is attention-only, the tracked checkpoint can be sliced from the post-chunk window instead of splitting the forward, which would remove one of the two boundary differences and the split's extra launch. |
| — | upstream-candidates | (no spec — repo-only) | ready | Offer upstream / sister repos the generic fixes: declared-tokenizer-class guard, `wrap_as_pixtral` mistral3 + top-level `spatial_merge_size`, pixtral MULTI_IMAGES splitter, mamba-radix `no_buffer` resolution on MLX, device-aware FutureMap stash, **the MLX buffer-cache cap** (unbounded-cache prefill growth affects every MLX serving stack), and from patch 021 the **pool-scatter evaluation** (`pending_pool_arrays`: upstream never evaluates the radix pool's scatters until a hit gathers, so every chunk's scatter pins the request KV it reads and the pool version it replaces — ~6 GB retained on a 13K-token cold prefill, surviving a cache flush — on any radix-on MLX model) plus the windowed-KV snapshot path itself for sliding-window models. |

Scheduling constraint: the box serves one model at a time — M4-E and any
probe sweep need exclusive serving windows; the repo-only items don't.

## Cross-rig edges

| From | To | What moves |
|---|---|---|
| M4-D | 3090-B | Executable Docker-scoring callout in the 3090 README (corrected score_docker.py command, runtime bound, result-return path). Shared drop-dir: `evals/swebench/m4-imports/qwen36-m4/` (3090-B's name wins as the executor). |
| 3090-B | M4-D | `scores-docker-summary.json` (first full-26 official resolve rate) committed back to this repo (`exports/qwen36-docker-summary.json` + README headline update), replacing the 5/13-local-subset cell. |
| beyond-128k | 3090 / R9700 | Buffer-cache-cap method note — delivered in the sister READMEs' cross-team notes (grep `M4→R9700 (2026-07-20)` / `M4→3090 (2026-07-20)`). |
| agentic-endurance-and-extend-tax | 3090 / R9700 | Turn-tax comparison — delivered in the same cross-team notes (85x does not reproduce on MLX: 2.6x@8K, 5.4x@32K). |
| flint-agentic | 3090 / R9700 | Flint availability note: Recursive's own SGLang kit (`flint_sglang` wheel, SGLang 0.5.16) targets one ≥141 GB GPU per replica in bf16, so the 2×24 GB rigs need a quantized variant (the Hub lists NVFP4/AWQ-class conversions) plus the kit's FlintConfig / parsers — patch 019 re-implements those as plain SGLang registrations (config registry, reasoning + tool-call parsers), which port as-is; only the MLX loader/attention pieces are Apple-specific. Two harness findings port with the model, not the hardware: under opencode Flint omits the `bash` tool's required `description` argument on nearly every call whatever the prompt says (the proxy backfill in `evals/swebench/no_thinking_proxy.py` fixes it), and under opencode's full tool set it mangles multi-argument `edit` calls and occasionally runs away inside a tool call under its own sampling — the bash-only harness shape (`DISABLE_TOOLS` + `PROMPT_STYLE=bash`) matches its training harnesses. Neither sister repo carries Flint work yet (both are on v0.5.15). |

## Fleet sync points involving this rig

- **docker-scoring-handoff** — participants: M4-D, 3090-B; trigger: the callout
  is committed with the resolved drop-dir; action: 3090-B runs score_docker.py
  under their score lock and commits the summary back here.
- **baseline-schema-v2-to-sister-rearm** — participants: 3090-D, R9700, M4;
  trigger: 3090-D commits schema v2 and arms `benchmarks/baselines.json`;
  action: this rig re-arms its baselines on its own depth-verified instrument
  (~32K with oom_guard).
