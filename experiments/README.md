# Experiments — M4 Pro MLX (m4)

Vetted execution queue. Each doc is a self-contained spec: a fresh agent starts
at Method step 1. The table is the execution order; statuses here are
authoritative.

| Order | ID | Spec | Status | Why now / gates |
|:-:|---|---|---|---|
| 1 | disk-triage | (no spec — ~1h task) | **ready, do first** | 12 GiB free of 926 GiB; HF cache holds 362 GB. Inventory `~/.cache/huggingface/hub` per-repo sizes + last access, propose deletions to Matt (gemma4 variants and superseded dupes are candidates). Gates M4-E and comfortable bisect arms. |
| 2 | beyond-128k | [Deep-prefill memory budget — phase 2](08-beyond-128k.md) | ready | 160K is validated (patch 008 cache cap + CHUNKED=2048 + patch 015 pre-size; `benchmarks/longctx-bisect/ATTRIBUTION.md`). 192K exhausts the budget at ~180K prefilled — the bf16 per-request attention cache is the dominant unquantized, uncapped term. Phase 2: quantize it (also attacks decode TPOT at depth, 13-19 s/token) or pool-backed prefill writes. Related envelope receipts: radix-off concurrent prefill (conc-8) and dense-devstral genuine-32K both trip the oom_guard; radix-on retains ~0.4 GB/request outside the pool (`benchmarks/radix-ab/VERDICT.md`). |
| 3 | M4-G | [check_tool_call boot gate](06-check-tool-call-gate.md) | ready | Cheap gate against the tool-call-class failure mode. |
| 4 | re-measure | (no spec — remaining legacy tables + a qwen36 SWE-bench re-run) | ready | Short-sweep and batched-decode README tables and the SWE-bench cell are from earlier stack pins; `qwen35-9b-8bit` / `qwen36-27b` presets unswept. The four tripwire presets (qwen36, coder-30b, qwen3-moe, devstral) are re-measured at genuine depth and armed in `benchmarks/baselines.json` (schema v2). |
| 5 | M4-E | [In-house qwen36 MLX 4-bit with exclusions](07-qwen36-inhouse-mlx4bit-exclusions.md) | **blocked: disk** | Needs ~70+ GB (67 GB BF16 + conversion scratch). Motivation is the DWQ-quality precedent; run its probe A/B after M4-B lands. |
| — | upstream-candidates | (no spec — repo-only) | ready | Offer upstream / sister repos the generic fixes: declared-tokenizer-class guard, `wrap_as_pixtral` mistral3 + top-level `spatial_merge_size`, pixtral MULTI_IMAGES splitter, mamba-radix `no_buffer` resolution on MLX, device-aware FutureMap stash, **the MLX buffer-cache cap** (unbounded-cache prefill growth affects every MLX serving stack). |

Scheduling constraint: the box serves one model at a time — beyond-128k, M4-G/E and any
probe sweep need exclusive serving windows; the repo-only items don't.

## Cross-rig edges

| From | To | What moves |
|---|---|---|
| M4-D | 3090-B | Executable Docker-scoring callout in the 3090 README (corrected score_docker.py command, runtime bound, result-return path). Shared drop-dir: `evals/swebench/m4-imports/qwen36-m4/` (3090-B's name wins as the executor). |
| 3090-B | M4-D | `scores-docker-summary.json` (first full-26 official resolve rate) committed back to this repo (`exports/qwen36-docker-summary.json` + README headline update), replacing the 5/13-local-subset cell. |
| beyond-128k | 3090 / R9700 | The buffer-cache-cap finding (unbounded MLX cache retained ~0.6 MB/token across chunked prefill; not a lib regression) is MLX-specific but the method — cap the allocator cache before bisecting libs — is worth a sister-README note. |

## Fleet sync points involving this rig

- **docker-scoring-handoff** — participants: M4-D, 3090-B; trigger: the callout
  is committed with the resolved drop-dir; action: 3090-B runs score_docker.py
  under their score lock and commits the summary back here.
- **baseline-schema-v2-to-sister-rearm** — participants: 3090-D, R9700, M4;
  trigger: 3090-D commits schema v2 and arms `benchmarks/baselines.json`;
  action: this rig re-arms its baselines on its own depth-verified instrument
  (~32K with oom_guard).
