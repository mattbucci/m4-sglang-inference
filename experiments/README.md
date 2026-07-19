# Experiments — M4 Pro MLX (m4)

Vetted execution queue. Each doc is a self-contained spec: a fresh agent starts
at Method step 1. The table is the execution order; statuses here are
authoritative.

| Order | ID | Spec | Status | Why now / gates |
|:-:|---|---|---|---|
| 1 | disk-triage | (no spec — ~1h task) | **ready, do first** | 12 GiB free of 926 GiB; HF cache holds 362 GB. Inventory `~/.cache/huggingface/hub` per-repo sizes + last access, propose deletions to Matt (gemma4 variants and superseded dupes are candidates). Gates M4-E and comfortable bisect arms. |
| 2 | M4-C | [Pin `--random-range-ratio 1` + flag legacy depth rows](01-pin-random-range-ratio-flag-legacy-rows.md) | ready | Repo edits + on-box smoke; prerequisite for honest bench rows everywhere. |
| 2 (parallel) | M4-D | [Docker-scoring ask as 3090-README callout](02-docker-scoring-ask-callout.md) | ready | Repo-only; unblocks the official full-26 SWE-bench cell on the 3090 side. |
| 3 | M4-F | [Patch replay gates + probe gate in setup.sh](03-patch-replay-gates-setup.md) | ready | Scripts the pristine-replay/byte-identity/double-apply gates; provides the per-arm gate tooling M4-A reuses. |
| 4 | radix-ab | (no spec — ~2h task) | ready | Decode-TPOT A/B on qwen36 at MR=1: radix-on (normal event loop) vs `--disable-radix-cache` (overlap loop). Keeps hybrid preset defaults data-driven. |
| 5 | M4-A | [Long-context: measure, then bisect if the regression persists](04-longctx-bisect-pre-rebase.md) | ready | Phase 0 measures the current stack's 32K growth rate — if clean, the item closes without arms. |
| 6 | M4-B | [Real sampling via mlx-lm make_sampler](05-mlx-sampling.md) | ready | Removes the greedy-only limitation; unblocks thinking-mode evals and fleet parity. |
| 7 | M4-G | [check_tool_call boot gate](06-check-tool-call-gate.md) | ready | Cheap gate against the tool-call-class failure mode. |
| 8 | re-measure | (no spec — fold into M4-C step 8-9 + a qwen36 SWE-bench re-run) | ready | README throughput/long-context tables and the SWE-bench cell are from earlier stack pins; `qwen35-9b-8bit` / `qwen36-27b` presets unswept. |
| 9 | M4-E | [In-house qwen36 MLX 4-bit with exclusions](07-qwen36-inhouse-mlx4bit-exclusions.md) | **blocked: disk** | Needs ~70+ GB (67 GB BF16 + conversion scratch). Motivation is the DWQ-quality precedent; run its probe A/B after M4-B lands. |
| — | upstream-candidates | (no spec — repo-only) | ready | Offer upstream / sister repos the generic fixes: declared-tokenizer-class guard, `wrap_as_pixtral` mistral3 + top-level `spatial_merge_size`, pixtral MULTI_IMAGES splitter, mamba-radix `no_buffer` resolution on MLX, device-aware FutureMap stash. |

Scheduling constraint: the box serves one model at a time — M4-A/B/G/E and any
probe sweep need exclusive serving windows; M4-C/D/F and the repo-only items
don't.

## Cross-rig edges

| From | To | What moves |
|---|---|---|
| M4-D | 3090-B | Executable Docker-scoring callout in the 3090 README (corrected score_docker.py command, runtime bound, result-return path). Shared drop-dir: `evals/swebench/m4-imports/qwen36-m4/` (3090-B's name wins as the executor). |
| 3090-B | M4-D | `scores-docker-summary.json` (first full-26 official resolve rate) committed back to this repo (`exports/qwen36-docker-summary.json` + README headline update), replacing the 5/13-local-subset cell. |
| M4-A | 3090 / R9700 | If the long-context regression is lib-attributed: the finding that an unpinned transitive MLX lib silently regressed a mission metric across a venv rebuild generalizes to both sister setup scripts (their extras are unpinned too); share via sister READMEs. |

## Fleet sync points involving this rig

- **docker-scoring-handoff** — participants: M4-D, 3090-B; trigger: the callout
  is committed with the resolved drop-dir; action: 3090-B runs score_docker.py
  under their score lock and commits the summary back here.
- **baseline-schema-v2-to-sister-rearm** — participants: 3090-D, R9700, M4;
  trigger: 3090-D commits schema v2 and arms `benchmarks/baselines.json`;
  action: this rig re-arms its baselines on its own depth-verified instrument
  (~32K with oom_guard).
