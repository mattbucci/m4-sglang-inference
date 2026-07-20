# Experiments — M4 Pro MLX (m4)

Vetted execution queue. Each doc is a self-contained spec: a fresh agent starts
at Method step 1. The table is the execution order; statuses here are
authoritative.

| Order | ID | Spec | Status | Why now / gates |
|:-:|---|---|---|---|
| 1 | disk-triage | (no spec — ~1h task) | **ready, do first** | 12 GiB free of 926 GiB; HF cache holds 362 GB. Inventory `~/.cache/huggingface/hub` per-repo sizes + last access, propose deletions to Matt (gemma4 variants and superseded dupes are candidates). Gates M4-E and comfortable bisect arms. |
| 2 | beyond-128k | [Deep-prefill memory budget — phase 2](08-beyond-128k.md) | ready | 160K is validated (patch 008 cache cap + CHUNKED=2048 + patch 015 pre-size; `benchmarks/longctx-bisect/ATTRIBUTION.md`). 192K exhausts the budget at ~180K prefilled — the bf16 per-request attention cache is the dominant unquantized, uncapped term. Phase 2: quantize it (also attacks decode TPOT at depth, 13-19 s/token) or pool-backed prefill writes. Related envelope receipts: radix-off concurrent prefill (conc-8) and dense-devstral genuine-32K both trip the oom_guard; radix-on retains ~0.4 GB/request outside the pool (`benchmarks/radix-ab/VERDICT.md`). |
| 3 | re-measure | (no spec — qwen36 SWE-bench re-run) | ready | The SWE-bench cell is from the earlier stack pin (dedicated opencode-harness session). Depth/concurrency re-measure is done: eight presets characterized at genuine depth and armed in `benchmarks/baselines.json` (schema v2); remaining legacy artifacts stay flagged in `benchmarks/LEGACY-DEPTH-SUSPECT.md` (coder-next infeasible, gemma4* blocked at boot, non-DWQ dupes pending disk triage). |
| 4 | M4-E | [In-house qwen36 MLX 4-bit with exclusions](07-qwen36-inhouse-mlx4bit-exclusions.md) | **blocked: disk** | Needs ~70+ GB (67 GB BF16 + conversion scratch). Motivation is the DWQ-quality precedent; run its probe A/B after M4-B lands. |
| 5 | prefill-budget-ledger-192k | [192K knob probes + GB-shortfall ledger](11-prefill-budget-ledger-192k.md) | ready | S-effort, no patch, no download — runs in any exclusive window regardless of disk-triage. The 192K death (guard-killed ~180K at CTX=210K mf 0.5; `benchmarks/longctx-bisect/ATTRIBUTION.md` row 34) was never probed with sub-2048 chunks, an exact-196,608 pool (the dead run overallocated ~14K tokens of pool), or MEM_FRAC 0.45; any arm may buy the missing ~2-3 GB, and the failed arms produce the GB-shortfall ledger beyond-128k phase 2 needs as its minimum-recovery target. Piggybacks the stale "Certified ceiling: 128K" ATTRIBUTION.md fix. |
| 6 | depth-recall-probe | [Multi-needle recall ladder to 157K + KV A/B](10-depth-recall-probe.md) | ready | The campaign validates that prefill completes, never that the model retrieves at depth: the only recall instrument is single-needle, run in practice at 1024-16384 (`scripts/eval/eval_and_chart.py:347`), and turboquant KV has never been A/B'd vs fp16 on this rig. Supplies the drift gate beyond-128k phase 2's kill criterion presumes (no implementation exists) and the quality-parity gate order 7 needs — land before or alongside both. If recall collapses <=128K, the 192K/256K campaign is optimizing a ceiling the model cannot use. |
| 7 | decode-tpot-truth-and-depth-curve | [Fix the contaminated TPOT instrument, publish the true decode depth curve, go/no-go the decode-topk port](09-decode-tpot-truth-and-depth-curve.md) | ready | The fleet-cited 13-19 s/token decode wall is arithmetic-provably ~95-97% prefill amortization: `scripts/bench/bench_long_context.py:58` divides whole-request elapsed by out-tokens, and the 160K receipt's TPOT=18965.2ms is exactly 606.9 s prefill / 32 tokens. Five doc locations repeat the number and beyond-128k phase 2 is planned against it. Fix the instrument, publish the streamed/server-log curve at 5 depths, then go/no-go the MLX decode-topk port (R9700 069 1.77x / 3090 059 2.03x precedent) on measured data — absorbs the port proposal. |
| 8 | agentic-endurance-and-extend-tax | [Session endurance + append-to-cached-prefix turn tax](12-agentic-endurance-and-extend-tax.md) | ready | The daily workload's two unowned risks on one axis: radix-on retention characterized to exactly 10 requests (~0.4 GB/request outside the pool; `benchmarks/radix-ab/VERDICT.md` open thread) — if linear, a 50-100-turn session crosses the 8 GB guard floor and stalls the box; and R9700's R97-J turn tax (604.6 ms TTFT to append 1 token to a 176K cached prefix, ~85x decode) with a direct fleet request to check on all rigs. Delivers the cross-rig note together with the queued buffer-cache-cap note below. |
| — | upstream-candidates | (no spec — repo-only) | ready | Offer upstream / sister repos the generic fixes: declared-tokenizer-class guard, `wrap_as_pixtral` mistral3 + top-level `spatial_merge_size`, pixtral MULTI_IMAGES splitter, mamba-radix `no_buffer` resolution on MLX, device-aware FutureMap stash, **the MLX buffer-cache cap** (unbounded-cache prefill growth affects every MLX serving stack). |

Scheduling constraint: the box serves one model at a time — beyond-128k, M4-E and any
probe sweep need exclusive serving windows; the repo-only items don't.

## Cross-rig edges

| From | To | What moves |
|---|---|---|
| M4-D | 3090-B | Executable Docker-scoring callout in the 3090 README (corrected score_docker.py command, runtime bound, result-return path). Shared drop-dir: `evals/swebench/m4-imports/qwen36-m4/` (3090-B's name wins as the executor). |
| 3090-B | M4-D | `scores-docker-summary.json` (first full-26 official resolve rate) committed back to this repo (`exports/qwen36-docker-summary.json` + README headline update), replacing the 5/13-local-subset cell. |
| beyond-128k | 3090 / R9700 | The buffer-cache-cap finding (unbounded MLX cache retained ~0.6 MB/token across chunked prefill; not a lib regression) is MLX-specific but the method — cap the allocator cache before bisecting libs — is worth a sister-README note. |
| agentic-endurance-and-extend-tax | 3090 / R9700 | Turn-tax comparison note (R97-J asked all three rigs: append-1/append-64 TTFT vs decode ms/token at matched cached-prefix depths), delivered together with the queued buffer-cache-cap note above. |

## Fleet sync points involving this rig

- **docker-scoring-handoff** — participants: M4-D, 3090-B; trigger: the callout
  is committed with the resolved drop-dir; action: 3090-B runs score_docker.py
  under their score lock and commits the summary back here.
- **baseline-schema-v2-to-sister-rearm** — participants: 3090-D, R9700, M4;
  trigger: 3090-D commits schema v2 and arms `benchmarks/baselines.json`;
  action: this rig re-arms its baselines on its own depth-verified instrument
  (~32K with oom_guard).
