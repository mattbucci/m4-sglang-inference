# Apple Silicon Inference: SGLang + MLX on M4 Pro

Long-context LLM inference on Apple M4 Pro (Mac mini, 64 GB unified memory) using SGLang with a native MLX backend. Stack: **SGLang v0.5.23** (commit `297e54f`) + 13 patches ([patches/README.md](patches/README.md)). Text and VLM/hybrid paths are validated: `qwen36` is the primary agentic model (codegen STRONG, vision STRONG, video STRONG, thinking VERIFIED); `coder-30b`/`qwen3-moe`/`qwen3-32b`, `qwen35`, `devstral`, `nemotron-30b`, and `flint` (Recursive's coding-agent MoE, oQ8e) all pass their gates. Hybrid (DeltaNet/Mamba2) presets and `flint` run with the radix cache (hits and cold runs each deterministic; trade-off: overlap schedule off for them). `gemma4*` is blocked at boot (mlx-vlm 0.6.5's Gemma 4 attention is not recognized by the MLX attention contract, so its layers are classed as auxiliary state and the runner's hybrid component extraction fails on `lm_head`).

**Long-context: 256K single-user context validated on qwen36 — and usable**: the multi-needle recall ladder scores 6/6 at every rung from 9.6K to 245,656 realized tokens (`benchmarks/quality/depth-recall/`), (the `--kv-cache-dtype` flag is inert on this backend — the MLX pool stores KV in the attention compute dtype — so the earlier turboquant-vs-fp16 recall A/B compared identical bf16 arms). Recipe: exact pool sizing (`CTX = label + 64`), `MEM_FRAC=0.5`, `CHUNKED=1024`, radix off, turboquant, on patch 008 (buffer-cache cap) + patch 015 (cache pre-sizing). in=251,659 server-verified tokens, ~23 min prefill — receipts in `benchmarks/longctx-bisect/`. The open deep-context constraint is decode TPOT at depth (decode-tpot-truth queue item).

## Action queue

Full specs and statuses live in [experiments/README.md](experiments/README.md); one-line summary, in execution order:

- [x] **Disk triage** — 213 GiB free of 926 GiB after the HF cache was thinned (hub 120 GB + 39 GB content store incl. the three Flint checkpoints); the in-house quant and bisect arms are no longer disk-gated.
- [x] **SWE-bench re-run** — the 3-instance smoke on the current stack reproduces the qwen36 cell (3/3 patches, same files, 56–78 s per instance with the prefix cache: `evals/swebench/runs/qwen36-v0523-2026-10-31/`); the full scorecard stays the earlier stack pin's. Depth/concurrency re-measure is complete: eight presets at genuine depth, all armed as schema-v2 baselines (see Performance).
- [ ] **In-house qwen36 MLX 4-bit** with router/DeltaNet/vision exclusions — after disk + sampling. Spec: [experiments/07](experiments/07-qwen36-inhouse-mlx4bit-exclusions.md). *(days)*
- [ ] **192K knob probes + GB-shortfall ledger** — sub-2048 chunks / exact-196,608 pool / MEM_FRAC 0.45 against the 192K death; failed arms yield the exact GB number phase 2 must recover. No patch, no download. Spec: [experiments/11](experiments/11-prefill-budget-ledger-192k.md). *(15-45 min/arm)*
- [ ] **Depth-recall probe** — multi-needle positional recall to 157K + a standing 32K recall tripwire (the KV-dtype A/B is moot: `--kv-cache-dtype` is inert on MLX); supplies the drift gate beyond-160K presumes. Spec: [experiments/10](experiments/10-depth-recall-probe.md). *(~2-3h)*
- [ ] **Session endurance + extend turn tax** — radix-on retention past 10 requests and append-to-cached-prefix TTFT on the daily agentic workload (R97-J cross-rig request). Spec: [experiments/12](experiments/12-agentic-endurance-and-extend-tax.md). *(~4-6h)*

## Primary target: long-context agentic coding

Single-user agentic coding at long context (tool-call-heavy multi-turn sessions, 10K–100K-token codebase prefixes) is what this stack is tuned for. Decode TPOT at long context matters more than peak batch throughput; the right model is one whose decode stays flat as context grows.

### Recommended picks (ranked by SWE-bench Lite agentic-coding ability)

**`qwen36` is the only configuration verified to produce real SWE-bench Lite patches on M4, reproducibly, at scale, across all 12 ecosystems in our test set.** Scorecard measured on the previous stack pin; the current stack reproduces its 3-instance smoke (3/3, same target files, `evals/swebench/runs/qwen36-v0523-2026-10-31/`):

- **Patch-engagement: 21/26 (80.8%) across 12 ecosystems** — model produces a non-empty patch targeting the canonical issue file.
- **Resolved (official Docker harness, scored on the 3090 rig): 9/26 = 34.6%** (9/21 = 42.9% of non-empty; 0 errors — all 21 non-empty ran, incl. the 8 old-Python instances M4 can't build). All 5 M4-local resolves re-resolved (no flips); 4 additional resolves came from the M4-untestable class. Receipt: [`exports/qwen36-docker-summary.json`](evals/swebench/exports/qwen36-docker-summary.json) + [exports/README.md](evals/swebench/exports/README.md) Results.

**Session endurance (radix-on agentic serving):** run a `/flush_cache` every ~25 turns — unmitigated radix-on sessions cross the 8 GB guard floor at ~100 turns under a 32K-rolling load, while the flush cadence holds 150+ turns at ~1.2% amortized latency cost; appending a tool result to a cached prefix costs ~45 ms at 8K / ~120 ms at 32K (`benchmarks/session-endurance/VERDICT.md`).

**The 80.8% → 34.6% gap is the real model characterization**: qwen36 makes patches that LOOK right (correct file, syntactically valid, often no regressions) but miss the semantic intent in most cases. Trust the model for first-pass exploration; verify with tests before merging. 34.6% sits above the typical 14–30% resolved-rate band for ~30B-class models on SWE-bench Lite, but below the pre-registered >50% bar that would have "confirmed" the recommendation — qwen36 stays recommended as the only model that completes the agentic loop on M4, with calibrated expectations.

The empty-patch instances (`django-11019`, `flask-4045`, `sphinx-10451`, `requests-2148`, `sympy-11870`) are "model gave up" cases — typically TIMEOUT after a handful of tool calls without a diff; the characterized ones share an **"add behavior" vs "fix visible behavior"** signature.

**Method note for multi-instance sweeps:** at CTX=131K, multi-instance single-server sweeps hit recurring macOS jetsam (the scheduler is reaped silently). `evals/swebench/run_rollouts.py` does a per-instance preflight and aborts cleanly on dead upstream; for sweeps beyond 1-2 instances use **per-instance server restart** (one `bash evals/swebench/smoke.sh` per instance, ~30s overhead each). Static HE/MMLU scores do not predict agentic-coding capability on this stack. Receipts: [`4pick-scorecard-2026-05-18/`](evals/swebench/runs/4pick-scorecard-2026-05-18/) (bake-off), [`qwen36-3instance-2026-05-18/`](evals/swebench/runs/qwen36-3instance-2026-05-18/) (reproducibility), [`qwen36-crossrepo-2026-05-18/`](evals/swebench/runs/qwen36-crossrepo-2026-05-18/) (generalization).

| Rank | Preset | Why | Agentic verdict |
|:----:|--------|-----|-----------------|
| **1** | **`qwen36` (Qwen3.6-35B-A3B-4bit MoE+DeltaNet)** | Only model to complete the agentic loop on M4, verified against a full bakeoff of 9 alternatives (decode-rate misses, parser mismatches, template failures — receipts in the bake-off run dir). MoE+3B-active decode speed appears load-bearing — Dense at 27B can't keep the agent loop alive within timeouts. Vision-capable. Use with [`no_thinking_proxy`](evals/swebench/no_thinking_proxy.py); per-instance server restart for multi-instance sweeps. | Patch-engagement 21/26 (80.8%); **official Docker 9/26 = 34.6%** (scored on the 3090; no local-vs-Docker flips). |
| 2 | `qwen35` (Qwen3.5-27B-4bit DeltaNet) | **Capability-equivalent to qwen36, not higher.** Succeeds where qwen36 succeeds (same patch at ~15× the wall time) and fails where it fails, including the algorithmic-rewrite miss at TIMEOUT=1800. Static MMLU 90 does not convert to a higher agentic ceiling. No agentic value over qwen36. | SWE-bench Lite 1/3 |
| 3 | `coder-30b` (Qwen3-Coder-30B-A3B-4bit-DWQ) | Best static HumanEval (95) and decode speed — use for **direct chat-completion codegen**, not agentic flows: under greedy MLX + opencode the agent loop gives up after one `glob`. | 1 glob then asks user, 0 edits |
| 4 | `gemma4-31b` (gemma-4-31b-it-mxfp4) | Top MMLU (92) + Needle 100. **Currently blocked at boot** (upstream sliding-window gap — see Known Issues). Historically unusable through opencode anyway (zero tokens under tool-call prompts). | 0 tool calls under opencode |

`gemma4` (26B MoE) is not in the recommendation set: its single prior RESOLVED did not reproduce across stack pins (single-trajectory coincidence, not capability).

### Flint (`flint`) — Recursive's coding-agent MoE

[RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) is a 39.8B-total / ~2.5B-active
decoder-only MoE trained for coding agents (its card reports Terminal-Bench 2.1
54.9% and SWE-bench Verified 74.2% at bf16): 49 layers (2 dense + 47 MoE with
256 routed experts, top-6, plus a shared expert), 35 sliding-window layers
(513 tokens) + 14 full-attention layers, learned per-head attention sinks, no
RoPE on the full layers, 262K context, Apache-2.0. It runs through mlx-lm's
`flint` model (git pin in `setup.sh`) plus patch 019; the oMLX `oQ*e`
checkpoints (mixed precision, imatrix-calibrated clipping) are converted
through mlx-vlm's model class, which is why the patch carries a
checkpoint-layout adapter.

| Variant | bpw | Resident | Decode tok/s @83 / @3.5K / @13.9K / @27K | Prefill tok/s @3.5K / @13.9K / @27K | Validator / codegen / thinking | Greedy vs mlx-lm (96 tok) |
|---------|:---:|:--------:|:----------------------------------------:|:-----------------------------------:|:------------------------------:|:-------------------------:|
| **`flint-oQ8e`** (default) | 8.5 | 42.3 GB | 61.9 / 59.1 / 54.4 / 49.3 | 926 / 809 / 634 | 3/3 · STRONG · VERIFIED | identical |
| `flint-oQ6e` | 6.6 | 32.9 GB | 64.8 / 61.2 / 56.0 / — | 760 / 638 / — | 3/3 · STRONG · VERIFIED | identical |
| `flint-oQ4e` | 4.7 | 23.5 GB | 70.5 / 67.5 / 61.5 / — | 708 / 595 / — | 3/3 · STRONG · VERIFIED | identical |

The table is measured with the radix cache off; the oQ8e row uses the
preset's `CHUNKED=1024`, 2 GB MLX buffer cache and `MAX_RUNNING=4`, the
oQ6e/oQ4e rows `CHUNKED=4096` with the 4 GB cache. The preset as shipped
runs the radix cache (patch 021), which keeps prefill where it is (924 /
794 / 645 tok/s at 3.5K / 13.9K / 27K) and costs ~12% of decode because
the hybrid `no_buffer` bookkeeping turns the overlap schedule off
(50.3 / 47.0 / 43.1 tok/s at the same depths); the 27K prefill then leaves
12.8 GB kernel-available instead of 17 GB
(`benchmarks/flint/2026-10-31/prefix-cache/RESULTS.md`). The 42.3 GB of oQ8e weights sit under the 51.8 GB Metal
working set, and the serving envelope is set by prefill transients rather
than weights: a 27K-token prefill at chunk 1024 leaves the kernel reporting
17 GB available (`kern.memorystatus_level`, the figure jetsam acts on) and
never leaves pressure level "normal"; the preset-style chunk 2048 with a 4 GB
cache leaves 15 GB and is slower at depth (585 vs 634 tok/s). vm_stat's
"free" reads 0.0 throughout, which is why the OOM guard now reads the kernel
level instead of free+inactive (see Known Issues). Four concurrent requests
decode at 123 tok/s aggregate; batched output is not bit-identical to
single-request output (the usual non-batch-invariance of batched kernels),
while single-request greedy output is token-identical to standalone mlx-lm
on all three quantizations.

- **Reasoning:** Flint decides per turn whether to write `<think>…</think>`;
  there is no template toggle (`enable_thinking` has no effect).
  `--reasoning-parser flint` returns the block as `reasoning_content` and
  never assumes a turn starts in reasoning; keep `reasoning_content` on
  assistant messages when sending the conversation back. Reasoning counts
  against `max_tokens`: give coding requests ≥ 4,096 (Recursive recommends
  ≥ 32,768 for hard problems). The 400-token default of `probe_codegen.py`
  truncates its second task mid-code (`finish=length`), so the probe sweep
  runs Flint with a 4,096-token budget.
- **Tool calls:** `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
  (GLM-4.7's format without line breaks); `--tool-call-parser flint` returns
  structured `tool_calls` (validator PASS). The constrained tool-call grammar
  of Recursive's CUDA kit (`--grammar-backend flint`) is not ported. Under
  opencode, Flint omits the `bash` tool's required `description` argument on
  nearly every call (0–3 of ~12 whichever way the rule is phrased in the
  prompt; see Known Issues), so opencode rejects the call; the SWE-bench
  proxy backfills it (`BACKFILL_BASH_DESCRIPTION=1`).
- **Sampling:** the preset applies the checkpoint's `generation_config`
  (temperature 1.0 / top-p 0.95 / top-k 20) to requests that omit sampling
  fields; greedy requests stay greedy and pass every gate here, although
  Recursive validates only the sampled setting.
- **Prefix cache:** on (patch 021). The 14 full layers share the pool
  (28 KB/token, 3.7 GB at the preset's 131,072 slots) and the 35 windowed
  caches are snapshotted at the prefix boundaries the radix tree names and
  restored on an exact hit, so an agent turn prefills only its new tokens:
  turn 2 of a 13.4K-token conversation starts in 0.6 s instead of 19 s
  (`benchmarks/flint/2026-10-31/prefix-cache/`). The pool costs ~2 GB of
  headroom relative to the radix-off preset (cold 13K-token prefill:
  16.0 vs 18.6 GB kernel-available minimum); `EXTRA_ARGS="--disable-radix-cache"`
  restores the pool-free recompute path. Set `MAX_TOTAL_TOKENS=131072`
  explicitly for CTX=131072 runs (`CTX * MAX_RUNNING` would be a 14.7 GB
  pool). Per-request KV is 28 KB/token on top of the pool: 0.9 GB at 32K.
- **Agentic cell:** the SWE-bench Lite run (opencode harness, `flint` entry
  in the opencode config) is queued as experiments row 5; `flint` is not
  ranked in the recommended-picks table until that cell exists.

### Probe matrix (current stack, radix on)

| Preset | codegen | vision | video | thinking |
|--------|:-------:|:------:|:-----:|:--------:|
| `qwen36` | **STRONG** | **STRONG** | **STRONG** | **VERIFIED** |
| `qwen35` | **STRONG** | **STRONG** | PARTIAL | **VERIFIED** |
| `devstral` | **STRONG** | **STRONG** | PARTIAL | n/a |
| `nemotron-30b` | **STRONG** | n/a | n/a | **VERIFIED** |
| `flint` (oQ8e / oQ6e / oQ4e) | **STRONG**† | n/a | n/a | **VERIFIED** |
| `coder-30b` / `qwen3-moe` / `qwen3-32b` | **STRONG** | n/a | n/a | pass validate gate |
| `qwen35-9b-8bit` / `qwen36-27b` | unswept (same code path as siblings) | | | |
| `gemma4` / `gemma4-31b` | blocked at boot (mlx-vlm Gemma 4 attention not recognised by the MLX attention contract) | | | |

Per-preset JSON: [`benchmarks/quality/probe-trio/`](benchmarks/quality/probe-trio/).
† Flint's codegen probe runs with a 4,096-token budget (`probe_all.sh` sets it): the model reasons before answering, and the 400-token default truncates its second task mid-code.

### Choosing a model

**MoE wins at long context.** Each decode token must (1) read model weights and (2) read the entire KV cache. At short context, weight loading dominates → MoE reads 1.5 GB vs Dense 14 GB (4× faster). At 256K with fp8, the KV read climbs to ~5–10 GB — comparable to dense weights — so MoE keeps the weight component small and the KV penalty proportionally less painful.

**DeltaNet hybrids** (Qwen3.5/3.6) alternate standard attention (O(n)) with linear attention (O(1)). Linear layers don't slow with context — architecturally suited for very long context — but the standard layers in the hybrid still pay full O(n).

## Quality table (100-sample MMLU + 20 HE + 25×7 LAB-Bench + Needle@{1K,4K,16K})

Measured on the previous stack pin with the jetsam-detect-hardened harness (re-run on the current stack queued). Qwen3 family uses `--no-thinking`; Gemma 4 family uses `--humaneval-mode chat` (IT-tuned Gemma 4 doesn't respond to bare base completions).

| Model | MMLU | HumanEval | LAB-Bench | Needle |
|:------|:----:|:---------:|:---------:|:------:|
| Gemma 4 31B-it-mxfp4 | **92%** | 50%‡ | partial⁂ | **100%** |
| Qwen3.5-27B-4bit | **90%** | **100%** | 53/125‡‡ (clean cats) | **100%** |
| Qwen3-32B-4bit-DWQ | **90%** | 95% | 33.1% | 100% |
| Qwen3.6-27B-4bit | 86% | **100%** | 42/125‡‡ | **100%** |
| Qwen3-30B-A3B-4bit-DWQ | 85% | 70% | 31.4% | 100% |
| Gemma 4 26B-A4B-it-4bit | 85% | 60%‡ | partial⁂ | 100% |
| Coder-30B-A3B-4bit-DWQ | 84% | 95% | 30.9% | 100% |
| Qwen3.5-9B-MLX-8bit | 81% | 80% | 52/150‡‡ (clean cats) | **100%** |
| Qwen3.6-35B-A3B-4bit | 80% | 85% | 60/175 | 100% |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | 82% | 65%❋ | 8.6%❋ | 100% |
| NVIDIA Nemotron-3-Nano-30B-A3B-4bit | 77% | 10%¶ | 19.4%¶ | 100% |
| Devstral-24B-4bit | 71% | 55% | 34.3% | 100% |

‡ Gemma 4 HumanEval runs in `--humaneval-mode chat` — not comparable to base-completions HE.

‡‡ LAB-Bench partial-data score: the full 175-sample run hit jetsam mid-eval, so the score covers only cleanly-completed categories (N/M shows the subset).

⁂ LAB-Bench partial: server died early in the run; only LitQA2 + partial DbQA ran clean. Full LAB-Bench needs a server restart between categories (follow-up on `run_all_evals.sh`).

❋ Nemotron-Omni's reasoning-mode wrapper consumes a chunk of the LAB-Bench answer budget on multi-letter QA; HE/MMLU gain comes from the same reasoning capability.

¶ Nemotron-3-Nano emits verbose reasoning traces; the `nemotron_3` reasoning parser is wired into the preset — these pre-parser numbers should rise on re-run.

## Checkpoint audits

Two complementary pre-launch gates — run both on every new checkpoint before adding numbers here:

```bash
python scripts/eval/check_mlx_quant_scales.py <repo>    # per-layer corruption (weights/scales/biases dequant to dead output)
python scripts/eval/audit_mlx_quant_metadata.py         # recipe hazards (wrong module classes quantized) — header-only, no download
```

**Per-layer scan:** `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` (non-DWQ) has **10 dead layers** (`model.layers.{36,46}.*` attention + gate all-zero) — the reason the preset points at the DWQ variant. The other shipped checkpoints scan clean. Raw output: [`benchmarks/quality/v0.5.11-quant-scan-2026-05-11.txt`](benchmarks/quality/v0.5.11-quant-scan-2026-05-11.txt).

**Recipe-metadata audit across the shipped set** (raw: [`benchmarks/quality/mlx-metadata-audit-2026-05-13.txt`](benchmarks/quality/mlx-metadata-audit-2026-05-13.txt)):

| Checkpoint | Recipe-level hazard |
|------------|---------------------|
| `gemma-4-26b-a4b-it-4bit` / `gemma-4-31b-it-mxfp4` | `embed_vision.embedding_projection` quantized — the exact layer class sister teams' calibration disaster zero-scaled. |
| `Qwen3.5-27B-4bit` / `Qwen3.5-9B-MLX-8bit` / `Qwen3.6-27B-4bit` | DeltaNet `linear_attn.in_proj_a/b` quantized across all layers — recurrent-state gate scalars that sister-team rules say must stay BF16. Candidate contributor to the Qwen3.5 infinite-`<think>` loop. |
| `Qwen3.6-35B-A3B-4bit` | DeltaNet in_proj INT4 (60 layers) **and** MoE `mlp.gate` router INT4 (40 layers). |
| `Qwen3-Coder-30B-A3B-4bit-DWQ` / `Qwen3-30B-A3B-4bit-DWQ` / `Qwen3-Coder-Next-4bit` | MoE router `mlp.gate` INT4 — bounded impact in practice. |
| `Devstral-24B`, `Qwen3-32B-DWQ`, `Nemotron-3-Nano-30B-A3B` | **clean** |
| `flint-oQ8e` / `flint-oQ6e` / `flint-oQ4e` | **clean** — router (`mlp.gate`), attention sinks and norms stay bf16; the per-layer scan reports all 486 quantized layers healthy on oQ8e and oQ6e. The only recipe quirk is mixed precision: 180 shared-expert projections use group size 128 (the rest 64), which the loader honours per tensor. |

**DWQ variants are used where they measured better** — `coder-30b` (+2.8 MMLU / +20 HE over the broken 4bit), `qwen3-moe` (+7.9 MMLU / −5 HE), `qwen3-32b` (+2.8 / +7.5). `qwen36` stays on standard 4bit (its DWQ trades −5.5 MMLU for +15 HE; `MODEL="mlx-community/Qwen3.6-35B-A3B-4bit-DWQ" launch.sh qwen36` overrides when code-specialist behavior is wanted). **Never blind-swap DWQ — recipes vary per upload; measure MMLU + HumanEval per model.**

## Known Issues

- **Sampling caveats.** Real per-request sampling (temperature/top-p/top-k/min-p, logit_bias, grammar masks, logprobs) is upstream on the MLX path behind `--mlx-enable-sampling` (every preset sets it); greedy stays bit-stable argmax and `--sampling-defaults model` applies checkpoint generation configs. Caveats: `top_k==1` is served greedy, and repetition/frequency/presence penalties are not applied. Earlier gate receipts: [benchmarks/sampling-ab.md](benchmarks/sampling-ab.md).
- **`gemma4` / `gemma4-31b` blocked at boot** — the per-layer sliding-window map now resolves on this pin, but mlx-vlm 0.6.5's Gemma 4 attention is not recognized by the MLX attention contract, so its layers are classed as auxiliary state and the runner's hybrid component extraction fails on `lm_head` (`AttributeError: 'LanguageModel' object has no attribute 'lm_head'`). Needs the attention contract to recognise mlx-vlm's Gemma 4 attention module (or a Gemma 4 entry like the NemotronH adapter); receipt in `benchmarks/quality/probe-trio/gemma4.json`.
- **Long-context ceiling is 256K** (validated: in=251,659, ~23 min prefill; recipe: `CTX=<label+64> MEM_FRAC=0.5 CHUNKED=1024`, radix off; the `--kv-cache turboquant` flag in the original recipe is inert, see below). Four load-bearing settings: the patch-008 MLX buffer-cache cap (`SGLANG_MLX_CACHE_LIMIT_GB`, default 4 GB — uncapped, chunked prefill retains ~0.6 MB/token and jetsams ~30K), patch-015 cache pre-sizing (the doubling ladder's power-of-two overshoot kills deep runs), EXACT pool sizing (overallocating CTX costs the last GBs — a 192K label at CTX=210K dies where CTX=196608 completes), and `CHUNKED=1024` (the 4096 default's transients kill deep prefills at ~100-113K; 1024 buys ~4.3 GB margin over 2048 at no speed cost). There is no decode wall: true streamed decode is 61.7 tok/s @32K → 36.0 @128K → 30.2 @151K (`benchmarks/longctx-bisect/decode-curve/DECISION.md`); the historical "13-19 s/token" figures were whole-request prefill amortization, since corrected in `bench_long_context.py`.
- **`--kv-cache-dtype` is inert on the MLX backend.** The `fp8`/`mxfp8`/`turboquant`/`fp16` aliases (patch 002) exist so the launch presets pass argparse; the MLX attention pool and the per-request caches store KV in the attention compute dtype (bf16 for the mlx-community checkpoints) at every pin this stack has run on. Memory levers that do work: `--max-total-tokens`, `CHUNKED`, `MAX_RUNNING`, request `max_tokens`, `SGLANG_MLX_CACHE_LIMIT_GB`. 4-bit KV on Apple Silicon remains a prospect ([MLX discussion](https://github.com/ml-explore/mlx/discussions/3134)), not a shipped lever.
- **64K dense-attention ceiling is structural.** The per-chunk attention-score tensor `mx.fast.scaled_dot_product_attention` materializes hits 30–90 GB at deep offsets. Without flash-attention-style block-streaming SDPA in MLX, decode at 32K is the practical ceiling for dense standard-attention models; DeltaNet hybrids (`qwen35`, `qwen36`, `qwen36-27b`) are the workaround.
- **Radix-on-hybrid disables the overlap schedule** (upstream `no_buffer` constraint) — hybrids and `flint` run the normal event loop. Prefix-cache wins dominate for agentic multi-turn work; decode A/B is queued.
- **Flint omits opencode's required bash `description` argument.** opencode's `bash` tool schema requires `description` next to `command`; Flint (trained on single-argument shell tools) sends `command` alone on nearly every call under opencode's real prompt, opencode rejects the call with a `SchemaError`, and the agent retries the same call until its timeout. No prompt placement of a rule fixes it (0–3 of ~12 calls; a short prompt gives 10/10). The SWE-bench harness proxy backfills the argument from the command (`BACKFILL_BASH_DESCRIPTION=1`); interactive opencode use needs the same shim in front of the server or the bash tool disabled (`tools: {bash: false}`).
- **Coder-Next-80B infeasible** — 42 GB weights alone exceed the safe budget; model load OOMs. No path on a single 64 GB Mac.
- **macOS has no OOM killer** — a process touching a page past physical RAM stalls the whole system until reboot. **OOM guard mandatory for ≥32K work:** `bash scripts/common/oom_guard.sh &` pkills the server when the kernel-reported available memory (`kern.memorystatus_level`, the percentage jetsam decides on) drops below 8 GB. It used to read vm_stat free+inactive, which undercounts by the file pages a checkpoint load streams through the page cache: a 42 GB Flint load read 6.9 GB on that metric while the kernel reported more than 15 GB available, and the guard reaped a load that completes fine.
- **macOS jetsam can silently reap the scheduler mid-eval** — no traceback, just `Connection refused` on subsequent requests. `needle_eval` tags `server_dead=True` on connection-class errors; `mmlu_eval`/`humaneval_eval`/`labbench_eval` don't yet. When an eval section drops to ~0%, suspect jetsam before model regression: relaunch fresh and re-run that section.
- **`--mem-fraction-static` is a fraction of TOTAL system RAM on unified memory**, not "GPU memory". 0.85 has hard-locked the box (compressor + swap ~150 GB effective). The 0.7 default ceiling is load-bearing; long-context presets go *down* to 0.4.
- **HDMI display blackout** — brief screen blank when heavy Metal compute starts. M4 Pro quirk, not an SGLang bug.
- **`/v1/responses` is disabled at boot** — SGLang v0.5.23 pins `openai==2.6.1` but its Responses serving module imports a type that only newer `openai` releases ship (`ResponseFunctionShellToolCall`); the server logs the ImportError and serves chat completions normally. `pip install -U openai` restores the endpoint at the cost of the pin.

## Quick Start

```bash
./scripts/setup.sh                          # conda env, SGLang v0.5.23, MLX deps, 13 patches

# Validated presets
./scripts/launch.sh qwen36                  # PRIMARY — MoE+DeltaNet+VL, full probe matrix green
./scripts/launch.sh coder-30b               # MoE — fastest decode, direct chat-completion codegen
./scripts/launch.sh qwen3-moe               # Qwen3-30B MoE (DWQ, MMLU 91)
./scripts/launch.sh qwen3-32b               # Dense (DWQ, clean audit)
./scripts/launch.sh qwen35                  # DeltaNet hybrid+VL
./scripts/launch.sh devstral                # Dense+VLM (Mistral3)
./scripts/launch.sh nemotron-30b            # NemotronH (Mamba2+Attn+MoE)
./scripts/launch.sh flint                   # Recursive Flint 40B-A2.5B coding-agent MoE (oQ8e; see Flint section)

# Same code path as validated siblings, not yet swept:
./scripts/launch.sh qwen35-9b-8bit          # 10 GB resident tight-memory variant
./scripts/launch.sh qwen36-27b              # Dense DeltaNet+VL

# Blocked / not in rotation:
# gemma4, gemma4-31b — boot failure: mlx-vlm Gemma 4 attention not recognised by the MLX attention contract
# coder-next          — infeasible on 64 GB
# smol-docling        — 256M VLM smoke test only

# Long-context (128K validated; OOM guard mandatory for ≥32K)
bash scripts/common/oom_guard.sh &
CTX=140000 MEM_FRAC=0.5 CHUNKED=2048 EXTRA_ARGS="--disable-radix-cache" \
    bash scripts/launch.sh qwen36

# Agentic coding (qwen36 + opencode). The Qwen3 <think> template blocks break
# the opencode loop (reasoning_content invisible to opencode / stray </think>
# confuses the tool-call parser), so requests go through a proxy that injects
# chat_template_kwargs={"enable_thinking": false}:
bash evals/swebench/smoke.sh                # orchestrates server + proxy; defaults qwen36, 1 instance
# or manually:
bash scripts/launch.sh qwen36 &
python evals/swebench/no_thinking_proxy.py &   # opencode → 23335 → SGLang :23334
# (point ~/.config/opencode/opencode.jsonc baseURL at :23335)

# Capability gates (run AFTER server is up on PORT 23334)
python scripts/eval/validate_capabilities.py --port 23334   # basic + thinking gate (loose keyword grep)
python scripts/eval/probe_thinking.py        --port 23334   # content-aware reasoning probe
python scripts/eval/probe_vision.py          --port 23334   # content-aware image probe (STRONG/DEGRADED/FAIL)
python scripts/eval/probe_codegen.py         --port 23334   # 2-task / 8-test code-synthesis probe
bash   scripts/eval/probe_all.sh                            # sweep the probe trio across presets

# Pre-launch checkpoint audits (no server needed)
python scripts/eval/check_mlx_quant_scales.py   <repo-or-path>
python scripts/eval/audit_mlx_quant_metadata.py
python scripts/eval/validate_chat_template.py --model <path>

bash   scripts/bench/bench_256k_all.sh                      # long-context sweep (guard first!)
```

## Prerequisites

- Apple Silicon Mac with **≥ 64 GB unified memory** (M4 Pro Mac mini is the reference rig)
- macOS 26+ (Tahoe), Xcode CLT
- Python 3.12, `uv` or `venv`
- ~200 GB disk for models

## Model Support

| Preset | Checkpoint (mlx-community) | Type | Wts | 1-user tok/s* | Max ctx* | Audit hazards |
|--------|---------------------------|------|:---:|:------------:|:-------:|:-------------|
| `qwen36` | `Qwen3.6-35B-A3B-4bit` | MoE+DeltaNet+VL | 17 GB | 51.8 (148 MR=2) | **128K measured** (262K label) | router INT4 + DeltaNet INT4 |
| `coder-30b` | `Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ` | MoE (3B active) | 16 GB | 68.4 | 256K (3.2) | router INT4 |
| `qwen3-moe` | `Qwen3-30B-A3B-4bit-DWQ` | MoE (3B active) | 16 GB | 69.0 | 64K (6.3) | router INT4 |
| `qwen35` | `Qwen3.5-27B-4bit` | DeltaNet hybrid+VL | 15 GB | 14.3 (34 MR=2) | 256K label | DeltaNet `in_proj_a/b` INT4 |
| `qwen35-9b-8bit` | `Qwen3.5-9B-MLX-8bit` | DeltaNet hybrid+VL | 10 GB | — | 32K | DeltaNet INT4 (8-bit) |
| `qwen36-27b` | `Qwen3.6-27B-4bit` | DeltaNet hybrid+VL | 14 GB | — (34 MR=2) | 256K label | DeltaNet INT4 |
| `flint` | `flint-oQ8e` (oMLX; `MODEL=` for `flint-oQ6e` / `flint-oQ4e`) | MoE 40B-A2.5B, sliding-window + sinks | 42 GB | 61.9 (49.3 @27K) | 32K preset (262K label) | **clean** — router, sinks and norms unquantized; mixed-precision shared-expert projections (group 128) |
| `devstral` | `Devstral-Small-2-24B-Instruct-2512-4bit` | Dense+VL (Mistral3) | 14 GB | 17.0 (40 MR=4) | 256K (1.8) | **clean** |
| `qwen3-32b` | `Qwen3-32B-4bit-DWQ` | Dense | 18 GB | 12.1 | 16K | **clean** |
| `gemma4` | `gemma-4-26b-a4b-it-4bit` | MoE (4B active) | 15 GB | 58.8 | blocked | `embed_vision.embedding_projection` INT4 |
| `gemma4-31b` | `gemma-4-31b-it-mxfp4` | Dense (sliding+full) | 17 GB | 11.7 | blocked | `embed_vision.embedding_projection` INT4 |
| `nemotron-30b` | `NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` | NemotronH (Mamba2+Attn+MoE) | 17 GB | — | 32K probe | **clean** |

\* tok/s and Max-ctx cells were measured on earlier stack pins (re-measure queued) and pre-date the bench depth pin — depth-labeled cells are suspect ([inventory](benchmarks/LEGACY-DEPTH-SUSPECT.md)). The validated long-context ceiling is 128K (see Known Issues). MR=N numbers are batched-decode peaks. Audit hazards from the [checkpoint audits](#checkpoint-audits).

## Performance

> Mac mini M4 Pro (64 GB), SGLang + MLX, `sglang.benchmark.serving`.

### Genuine-depth context sweep (current stack, `--random-range-ratio 1`)

Decode tok/s with the server-verified depth instrument (every point's
`actual_input_tokens` == label). Tripwire recipe: `CTX=36000 MEM_FRAC=0.45
SGLANG_MLX_CACHE_LIMIT_GB=2 --disable-radix-cache` (bf16 KV; the `--kv-cache turboquant` the rows were run with is inert).
These rows are the regression baselines (`benchmarks/baselines.json`, schema v2).

| Preset | @1K | @8K | @32K | TTFT @32K |
|--------|:---:|:---:|:----:|:---------:|
| qwen36 | 82.4 | 72.8 | **57.8** | 57 s |
| coder-30b | 80.1 | 55.8 | 28.7 | 100 s |
| qwen3-moe | 78.0 | 55.7 | 28.8 | 99 s |
| qwen35-9b-8bit | 26.9 | 26.3 | 23.6 | 128 s |
| devstral | 16.5 | 14.8 | OOM | — |
| qwen35 | 15.0 | 14.6 | 12.8 | 307 s |
| qwen36-27b | 15.0 | 14.4 | 12.8 | 262 s |
| qwen3-32b | 12.3 | 10.7 | 7.5† | 494 s |

† qwen3-32b's 32K row needs `MEM_FRAC=0.5` (the 0.45 pool caps requests at
25,241 tokens). All eight rows above are armed as schema-v2 regression
baselines. **DeltaNet holds decode flat at genuine depth** across every
hybrid (qwen35/qwen36-27b retain 85% of short-context speed at true 32K;
qwen35-9b-8bit retains 88%) while full-attention models halve or worse.

Full curves (128→32K, 9 points) + concurrency rows live in each preset's
`benchmarks/<slug>/results.json`; serving-envelope receipts (radix-on 32K
prefill OOM, radix-off conc-8 OOM, dense 32K walls) are the same files'
error rows.

### Legacy tables (earlier stack pins)

> **Everything below was measured on earlier stack pins** and pre-dates the
> `--random-range-ratio 1` pin — labeled depths drew uniform [1,N] and are
> flagged suspect ([benchmarks/LEGACY-DEPTH-SUSPECT.md](benchmarks/LEGACY-DEPTH-SUSPECT.md)).
> Re-measure for the remaining presets is in the action queue.

### Short-sweep decode tok/s (fp16 KV, single user, 64 output tokens)

| Preset | tok/s @128 | @4K | @16K |
|--------|:----------:|:---:|:----:|
| qwen3-moe | **59.2** | 10.5 | 1.7 |
| coder-30b | **57.9** | 10.3 | 1.7 |
| qwen36 | **52.5** | 11.0 | 2.6 |
| gemma4 | **47.5** | 8.7 | n/a |
| qwen35-9b-8bit | 23.7 | 5.0 | 1.3 |
| devstral | 14.2 | 2.0 | 0.5 |
| qwen35 | 11.8 | 1.7 | 0.4 |
| qwen36-27b | 11.7 | 1.6 | 0.4 |
| gemma4-31b | 10.4 | 1.3 | n/a |
| qwen3-32b | 10.0 | 1.3 | 0.3 |

### Long-context turboquant sweep (`--chunked-prefill-size 1024 --mem-fraction-static 0.4`)

| Preset | KV | @128 | @4K | @8K | @16K | @32K |
|--------|----|:----:|:---:|:---:|:----:|:----:|
| coder-30b | turboquant | **73.8** | 66.6 | 55.2 | 43.0 | — |
| gemma4 | turboquant | 58.9 | 55.2 | 52.8 | 49.8 | **44.7** |
| devstral | turboquant | 17.4 | 17.1 | 16.2 | — | — |
| qwen35 | fp8 | 14.7 | 14.3 | 14.0 | 13.5 | **12.6** |
| gemma4-31b | turboquant | 13.5 | 12.7 | 12.4 | 11.7 | — |

`—` = OOM-guard tripped at that context. Two patterns: **MoE shapes the short-context win** (small active-weight reads); **DeltaNet keeps decode flat** (qwen35: 14.7→12.6 across the whole sweep — the O(1) linear-attention signature, and the load-bearing reason DeltaNet hybrids stay viable at long context on Apple Silicon).

The KV column records the flag each row was launched with; `--kv-cache-dtype` is inert on the MLX backend (bf16 KV in every row), so the "turboquant" and "fp8" rows differ only by model.

### Batched-decode peaks (single-server multi-prompt)

| Preset | Single user | Peak @ MR |
|--------|:-----------:|:---------:|
| `qwen3-moe` | 69 | **160 @ MR=8** |
| `qwen36` | 52 | **148 @ MR=2** |
| `devstral` | 17 | **40 @ MR=4** |
| `qwen35` / `qwen36-27b` | 12–14 | **34 @ MR=2** |

### Memory budget at 256K (64 GB Mac)

| Model class | Weights | KV @256K (bf16, what the MLX pool stores) | Fits? |
|-------|:-------:|:-----------------------------------------:|:------|
| MoE 3B-active (Coder-30B, Qwen3-30B-DWQ) | 16 GB | 24 GB | yes, with the long-context recipe |
| MoE+DeltaNet (Qwen3.6-35B-A3B) | 17 GB | varies† | yes (256K validated) |
| Flint 40B-A2.5B (oQ8e) | 42 GB | 7.5 GB (14 full layers, 4 KV heads) | 32K preset validated; deeper runs need the headroom measured per depth |
| Dense+VL (Devstral, Qwen3.5-27B) | 14–15 GB | 42 GB | no — and the dense 32K attention ceiling applies first |
| Dense (Qwen3-32B) | 18 GB | 66 GB | no |

† DeltaNet layers hold recurrent state instead of KV; only full-attention layers store KV.

## Setup

```bash
./scripts/setup.sh
```

Manually:
```bash
conda create -y -n sglang-0523 python=3.12 && conda activate sglang-0523
conda install -y -c conda-forge "rust>=1.88"   # SGLang builds a Rust extension at install time
git clone --depth 1 --branch v0.5.23 https://github.com/sgl-project/sglang.git components/sglang
cd components/sglang
for p in ../../patches/0[012][0-9]-*.patch; do git apply "$p"; done
cd python && cp pyproject_other.toml pyproject.toml
pip install -e ".[srt_mps]"
pip install "mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git@736ab0d8c1d0edc79c3d798044fe6904e2fa23be"   # Flint model (post-0.32.2)
pip install "mlx-vlm==0.6.5" --no-deps    # VLM loader; --no-deps keeps SGLang's transformers / mlx-lm pins
```

**Gate protocol (publish precondition):** every setup run — and any stack
change before its numbers are published — must pass
`scripts/test_patch_gates.sh` (pristine replay + byte-identity + double-apply
rejection; `setup.sh` runs it automatically and FATAL-aborts on any patch
failure), `scripts/eval/import_smoke.py` (eager import of every patch-touched
module + load-bearing symbols), and the behavioral probe gate
(`PRESETS="devstral qwen36" scripts/eval/probe_all.sh` — verdicts must match
`benchmarks/quality/probe-trio/`; a STRONG→FAIL flip is a stop-and-investigate
finding, the class that caught a month of fabricated VLM output). Receipts:
`benchmarks/patch-gates/`.

| Component | Version |
|-----------|---------|
| SGLang | **v0.5.23** (`297e54f`) + 13 patches |
| MLX | 0.32.4 |
| mlx-lm | git `736ab0d` (0.32.3.dev — carries the Flint model) |
| mlx-vlm | 0.6.5 |
| PyTorch | 2.13.0 (MPS) |
| transformers | 5.19.0 |
| Python | 3.12 |

## Patches

Thirteen patches on top of `v0.5.23` — full rationale per patch in [patches/README.md](patches/README.md). Per-request sampling is upstream at this pin (`--mlx-enable-sampling`, set by every preset).

| # | Patch | What |
|:-:|-------|------|
| 002 | mps-backend-defaults | MPS multimodal default + MLX `--kv-cache-dtype` aliases (boot-critical). |
| 003 | mlx-skip-quantization-check | Accept MLX checkpoints' `quantization_config` shape. |
| 005 | mlx-attn-wrapper-varargs | Attention-wrapper vararg + attribute delegation to the wrapped module. |
| 007 | mlx-multimodal-and-mps-shim | cuda→cpu redirect in the MPS tensor shim; shm page-rounding for multimodal data. |
| 008 | mlx-vlm-hybrid-integration | The VLM/hybrid path: mlx_vlm loader, attention detection, radix-for-hybrids (`no_buffer`), vision plumbing, NemotronH. |
| 014 | mlx-hf-processor-fixes | Gemma 4 image-only processor; Mistral3/Devstral processor + tokenizer resolution. |
| 015 | mlx-presize-attention-cache | Exact per-request attention-cache pre-sizing (replaces the doubling ladder for known-length requests). |
| 018 | mlx-aux-slot-admission-eviction | Admission-time auxiliary-slot eviction for the MLX pool (upstream's pre-pass is Hybrid-pool-only). |
| 019 | mlx-flint-support | Recursive Flint on the MLX path: mlx-vlm checkpoint-layout adapter, per-layer window map from `ModelArgs`, sinks + NoPE in batched decode, `FlintConfig`, `flint` parsers, Torch-registry non-gate on MLX. |
| 020 | mlx-prefill-last-token-logits | Final-chunk prefill computes logits for the last position only (drops the `[chunk, vocab]` transient: 1.2 GB at chunk 4096 on a 150K vocab). |
| 021 | mlx-swa-prefix-snapshots | Exact prefix cache for sliding-window models: the windowed caches are snapshotted through the auxiliary-state pool at the boundaries the radix tree names and restored on a hit; full layers gather from the pool; pool scatters evaluated per step. |
| 022 | mlx-pool-backed-cache-append | The pool-backed attention cache keeps the chunks already fetched in front of the next forward, so the trunk-then-last-token split of a prefix hit's final chunk sees every new token (the upstream cache re-gathered only the prefix: a blind first token and stale pool rows on every hit). |
| 023 | mlx-release-finished-request | The scheduler's KV-release hook runs for every finished request and the MLX worker drops the request's caches there, deferred while an overlap-schedule launch that includes the request is in flight (upstream sweeps them only at the next decode batch: a request that never decodes keeps its KV for the server's lifetime; the last turn's KV sits idle between agent turns). |

## Repo layout

```
patches/                    # 13 numbered patches — see patches/README.md
experiments/                # Vetted execution queue (specs + statuses)
benchmarks/                 # Per-model JSON + charts
  quality/                  #   MMLU / HumanEval / Needle / probe-trio verdicts
  <slug>/                   #   throughput + long-context sweeps
evals/swebench/             # Agentic-coding harness + dated run receipts
scripts/
  launch.sh                 # Unified launcher — launch.sh <preset>
  common.sh                 # Shared MLX env setup
  setup.sh                  # Full setup (venv, SGLang clone, patches)
  common/oom_guard.sh       # MANDATORY for ≥32K work — pkills server below 8 GB free
  common/mem_profile.sh     # CSV memory profile companion
  bench/                    # bench_long_context.py, bench_256k_all.sh, charts
  eval/                     # validate_capabilities, probes, eval_and_chart, audits
components/sglang/          # SGLang checkout + applied patches (cloned by setup.sh)
```

## Test System

```
Mac mini (Mac16,11)
Apple M4 Pro — 14-core CPU, 20-core GPU
64 GB unified memory (LPDDR5, ~273 GB/s)
macOS 26.2 (Tahoe)
```

## Long context: how it works

`launch.sh --kv-cache {fp8,turboquant,fp16}` is accepted for CLI compatibility but inert on this backend: the MLX attention pool and the per-request caches hold KV in the attention compute dtype (bf16). Capacity comes from exact pool sizing (`CTX = label + 64`), the MLX buffer-cache cap, patch-015 pre-sizing and small prefill chunks (see Known Issues). [Research](https://github.com/ml-explore/mlx/discussions/3134) suggests 4-bit KV could be *faster* than unquantized on M4 Pro (less memory traffic outweighs the dequant cost); it is a candidate for the backend, not a shipped lever.

**Health check timeout** — SGLang's 20 s default is too short for chunked prefill at depth on Apple Silicon (each 4 K chunk takes 50–80 s); `common.sh` sets `SGLANG_HEALTH_CHECK_TIMEOUT=120`.

**Automatic RoPE scaling** — models with native context < requested get linear interpolation (logged at boot as `RoPE scaling: ... applying linear scale`).

**Radix prefix cache** — agentic workloads re-send the same system prompt + history every turn; the radix cache reuses shared-prefix KV so follow-up turns skip the prefill. Enabled on all presets, including hybrids (DeltaNet/Mamba2 recurrent state is snapshotted per prefix) and Flint (sliding-window KV snapshotted per prefix, patch 021). Greedy output is deterministic on both sides of the cache — repeated hits match, repeated cold runs match — and a hit differs from a cold run of the same prompt only as much as two cold runs on different chunk sizes differ: over 24 positions of a 16.6K-token agent request the top token matches at 20–21 of 24 with a median top-1 shift of 0.02–0.05 nats, every mismatch at a position where the cold top token sits below 0.75 probability ([benchmarks/flint/2026-10-31/prefix-cache/RESULTS.md](benchmarks/flint/2026-10-31/prefix-cache/RESULTS.md)). `EXTRA_ARGS="--disable-radix-cache"` restores the uncached behavior for A/B runs.
