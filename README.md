# Apple Silicon Inference: SGLang + MLX on M4 Pro

Long-context LLM inference on Apple M4 Pro (Mac mini, 64 GB unified memory) using SGLang with a native MLX backend. Stack: **SGLang v0.5.15.post1** (commit `0b3bb0c`) + 6 patches ([patches/README.md](patches/README.md)). Text and VLM/hybrid paths are validated: `qwen36` is the primary agentic model (codegen STRONG, vision STRONG, video STRONG, thinking VERIFIED); `coder-30b`/`qwen3-moe`/`qwen3-32b`, `qwen35`, `devstral`, and `nemotron-30b` all pass their gates. Hybrid (DeltaNet/Mamba2) presets run with the radix cache (greedy-determinism-validated prefix caching; trade-off: overlap schedule off for hybrids). `gemma4*` is blocked by an upstream sliding-window gap.

**Long-context: 128K single-user context validated on qwen36** (the buffer-cache root cause of the prior ~32K ceiling is fixed in patch 008 — receipts in `benchmarks/longctx-bisect/`). Throughput benchmark tables below were measured on earlier stack pins; the re-measure is queued.

## Action queue

Full specs and statuses live in [experiments/README.md](experiments/README.md); one-line summary, in execution order:

- [ ] **Disk triage** — 12 GiB free of 926 GiB; the HF cache holds 362 GB. Inventory, propose deletions to Matt. Gates the in-house quant and bisect arms. *(~1h)*
- [ ] **Bench-flag pin** — `--random-range-ratio 1` in every `sglang.bench_serving` invocation + flag legacy depth-labeled rows as suspect. Spec: [experiments/01](experiments/01-pin-random-range-ratio-flag-legacy-rows.md). *(hours)*
- [ ] **Patch replay gates in setup.sh** — script the pristine-replay + byte-identity + double-apply gates and wire the probe suite in as a post-setup gate. Spec: [experiments/03](experiments/03-patch-replay-gates-setup.md). *(hours)*
- [ ] **Radix/overlap decode A/B** — quantify the overlap-schedule cost of radix-on-hybrid at MR=1; keep preset defaults on data. *(~2h)*
- [ ] **Beyond 128K** — survive the contiguous-attention cache's 131K doubling boundary (incremental growth or pool-backed prefill writes), and attack decode TPOT at depth. *(the long-context growth regression itself is resolved — patch 008 cache cap; receipts in `benchmarks/longctx-bisect/`)*
- [ ] **Real sampling** — wire temp/top-p/top-k/min-p into the MLX backend via `mlx_lm make_sampler`; unblocks thinking-mode evals. Spec: [experiments/05](experiments/05-mlx-sampling.md). *(days)*
- [ ] **Tool-call boot gate** — port the 3090's `check_tool_call` into `validate_capabilities.py`. Spec: [experiments/06](experiments/06-check-tool-call-gate.md). *(hours)*
- [ ] **Re-measure on the current stack** — throughput/long-context tables, the SWE-bench cell, the unswept presets (`qwen35-9b-8bit`, `qwen36-27b`). *(hours per piece)*
- [ ] **Re-arm baselines.json (schema v2, 3090 fleet standard 2026-07-19)** — current file is a 2026-04-12 relic in the old parser format; rides the re-measure evening. M4 ceiling depths 1024/8192/32768, `--random-range-ratio 1` pinned (M4-C already landed the pin), `oom_guard.sh` mandatory, keyed by launch-preset name, `invalid`/`depth_shortfall` points never saved. Schema + rules: 3090 `scripts/bench/README.md` (their tripwire armed 7/7 within 3% of receipts, validated exit-1 both ways).
- [ ] **In-house qwen36 MLX 4-bit** with router/DeltaNet/vision exclusions — after disk + sampling. Spec: [experiments/07](experiments/07-qwen36-inhouse-mlx4bit-exclusions.md). *(days)*

## Primary target: long-context agentic coding

Single-user agentic coding at long context (tool-call-heavy multi-turn sessions, 10K–100K-token codebase prefixes) is what this stack is tuned for. Decode TPOT at long context matters more than peak batch throughput; the right model is one whose decode stays flat as context grows.

### Recommended picks (ranked by SWE-bench Lite agentic-coding ability)

**`qwen36` is the only configuration verified to produce real SWE-bench Lite patches on M4, reproducibly, at scale, across all 12 ecosystems in our test set.** Measured on the previous stack pin (re-run queued):

- **Patch-engagement: 21/26 (80.8%) across 12 ecosystems** — model produces a non-empty patch targeting the canonical issue file.
- **Resolved (official Docker harness, scored on the 3090 rig): 9/26 = 34.6%** (9/21 = 42.9% of non-empty; 0 errors — all 21 non-empty ran, incl. the 8 old-Python instances M4 can't build). All 5 M4-local resolves re-resolved (no flips); 4 additional resolves came from the M4-untestable class. Receipt: [`exports/qwen36-docker-summary.json`](evals/swebench/exports/qwen36-docker-summary.json) + [exports/README.md](evals/swebench/exports/README.md) Results.

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

### Probe matrix (current stack, radix on)

| Preset | codegen | vision | video | thinking |
|--------|:-------:|:------:|:-----:|:--------:|
| `qwen36` | **STRONG** | **STRONG** | **STRONG** | **VERIFIED** |
| `qwen35` | **STRONG** | **STRONG** | PARTIAL | skipped (known greedy loop) |
| `devstral` | **STRONG** | **STRONG** | PARTIAL | n/a |
| `nemotron-30b` | **STRONG** | n/a | n/a | **VERIFIED** |
| `coder-30b` / `qwen3-moe` / `qwen3-32b` | **STRONG** | n/a | n/a | pass validate gate |
| `qwen35-9b-8bit` / `qwen36-27b` | unswept (same code path as siblings) | | | |
| `gemma4` / `gemma4-31b` | blocked at boot (upstream sliding-window gap) | | | |

Per-preset JSON: [`benchmarks/quality/probe-trio/`](benchmarks/quality/probe-trio/).

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

**DWQ variants are used where they measured better** — `coder-30b` (+2.8 MMLU / +20 HE over the broken 4bit), `qwen3-moe` (+7.9 MMLU / −5 HE), `qwen3-32b` (+2.8 / +7.5). `qwen36` stays on standard 4bit (its DWQ trades −5.5 MMLU for +15 HE; `MODEL="mlx-community/Qwen3.6-35B-A3B-4bit-DWQ" launch.sh qwen36` overrides when code-specialist behavior is wanted). **Never blind-swap DWQ — recipes vary per upload; measure MMLU + HumanEval per model.**

## Known Issues

- **Greedy-only sampling.** The MLX backend selects tokens with `mx.argmax`; temperature/top-p/top-k are silently ignored. On Qwen3.5-27B this causes infinite `<think>` loops on reasoning-heavy prompts (`validate_capabilities.py` includes a loop detector). Real sampling is queued ([experiments/05](experiments/05-mlx-sampling.md)).
- **`gemma4` / `gemma4-31b` blocked at boot** — upstream `_attention_kv_config_for_layer` raises `NotImplementedError` for sliding-window attention at runner construction. Upstream feature gap; needs window-aware pools.
- **Long-context ceiling is 128K** (validated: in=125,830, ~7 min prefill, decode 0.1 tok/s). 192K+ dies at the contiguous-attention cache's 131K capacity-doubling boundary — needs incremental cache growth or pool-backed prefill writes (queued). Decode TPOT at depth (13 s/token at 128K) is the other open constraint. The MLX buffer cache is capped by patch 008 (`SGLANG_MLX_CACHE_LIMIT_GB`, default 4 GB) — do not remove the cap; uncapped, chunked prefill retains ~0.6 MB/token and jetsams around 30K.
- **64K dense-attention ceiling is structural.** The per-chunk attention-score tensor `mx.fast.scaled_dot_product_attention` materializes hits 30–90 GB at deep offsets. Without flash-attention-style block-streaming SDPA in MLX, decode at 32K is the practical ceiling for dense standard-attention models; DeltaNet hybrids (`qwen35`, `qwen36`, `qwen36-27b`) are the workaround.
- **Radix-on-hybrid disables the overlap schedule** (upstream `no_buffer` constraint) — hybrids run the normal event loop. Prefix-cache wins dominate for agentic multi-turn work; decode A/B is queued.
- **Coder-Next-80B infeasible** — 42 GB weights alone exceed the safe budget; model load OOMs. No path on a single 64 GB Mac.
- **macOS has no OOM killer** — a process touching a page past physical RAM stalls the whole system until reboot. **OOM guard mandatory for ≥32K work:** `bash scripts/common/oom_guard.sh &` pkills the server when free+inactive drops below 8 GB.
- **macOS jetsam can silently reap the scheduler mid-eval** — no traceback, just `Connection refused` on subsequent requests. `needle_eval` tags `server_dead=True` on connection-class errors; `mmlu_eval`/`humaneval_eval`/`labbench_eval` don't yet. When an eval section drops to ~0%, suspect jetsam before model regression: relaunch fresh and re-run that section.
- **`--mem-fraction-static` is a fraction of TOTAL system RAM on unified memory**, not "GPU memory". 0.85 has hard-locked the box (compressor + swap ~150 GB effective). The 0.7 default ceiling is load-bearing; long-context presets go *down* to 0.4.
- **HDMI display blackout** — brief screen blank when heavy Metal compute starts. M4 Pro quirk, not an SGLang bug.

## Quick Start

```bash
./scripts/setup.sh                          # venv, SGLang v0.5.15.post1, MLX deps, 6 patches

# Validated presets
./scripts/launch.sh qwen36                  # PRIMARY — MoE+DeltaNet+VL, full probe matrix green
./scripts/launch.sh coder-30b               # MoE — fastest decode, direct chat-completion codegen
./scripts/launch.sh qwen3-moe               # Qwen3-30B MoE (DWQ, MMLU 91)
./scripts/launch.sh qwen3-32b               # Dense (DWQ, clean audit)
./scripts/launch.sh qwen35                  # DeltaNet hybrid+VL
./scripts/launch.sh devstral                # Dense+VLM (Mistral3)
./scripts/launch.sh nemotron-30b            # NemotronH (Mamba2+Attn+MoE)

# Same code path as validated siblings, not yet swept:
./scripts/launch.sh qwen35-9b-8bit          # 10 GB resident tight-memory variant
./scripts/launch.sh qwen36-27b              # Dense DeltaNet+VL

# Blocked / not in rotation:
# gemma4, gemma4-31b — upstream sliding-window gap (boot failure)
# coder-next          — infeasible on 64 GB
# smol-docling        — 256M VLM smoke test only

# Long-context (128K validated; OOM guard mandatory for ≥32K)
bash scripts/common/oom_guard.sh &
CTX=140000 MEM_FRAC=0.5 EXTRA_ARGS="--disable-radix-cache" \
    bash scripts/launch.sh qwen36 --kv-cache turboquant

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
| `devstral` | `Devstral-Small-2-24B-Instruct-2512-4bit` | Dense+VL (Mistral3) | 14 GB | 17.0 (40 MR=4) | 256K (1.8) | **clean** |
| `qwen3-32b` | `Qwen3-32B-4bit-DWQ` | Dense | 18 GB | 12.1 | 16K | **clean** |
| `gemma4` | `gemma-4-26b-a4b-it-4bit` | MoE (4B active) | 15 GB | 58.8 | blocked | `embed_vision.embedding_projection` INT4 |
| `gemma4-31b` | `gemma-4-31b-it-mxfp4` | Dense (sliding+full) | 17 GB | 11.7 | blocked | `embed_vision.embedding_projection` INT4 |
| `nemotron-30b` | `NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` | NemotronH (Mamba2+Attn+MoE) | 17 GB | — | 32K probe | **clean** |

\* tok/s and Max-ctx cells were measured on earlier stack pins (re-measure queued) and pre-date the bench depth pin — depth-labeled cells are suspect ([inventory](benchmarks/LEGACY-DEPTH-SUSPECT.md)). The validated long-context ceiling is 128K (see Known Issues). MR=N numbers are batched-decode peaks. Audit hazards from the [checkpoint audits](#checkpoint-audits).

## Performance

> Mac mini M4 Pro (64 GB), SGLang + MLX, `sglang.benchmark.serving`.
> **All numbers in this section were measured on earlier stack pins** — the current-stack re-measure is in the action queue. Depth-labeled rows additionally pre-date the `--random-range-ratio 1` pin (labeled depths drew uniform [1,N]) and are flagged suspect — mechanism + full inventory in [benchmarks/LEGACY-DEPTH-SUSPECT.md](benchmarks/LEGACY-DEPTH-SUSPECT.md).

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

**Turboquant works:** ~7× more KV slots than fp16 at the same mem-fraction, outputs identical within tolerance, decode within 1% of fp16 at short context. The win is at long context (reduced KV bandwidth) and memory budget.

### Batched-decode peaks (single-server multi-prompt)

| Preset | Single user | Peak @ MR |
|--------|:-----------:|:---------:|
| `qwen3-moe` | 69 | **160 @ MR=8** |
| `qwen36` | 52 | **148 @ MR=2** |
| `devstral` | 17 | **40 @ MR=4** |
| `qwen35` / `qwen36-27b` | 12–14 | **34 @ MR=2** |

### Memory budget at 256K (64 GB Mac)

| Model class | Weights | KV @256K fp8 | Fits? |
|-------|:-------:|:------------:|:------|
| MoE 3B-active (Coder-30B, Qwen3-30B-DWQ) | 16 GB | 12 GB | **fp8** comfortably |
| MoE+DeltaNet (Qwen3.6-35B-A3B) | 17 GB | varies† | **turboquant** |
| Dense+VL (Devstral, Qwen3.5-27B) | 14–15 GB | 21 GB | **fp8** |
| Dense (Qwen3-32B) | 18 GB | 33 GB | **turboquant** required |

† DeltaNet layers hold recurrent state instead of KV; only full-attention layers store KV.

## Setup

```bash
./scripts/setup.sh
```

Manually:
```bash
python3 -m venv .venv && source .venv/bin/activate
git clone --depth 1 --branch v0.5.15.post1 https://github.com/sgl-project/sglang.git components/sglang
cd components/sglang
for p in ../../patches/0[01][0-9]-*.patch; do git apply "$p"; done
cd python && cp pyproject_other.toml pyproject.toml
pip install -e ".[srt_mps]"
pip install "mlx-vlm==0.6.5" --no-deps    # VLM loader; SGLang pins transformers==5.12.1, hence --no-deps
```

| Component | Version |
|-----------|---------|
| SGLang | **v0.5.15.post1** (`0b3bb0c`) + 6 patches |
| MLX | 0.32.0 |
| mlx-lm | 0.31.3 |
| mlx-vlm | 0.6.5 |
| PyTorch | 2.11.0 (MPS) |
| transformers | 5.12.1 |
| Python | 3.12 |

## Patches

Six patches on top of `v0.5.15.post1` — full rationale per patch in [patches/README.md](patches/README.md):

| # | Patch | What |
|:-:|-------|------|
| 002 | mps-backend-defaults | MPS multimodal default + MLX `--kv-cache-dtype` aliases (boot-critical). |
| 003 | mlx-skip-quantization-check | Accept MLX checkpoints' `quantization_config` shape. |
| 005 | mlx-attn-wrapper-varargs | Attention-wrapper vararg + attribute delegation to the wrapped module. |
| 007 | mlx-multimodal-and-mps-shim | cuda→cpu redirect, shm page-rounding, `MULTI_IMAGES` modality. |
| 008 | mlx-vlm-hybrid-integration | The VLM/hybrid path: mlx_vlm loader, attention detection, mamba-allocator contract, radix-for-hybrids, vision plumbing, NemotronH. |
| 014 | mlx-hf-processor-fixes | Gemma 4 image-only processor; Mistral3/Devstral processor + tokenizer resolution. |

## Repo layout

```
patches/                    # 6 numbered patches — see patches/README.md
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

```bash
--kv-cache fp8          # MXFP8, ~1.9× memory savings (default)
--kv-cache turboquant   # Affine 4-bit, ~3.6× savings (large-KV models / deep context)
--kv-cache fp16         # Debugging only
```

[Research](https://github.com/ml-explore/mlx/discussions/3134) shows 4-bit KV can be *faster* than unquantized on M4 Pro — less memory traffic outweighs dequant cost.

**Health check timeout** — SGLang's 20 s default is too short for chunked prefill at depth on Apple Silicon (each 4 K chunk takes 50–80 s); `common.sh` sets `SGLANG_HEALTH_CHECK_TIMEOUT=120`.

**Automatic RoPE scaling** — models with native context < requested get linear interpolation (logged at boot as `RoPE scaling: ... applying linear scale`).

**Radix prefix cache** — agentic workloads re-send the same system prompt + history every turn; the radix cache reuses shared-prefix KV so follow-up turns skip the prefill. Enabled on all presets, including hybrids (DeltaNet/Mamba2 recurrent state is snapshotted per prefix — greedy outputs are token-identical on cache hits). `EXTRA_ARGS="--disable-radix-cache"` restores the uncached behavior for A/B runs.
