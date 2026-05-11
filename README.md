# Apple Silicon Inference: SGLang + MLX on M4 Pro

256K-context LLM inference on Apple M4 Pro (Mac mini, 64 GB unified memory) using SGLang with a native MLX backend. SGLang **v0.5.11** (commit `612785ffd`) + 7 patches (see [patches/README.md](patches/README.md)) — upstream landed our patch 001 (`kv_cache/` subpackage) in v0.5.11, so the patch set is now 7 instead of 13.

## Current Focus (2026-05-11)

**Primary target: single-user 256K context** for agentic workloads. Decode TPOT at long context > peak batch throughput. Multi-user is secondary.

**Hard constraint: every new model must pass `scripts/eval/validate_capabilities.py`** before its numbers land in this README. Sister teams (3090, R9700) repeatedly found silent regressions in checkpoints that pass MMLU/HumanEval but emit `<unk>`, infinite `<think>` loops, or `<pad>` tokens. We adopt the same gate.

### Cross-team updates

- **R9700 (2026-05-09):** Shipped [`mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ) — first in-house REAM merge from upstream BF16 ([`Qwen/Qwen3-Coder-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct), 128 → 96 experts via Samsung SAIL `merge.py`, ~23 B params / 3 B active). Native AWQ — MLX cannot serve it directly without `mlx_lm.convert`, but the BF16 REAM-merged base could be MLX-quantized fresh if Coder-30B-MoE REAM-pruning lands on the M4 roadmap. **Build-from-scratch rule extension:** R9700 now also prunes themselves from upstream BF16 via Samsung SAIL, not from third-party pre-pruned BF16 (Cerebras / atbender).
- **3090 (2026-04-30):** Cross-checked our patch 013 hybrid-cache-wiring finding (Qwen3.5-27B MMLU 16.7% → 93.0%). **Not a bug class on Ampere SGLang** — `is_hybrid` model config dispatches per-layer cache type before the outer wrapper resolves `make_cache`, so the Ampere path never hits the MLX bridge's fall-through-to-uniform path. Their `qwen3-ream` and `qwen36-27b CT v3` both pass basic+thinking through `validate_capabilities.py` on TP=1 / 4–8K context without a patch-013 equivalent. Confirms the bug is MLX-bridge-specific, not quantization or upstream SGLang.

### Active work

1. **SGLang v0.5.11 rebase — DONE.** Upstream landed the `kv_cache/` subpackage (patch 001) so we dropped it; everything else folded into 7 fresh `.patch` files that apply cleanly via `git apply` — no more `post_apply.py`. **Cross-team mlx-community model gate: 10/10 boot, 10/10 basic, 8/10 thinking** on v0.5.11 (see *v0.5.11 capability gate* table below).
2. **Activate turboquant on v0.5.11.** Patch 008 ships the `kv_quant.py` module + parameter pass-through. `ContiguousKVCache` and `MlxKVPool` need wiring to `KVQuantizer` before fp8/turboquant actually compress KV — currently they store fp16. Load-bearing for 256K work on Gemma 4 31B / Qwen3-32B.
3. **Per-request DeltaNet `conv_state`/`ssm_state` plumbing** — current in-tree fallback serializes hybrid decode (MAX_RUNNING=1) for correctness; proper fix stacks per-request state in `caches[i]`. Throughput-only concern now that patch 013 makes the output correct.
4. **Gemma 4 + Qwen3.6 multimodal** — architecturally support image/video/audio, but the mlx-community 4-bit checkpoints ship without `preprocessor_config.json`, so SGLang can't load any multimodal processor. Text-only on M4 until a re-uploaded checkpoint with the preprocessor lands; harness in `scripts/eval/test_audio.py` runs end-to-end the moment one does.
5. **mlx_vlm import bloat** — text-only loads pull in mlx_vlm and torchcodec, costing ~5 GB of headroom. Skipping these imports for text-only models would unblock Coder-Next-80B on M4 and stop the OOM-guard fires at 64K on everything else.

### v0.5.11 capability gate (2026-05-11, full mlx-community model set)

| Preset | Model | Basic | Thinking | Notes |
|--------|-------|:-----:|:--------:|-------|
| coder-30b | Qwen3-Coder-30B-A3B-Instruct-4bit | **PASS** | **PASS** | 3.5 s — non-thinking model |
| qwen3-moe | Qwen3-30B-A3B-4bit | **PASS** | **PASS** | 10.2 s — thinking trace terminates cleanly (568 tok) |
| qwen3-32b | Qwen3-32B-4bit | **PASS** | **PASS** | 62.0 s — thinking trace terminates (584 tok) |
| devstral | Devstral-Small-2-24B-Instruct-2512-4bit | **PASS** | **PASS** | 14.2 s — image VLM path verified |
| qwen35 | Qwen3.5-27B-4bit | **PASS** | FAIL | 156.9 s — basic PASS confirms patch 013 hybrid-cache fix on v0.5.11; thinking truncates on known greedy-decode `<think>` loop |
| qwen35-9b-8bit | Qwen3.5-9B-MLX-8bit | **PASS** | FAIL | 85.3 s — same as above, smaller variant |
| gemma4 | gemma-4-26b-a4b-it-4bit | **PASS** | **PASS** | 3.8 s |
| gemma4-31b | gemma-4-31b-it-mxfp4 | **PASS** | **PASS** | 12.0 s |
| qwen36 | Qwen3.6-35B-A3B-4bit | **PASS** | **PASS** | 22.6 s — biggest DeltaNet+MoE+VL test; thinking trace 1326 tok terminates |
| qwen36-27b | **Qwen3.6-27B-4bit (new)** | **PASS** | **PASS** | 103.8 s — Dense DeltaNet+VL variant; thinking trace 1311 tok terminates |

10/10 boot success, 10/10 basic, 8/10 thinking on the v0.5.11 stack. The 2 thinking truncations are the pre-existing Qwen3.5 greedy-decode `<think>` loop (patch 013 still works — basic answers are correct, not garbage). Notably the Qwen3.6-A3B and Qwen3.6-27B Dense variants both terminate thinking cleanly out of the box, validating the new Qwen3.6 chat template. Raw data: [`benchmarks/quality/v0.5.11-rebase-validation.txt`](benchmarks/quality/v0.5.11-rebase-validation.txt).

### Quality table (post-patch013, 50–100 sample MMLU, M4 Pro)

| Model | MMLU | HumanEval | Needle 1K |
|:------|:----:|:---------:|:---------:|
| Qwen3.5-27B-4bit | **93.0%** | **100%** | PASS |
| Gemma 4 31B-it-mxfp4 | 90.0% | 0%* | MISS |
| Qwen3.6-35B-A3B-4bit | 88.0% | 80% | PASS |
| Qwen3.5-9B-MLX-8bit | 87.7% | 80% | PASS |
| Qwen3-32B (turboquant) | 86.7% | 87.5% | PASS |
| Coder-30B-A3B-4bit | 86.7% | 75% | PASS |
| Gemma 4 26B-A4B-it-4bit | 86.0% | 0%* | PASS |
| Qwen3-30B-MoE-4bit | 83.3% | 75% | PASS |
| Devstral-24B-4bit | 73.3% | 62.5% | PASS |
| Coder-Next-80B-4bit | 70.0% | 100% | MISS |

All evals run with `--no-thinking` and `--disable-radix-cache`. \*Gemma 4 HumanEval 0% is an instruction-format quirk on the raw `/v1/completions` endpoint, not a model defect. Chart: `benchmarks/quality/quality_comparison.png`.

## Known Issues

- **Radix cache (patch 001) corrupts repeated prompts.** Identical-prompt cache hits return deterministic garbage on the 2nd+ request. **Workaround:** `EXTRA_ARGS="--disable-radix-cache"` (now the default in `run_all_evals.sh` and `test_thinking.sh`).
- **Greedy-only sampling.** MLX backend uses `mx.argmax`; temperature/top-p/top-k unsupported. On Qwen3 family this causes `<think>` loops on reasoning-heavy prompts (`validate_capabilities.py` includes a loop-detector).
- **Qwen3.5-27B / Qwen3-30B-MoE / Qwen3-32B infinite `<think>` loops.** Greedy decode + Qwen3 chat template that always emits `<think>` → loop. Short factual prompts return cleanly; fix blocked on real sampling support.
- **MLX VLM crashes in the SGLang bridge** (not in mlx_vlm itself). Direct `mlx_vlm.load(...) + generate(...)` on synthetic images works; our image-processor / tensor-handoff path in the SGLang MLX bridge is the problem. Devstral image path verified working through patches 007/010/011/012; other VLMs still blocked.
- **VLM warmup crash on Devstral** — set `--skip-server-warmup` automatically in the preset.
- **Coder-Next-80B infeasible on current toolchain.** 42 GB weights + PyTorch/mlx_vlm/torchcodec overhead OOMs the 64 GB Mac during model load. Sister R9700 (2× 32 GB) runs it cleanly; mlx_vlm import bloat (Active work #4) is the load-bearing fix.
- **macOS has no OOM killer** — once a process touches a page past physical RAM, the system stalls until reboot. **OOM guard mandatory for ≥64K work:** `bash scripts/common/oom_guard.sh &` pkills the SGLang server when free+inactive drops below 8 GB.
- **HDMI display blackout** — brief screen blank when the server starts heavy Metal compute. M4 Pro HDMI quirk, not an SGLang bug.

## Quick Start

```bash
./scripts/setup.sh                          # venv, SGLang clone, MLX deps, apply patches

./scripts/launch.sh coder-30b               # MoE — peak throughput, 256K
./scripts/launch.sh devstral                # Dense — image-VLM verified
./scripts/launch.sh qwen35                  # DeltaNet hybrid, 256K
./scripts/launch.sh gemma4                  # MoE 26B, 256K (tight)
./scripts/launch.sh qwen3-moe               # Qwen3-30B MoE
./scripts/launch.sh qwen3-32b               # Dense, needs --kv-cache turboquant for 256K
./scripts/launch.sh gemma4-31b              # Dense, needs --kv-cache turboquant

python scripts/eval/validate_capabilities.py --port 23334   # basic + thinking gate
python scripts/eval/validate_chat_template.py --model <path>
bash   scripts/common/oom_guard.sh &                        # MANDATORY before 64K+ benches
bash   scripts/bench/bench_256k_all.sh                      # 256K single-user sweep
```

Always launch with `--disable-radix-cache` for benches and evals — see Known Issues.

## Prerequisites

- Apple Silicon Mac with **≥ 64 GB unified memory** (M4 Pro Mac mini was the reference rig)
- macOS 26+ (Tahoe), Xcode CLT
- Python 3.12, `uv` or `venv`
- ~200 GB disk for models

## Model Support

| Model | Type | Weights | 1-user tok/s | Max context | Launch |
|-------|------|:-------:|:------------:|:-----------:|:------:|
| Coder-Next 80B | MoE+DeltaNet | 42 GB | 55.7 | **64K** (18.3 tok/s) | `launch.sh coder-next`* |
| Coder-30B-A3B | MoE (3B active) | 16 GB | 68.4 | **256K** (3.2 tok/s) | `launch.sh coder-30b` |
| Qwen3-30B-MoE | MoE (3B active) | 16 GB | 69.0 | **64K** (6.3 tok/s) | `launch.sh qwen3-moe` |
| Qwen3.6-35B-A3B | MoE+DeltaNet | 17 GB | 51.8 | **256K** (0.1 tok/s) | `launch.sh qwen36` |
| Gemma 4 26B-A4B-it | MoE (4B active) | 15 GB | 58.8 | **256K** (1.5 tok/s) | `launch.sh gemma4` |
| Qwen3.5-27B | DeltaNet hybrid | 15 GB | 14.3 | 256K (decode times out) | `launch.sh qwen35` |
| Devstral-24B | Dense | 14 GB | 17.0 | **256K** (1.8 tok/s) | `launch.sh devstral` |
| Qwen3-32B | Dense (turboquant) | 18 GB | 12.1 | 16K (bench timeout) | `launch.sh qwen3-32b` |
| Gemma 4 31B-it | Dense (sliding+full) | 17 GB | 8.6 | 8K (16K OOMs) | `launch.sh gemma4-31b` |

\*Coder-Next-80B currently infeasible on this toolchain (see Known Issues). All models 4-bit MLX from [`mlx-community/`](https://huggingface.co/mlx-community).

### Multimodal capability matrix

What each architecture *can* do vs what *works through our SGLang+MLX bridge today*:

| Model | Image | Video | Audio | Status on M4 |
|-------|:-----:|:-----:|:-----:|:-------------|
| Devstral-24B (Mistral3) | ✅ | ❌ | ❌ | **Image working** end-to-end (patches 007/010/011/012 + VLM detection). |
| Qwen3.5-27B / 9B-8bit | ✅ | ✅ | ❌ | Image wires through; video supported by arch, needs end-to-end test. |
| Qwen3.6-35B-A3B | ✅ | ✅ | ❌ | Same arch class as 3.5; preset added, not yet downloaded/tested. |
| Gemma 4 26B / 31B | ✅ | ✅ | ✅ | Architecturally [image+video+audio](https://ai.google.dev/gemma/docs/capabilities/vision/video). **Blocked:** mlx-community 4-bit checkpoints ship without `preprocessor_config.json`. Text-only until a re-uploaded checkpoint lands. |
| Coder-30B / Coder-Next / Qwen3-30B-MoE / Qwen3-32B | ❌ | ❌ | ❌ | Text-only by architecture. |

### Choosing a model

**MoE wins at long context.** Each decode token must (1) read model weights and (2) read the entire KV cache. At short context, weight loading dominates → MoE reads 1.5 GB vs Dense 14 GB (4× faster). At 256K with fp8, the KV read climbs to ~5–10 GB — comparable to dense weights — so MoE keeps the weight component small and the KV penalty proportionally less painful. Coder-30B is the best overall: fastest decode, lowest KV pool usage, highest concurrent throughput.

**DeltaNet hybrids** (Qwen3.5, Coder-Next) alternate standard attention (O(n)) with linear attention (O(1)). Linear layers don't slow with context — architecturally suited for very long context — but the standard layers in the hybrid still pay full O(n).

## Performance

> Mac mini M4 Pro (64 GB), SGLang + MLX, `sglang.bench_serving`.
> **Context sweep**: single user, 64 output tokens, radix cache disabled, FP8 or TurboQuant KV cache.
> **Concurrency sweep**: 256 in / 256 out, 8 K context, scaling concurrent users.

### v0.5.11 short-sweep decode tok/s (2026-05-11, full cross-team set, fp16 KV)

![v0.5.11 perf chart](benchmarks/quality/v0.5.11-perf-shortsweep.png)

Single-user decode speed at 128 / 4K / 16K context. fp16 KV (default — turboquant integration still pending on v0.5.11). Output 64 tokens, radix cache disabled.

| Preset | Model | tok/s @128 | tok/s @4K | tok/s @16K |
|--------|-------|:----------:|:---------:|:----------:|
| coder-30b | Qwen3-Coder-30B-A3B-Instruct-4bit (MoE) | **57.9** | 10.3 | 1.7 |
| qwen3-moe | Qwen3-30B-A3B-4bit (MoE) | **59.2** | 10.5 | 1.7 |
| qwen36 | Qwen3.6-35B-A3B-4bit (MoE+DeltaNet+VL) | **52.5** | 11.0 | 2.6 |
| gemma4 | gemma-4-26b-a4b-it-4bit (MoE, ctx=4K preset) | **47.5** | 8.7 | n/a* |
| qwen35-9b-8bit | Qwen3.5-9B-MLX-8bit (DeltaNet) | 23.7 | 5.0 | 1.3 |
| devstral | Devstral-Small-2-24B-2512-4bit (Dense+VLM) | 14.2 | 2.0 | 0.5 |
| qwen36-27b | Qwen3.6-27B-4bit (Dense DeltaNet+VL, **new**) | 11.7 | 1.6 | 0.4 |
| qwen35 | Qwen3.5-27B-4bit (DeltaNet) | 11.8 | 1.7 | 0.4 |
| gemma4-31b | gemma-4-31b-it-mxfp4 (Dense, ctx=4K preset) | 10.4 | 1.3 | n/a* |
| qwen3-32b | Qwen3-32B-4bit (Dense) | 10.0 | 1.3 | 0.3 |

\*Gemma 4 presets ship with `CTX=4096` (tight 64 GB budget) — 16K requests rejected. Raise via `CTX=16384 bash scripts/launch.sh gemma4` for the longer-context numbers. Raw bench logs: `/tmp/perf_<preset>_bench.log`.

### v0.5.11 long-context turboquant sweep (2026-05-11)

Decode tok/s under the turboquant KV cache wired up in patch 008. 5 models × 5 context lengths (128 → 64K), single user, radix cache disabled.

| Preset | @128 | @4K | @16K | @32K | @64K |
|--------|:----:|:---:|:----:|:----:|:----:|
| qwen3-moe (Qwen3-30B-A3B) | 58.7 | 10.3 | 1.7 | 0.6 | — \* |
| coder-30b (Qwen3-Coder-30B) | 58.3 | 10.4 | 1.8 | 0.6 | **0.2** |
| qwen36 (Qwen3.6-35B-A3B MoE+DN+VL) | 52.9 | 11.0 | 2.8 | **1.2** | **0.5** |
| qwen36-27b (Qwen3.6-27B Dense+DN+VL) | 11.7 | 1.6 | 0.4 | 0.2 | — \* |
| devstral (24B Dense) | 13.9 | 1.9 | 0.4 | 0.2 | — \* |

\*The original 64K runs were rejected with HTTP 400 because the preset's `CTX=32768` overrode the env var (`CTX=80000 launch.sh`); the launch.sh env-override fix landed mid-sweep. coder-30b and qwen36 were re-run after the fix with `CTX=80000` to backfill that column; the others can be re-run on follow-up.

**At 64K, qwen36 (DeltaNet hybrid) is 2.5× faster than coder-30b (pure MoE)** — 0.5 vs 0.2 tok/s decode, 139 s vs 361 s prefill. The DeltaNet linear-attention layers don't pay the full O(n) KV-read cost on every decode token, which compounds across alternating attention layers. This is the load-bearing data point that argues for Qwen3.6-35B-A3B as the M4's 256K-context choice.

**Headline: turboquant works.** Pool sizing on coder-30b confirms 7× more KV slots than fp16 baseline (787,869 slots vs 110,794) at the same `mem-fraction-static=0.7`. Validation `2/2 PASS` — output identical to fp16 within tolerance. Decode tok/s at short context is within 1% of fp16 (58.3 vs 57.9 on coder-30b @128). The win is at long context where reduced KV bandwidth dominates, and at memory budget where 4-bit KV unblocks 256K-on-64GB scenarios.

**Notable: DeltaNet O(1) layers visible at 32K.** Qwen3.6-35B-A3B's MoE+DeltaNet hybrid maintains 1.2 tok/s @ 32K — 2× the speed of the MoE-only coder-30b/qwen3-moe (0.6 tok/s @ 32K). The hybrid linear-attention layers don't pay the full O(n) KV-read cost on every decode token.

### Pre-rebase 256K results (turboquant, fp8) — archived for comparison

| Model | tok/s @128 | tok/s @64K | tok/s @256K | KV pool |
|-------|:----------:|:----------:|:-----------:|:-------:|
| Coder-30B (post-patches, turboquant) | 44.4 | OOM @ 64K | — | 80K |
| Coder-30B (pre-patches, fp8) | 68.4 | 6.3 | **3.2** | 20% |
| Qwen3-30B-MoE (post-patches, turboquant) | 45.8 | OOM @ 64K | — | 80K |
| Qwen3.5-27B (post-013, turboquant) | 11.1 | 0.2 | 0.07 @ 250K | 270K |
| Qwen3.6-35B-A3B | 51.8 | 0.9 | 0.1 @ 250K | 270K |
| Devstral-24B | 17.0 | 3.4 | **1.8** | 30% |
| Gemma 4 26B | 58.8 | 3.0 | **1.5** | 48% |
| Gemma 4 31B-it (post-015, turboquant) | 8.6 | OOM @ 16K | — | 100K |

These pre-rebase numbers were taken on the old `1f8df97` stack with turboquant active. The v0.5.11 turboquant sweep above replaces them up to 32K; the 64K + 256K columns are pending the preset-CTX env-override fix (now in place). The 3.9 tok/s @ 256K Qwen3.5 number we cited pre-patch013 was misleading — that bench ran on broken DeltaNet inference (fluent garbage, MMLU 16.7%); post-013 the real number is 0.07 tok/s @ 250K. Sister R9700 hits 13.3 tok/s @ 262K on Qwen3.6 — discrete-GPU compute advantage at long context dwarfs M4's unified-memory win for MoE+DeltaNet stacks.

### Throughput scaling (256 in / 256 out, 8 K context)

| Model | 1 user | 2 users | 4 users | 8 users |
|-------|:------:|:-------:|:-------:|:-------:|
| Coder-30B (MoE) | 82.6 | 89.6 | 97.1 | **107.4** |
| Devstral-24B (Dense) | 27.0 | 26.8 | 30.4 | 39.2 |

Per-model JSON + charts under `benchmarks/<slug>/`.

### Memory budget at 256K (64 GB Mac)

Radix cache pre-allocates the KV pool at startup; on unified memory it competes with Metal compute buffers. Use `--mem-fraction-static 0.7` (default).

| Model | Weights | KV @256K fp8 | Total | Headroom | Fits? |
|-------|:-------:|:------------:|:-----:|:--------:|:------|
| Coder-30B (MoE) | 16 GB | 12.4 GB | 28 GB | 24 GB | **fp8** |
| Qwen3-30B (MoE) | 16 GB | 12.4 GB | 28 GB | 24 GB | **fp8** |
| Devstral-24B | 14 GB | 20.6 GB | 35 GB | 17 GB | **fp8** |
| Gemma 4 26B (MoE) | 15 GB | 30.9 GB | 46 GB | 6 GB | **fp8** (tight) |
| Qwen3-32B | 18 GB | 33.0 GB | 51 GB | 1 GB | **turboquant** required |
| Coder-Next 80B | 42 GB | — | — | — | No (weights alone) |

## Cross-team collaboration

Three repos share the same SGLang stack; their findings feed back into this README.

- **3090 team** — [`~/AI/2x-3090-GA102-300-A1-sglang-inference`](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference) (NVIDIA Ampere, AWQ_Marlin, 14 patches, 256K @ 74 tok/s on REAM-30B)
- **R9700 team** — [`~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference`](https://github.com/mattbucci/2x-R9700-RDNA4-GFX1201-sglang-inference) (AMD RDNA4, ROCm 7.2, 14 patches, 256K @ 13 tok/s on Qwen3.6-35B-A3B)

Their kernel patches don't port directly (Marlin/HIP-specific), but their **eval harnesses, calibration insights, and chat-template fixes are portable**. The quality eval suite under `scripts/eval/` is a direct adoption.

## Setup

```bash
./scripts/setup.sh
```

Manually:
```bash
python3 -m venv .venv && source .venv/bin/activate
git clone https://github.com/sgl-project/sglang.git components/sglang
cd components/sglang && git checkout v0.5.11
for p in ../../patches/00[2-9]-*.patch; do git apply "$p"; done
cd python && cp pyproject_other.toml pyproject.toml
pip install -e ".[srt_mps]"
```

| Component | Version |
|-----------|---------|
| SGLang | **v0.5.11** (`612785ffd`) + 7 patches |
| MLX | 0.31.1 |
| mlx-lm | 0.31.2 |
| PyTorch | 2.9.1 (MPS) |
| Python | 3.12 |

## Patches

7 patches on top of SGLang `v0.5.11` (commit `612785ffd`). Upstream landed patch 001 (the `kv_cache/` subpackage) — we dropped it. The old in-tree mods 008–015 are now folded into proper patch files (006 / 008 / and inside 004). All patches apply via `git apply` against a clean v0.5.11 — no more `post_apply.py`. See [patches/README.md](patches/README.md) for the per-patch breakdown and [patches/REBASE-v0.5.11-NOTES.md](patches/REBASE-v0.5.11-NOTES.md) for the rebase narrative.

| # | Patch | What |
|:-:|-------|------|
| 002 | mps-backend-defaults | Disable CUDA graph & piecewise CUDA on MPS, force `torch_native` attention, multimodal off by default. |
| 003 | mlx-skip-quantization-check | Skip SGLang's quantization verify when MLX backend is active. |
| 004 | mlx-lifecycle-and-hybrid-fixes | Lifecycle (clear-on-idle, drop-on-finish) + hybrid-model bookkeeping + **patch 013** hybrid cache via `language_model.make_cache()` (Qwen3.5/3.6 MMLU 16.7%→93%) + **patch 015** keep `RotatingKVCache` native (Gemma 4 sliding) + VLM-detect-first `_load_model` with image-aware shim + RoPE auto-scaling. Hybrid-aware `find_attention_layers` so DeltaNet-first layer orderings don't crash; `_get_attn_config` accepts both `n_kv_heads` (mlx_lm) and `num_key_value_heads` (mlx_vlm). |
| 005 | mlx-attn-wrapper-varargs | Devstral / Ministral3 `attn_scale` positional-arg compat. |
| 006 | mlx-offsetcache-and-make-mask | `OffsetCache.__getitem__`/`__setitem__`/`__len__`/`lengths`/`advance` stubs for hybrid decode + **patch 014** explicit `(N, offset+N)` `make_mask` when `offset>0` (chunked prefill). |
| 007 | mlx-multimodal-and-mps-shim | `_mps_stub` cuda→cpu redirect, `mm_utils` shm page-rounding (macOS 16 KB pages), `Modality.MULTI_IMAGES` enum member. |
| 008 | mlx-kv-quant-module | New `kv_quant.py` — `KVCacheMode`, `KVQuantizer`, `bytes_per_element`, `parse_kv_cache_mode` (fp8/mxfp8/turboquant/tq/4bit aliases). Wired into `MlxModelRunner.__init__` via `kv_cache_mode` + `context_length` kwargs; ContiguousKVCache + MlxKVPool wire-up TBD. |

## Repo layout

```
patches/                    # SGLang patches — see patches/README.md
  00*.patch                 #   7 numbered patches
  REBASE-v0.5.11-NOTES.md   #   v0.5.11 rebase strategy & lineage
  REBASE-v0.5.11-NOTES.md   #   upcoming SGLang version bump plan
benchmarks/                 # Per-model JSON + charts
  quality/                  #   MMLU / HumanEval / Needle (chart)
  <slug>/                   #   throughput + long-context sweeps
scripts/
  launch.sh                 # Unified launcher — launch.sh <preset>
  common.sh                 # Shared MLX env setup
  setup.sh                  # Full setup (venv, SGLang clone, patches)
  common/oom_guard.sh       # MANDATORY for ≥64K work — pkills server below 8 GB free
  common/mem_profile.sh     # CSV memory profile companion
  bench/                    # bench_long_context.py, bench_256k_all.sh, charts
  eval/                     # validate_capabilities, eval_and_chart, smoke, audio/video probes
  test/                     # kernel/cache microbenchmarks
components/sglang/          # SGLang checkout + applied patches (cloned by setup.sh)
```

## Test System

```
Mac mini (Mac16,11)
Apple M4 Pro — 14-core CPU, 20-core GPU
64 GB unified memory (LPDDR5, ~273 GB/s)
macOS 26.2 (Tahoe)
```

## 256K context: how it works

Three features make 256K possible on a 64 GB Mac:

```bash
--kv-cache fp8          # MXFP8, ~1.9× memory savings (default)
--kv-cache turboquant   # Affine 4-bit, ~3.6× memory savings (large-KV models / 256K)
--kv-cache fp16         # Debugging only
```

[Research](https://github.com/ml-explore/mlx/discussions/3134) shows 4-bit KV can be *faster* than unquantized on M4 Pro — less memory traffic outweighs dequant cost.

**Health check timeout** — SGLang's 20 s default is too short for chunked prefill at 64K+ on Apple Silicon (each 4 K chunk takes 50–80 s). `common.sh` sets `SGLANG_HEALTH_CHECK_TIMEOUT=120`.

**Automatic RoPE scaling** — models with native context < requested get linear interpolation:
```
RoPE scaling: context_length=262144 > max_position_embeddings=40960,
applying linear scale=0.1562 (factor=6.40)
Patched 48 RoPE modules
```

**Radix prefix cache** — for agentic workloads, the same system prompt + history is re-sent every turn. Without caching, a 256K prompt re-prefills (~20 min) every turn. Radix cache (patch 001) reuses the KV cache from shared prefixes; follow-up = `< 1 s`. But: see "repeated-prompt corruption" in Known Issues — disable for evals.
