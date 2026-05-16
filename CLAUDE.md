# M4 Pro Inference Project

SGLang with native MLX backend on Apple M4 Pro (64GB unified memory).

**All inference MUST use SGLang with the MLX backend.** Set `SGLANG_USE_MLX=1` for all operations.

## Documentation

| File | Purpose |
|------|---------|
| [README.md](README.md) | Setup, benchmarks, model support, known issues |
| [rules-for-agents.md](rules-for-agents.md) | Apple Silicon constraints, launch rules, MLX specifics |
| [patches/README.md](patches/README.md) | Per-patch notes (5 patches on top of SGLang main) |

## Key Commands
```bash
scripts/setup.sh                                  # venv, SGLang v0.5.11, MLX deps, apply 13 patches
# Production presets (all verified through the v0.5.11 capability gate):
scripts/launch.sh devstral                        # Devstral 24B + image VLM
scripts/launch.sh coder-30b                       # Qwen3-Coder-30B-A3B-DWQ MoE
scripts/launch.sh gemma4                          # Gemma 4 26B MoE (text-only on M4)
scripts/launch.sh gemma4-31b                      # Gemma 4 31B Dense
scripts/launch.sh qwen35                          # Qwen3.5-27B DeltaNet hybrid+VL
scripts/launch.sh qwen35-9b-8bit                  # Qwen3.5-9B 8-bit (tight-memory variant)
scripts/launch.sh qwen3-32b                       # Qwen3-32B-DWQ Dense
scripts/launch.sh qwen3-moe                       # Qwen3-30B-A3B-DWQ MoE
scripts/launch.sh qwen36                          # Qwen3.6-35B-A3B MoE+DeltaNet+VL
scripts/launch.sh qwen36-27b                      # Qwen3.6-27B Dense+DeltaNet+VL
scripts/launch.sh nemotron-30b                    # NemotronH (Mamba2+Attn+MoE)
# coder-next is infeasible on 64 GB; smol-docling is a VLM smoke test only.

# Capability gates (run AFTER server is up on PORT 23334)
python scripts/eval/validate_capabilities.py --port 23334    # basic + thinking gate (loose grep)
python scripts/eval/probe_thinking.py        --port 23334    # content-aware reasoning probe
python scripts/eval/probe_vision.py          --port 23334    # content-aware image probe
python scripts/eval/probe_codegen.py         --port 23334    # 2-task / 8-test codegen probe
bash   scripts/eval/probe_all.sh                              # full sweep across presets
PRESETS="nemotron-30b" bash scripts/eval/probe_all.sh         # single-preset sweep

# Pre-launch checkpoint audits (no server needed)
python scripts/eval/check_mlx_quant_scales.py    <repo>       # per-layer scale corruption scan
python scripts/eval/audit_mlx_quant_metadata.py               # recipe-level hazards across M4 set
python scripts/eval/validate_chat_template.py    --model <path>

# Quality + benchmarks
python scripts/eval/eval_and_chart.py --run --port 23334 --tag "Coder-30B"
bash   scripts/eval/run_all_evals.sh              # full quality sweep across presets
bash   scripts/bench/bench_256k_all.sh            # 256K single-user context sweep
```

## Critical Rules
- **SGLang + MLX only** — all models must run on SGLang with `SGLANG_USE_MLX=1`
- **No tensor parallelism** — MLX runs on a single unified memory device
- **Greedy sampling only** — MLX backend uses argmax; temperature/top-p not yet supported
  - This is a *correctness* concern: 3090 team confirmed Qwen3.6 (and likely Qwen3 family)
    enters a `"</think>\nParis\n</think>…"` repetition loop at temperature=0. Validate
    every Qwen-family model with `validate_capabilities.py` before publishing numbers.
- **MLX-format models required** — AWQ/GPTQ models from other platforms won't work; use `mlx_lm.convert` or download from `mlx-community/` on HuggingFace
- **Always use full-size models for evals + benchmarks** — never substitute a smaller
  variant (e.g. Qwen3.5-9B) "because it loads faster" when the README and quality
  table track the full model (e.g. Qwen3.5-27B). 64GB unified memory was chosen
  precisely so the full models fit; small variants exist for resource-constrained
  users, not for our characterization runs.
- Always source `scripts/common.sh` before launching
- **Model status and benchmarks** are in README.md (single source of truth)
- **OOM guard MANDATORY for long-context (≥64K) work.** macOS doesn't have a
  Linux-style OOM killer; once a process touches a page past physical RAM, the
  whole system stalls until reboot. Never run a 128K+ bench without
  `bash scripts/common/oom_guard.sh &` running in the background — it pkill's
  the SGLang server when free+inactive drops below 8 GB.
- **DO NOT raise `MEM_FRAC` / `--mem-fraction-static` above 0.7 default.**
  On unified memory the flag is a fraction of TOTAL system RAM, not "GPU
  memory." 0.85 crashed the box on 2026-05-14 (macOS compressor + swap hit
  ~150 GB effective; jetsam reaped the server; required reboot). The lever
  moves DOWN for long-context (0.4-0.5 baked into 128K+ presets) — never
  UP. Right levers for memory pressure: `--max-total-tokens`,
  `--chunked-prefill-size`, `MAX_RUNNING`, request `max_tokens`,
  `chat_template_kwargs={"enable_thinking": false}`, `--kv-cache-dtype turboquant`.
- **Long-context launch flags (validated 2026-05-11 on v0.5.11):**
  For 128K on qwen36-class models (35B weights), the working recipe is:
  `--kv-cache-dtype turboquant --chunked-prefill-size 2048 --mem-fraction-static 0.5`.
  This combination prefills 128K in ~6.5 min on Qwen3.6-35B-A3B; default
  `--mem-fraction 0.7` or `--chunked-prefill 4096` will OOM-guard-fire at
  the prefill phase. Bench tooling needs `urllib timeout=1800` for 128K
  (`scripts/bench/bench_long_context.py` defaults to 120 s — severs the
  connection mid-decode at long context). OOM root cause clarified: not
  import bloat (Python imports total ~547 MB resident); the driver is
  chunked-prefill activation tensors growing with context.

## Optimization Target
- **Primary:** single-user **256K context** performance (decode tok/s, TPOT). Measure
  at long context first — that is the workload Apple Silicon is uniquely good at.
- **Secondary:** multi-user throughput. Do not sacrifice single-user latency to win
  batch benchmarks.

## Quality Rules
- **Every new MLX model must pass `validate_capabilities.py`** before its numbers
  are added to README. The validator catches silent regressions:
  - **Thinking** — Qwen-family models with greedy decode regress into infinite
    `<think>…</think><think>…</think>` loops. Must terminate before max_tokens.
  - **Basic** — short factual answer survives the chat template (catches Devstral
    `<unk>` from doubled BOS, Gemma4 `<pad>` token emission).
  - **Vision** — VLM image roundtrip works (currently skipped on M4: VLM
    pipelines crash on MPS; see patch 002 + Devstral VLM warmup workaround).
- **Inspect chat templates BEFORE launching** any new community checkpoint with
  `validate_chat_template.py --model <path>`. Past incidents:
  - Devstral AWQ leading-BOS produced `<unk>` outputs → custom jinja template fix.
  - Qwen3.5 thinking tags in template without calibrated thinking data → infinite `<think>` loop.
  - Gemma4 thinking requires `chat_template_kwargs={"enable_thinking": true}` per request.

## Cross-Team Collaboration

We share findings with two sister projects on the same SGLang stack:
- **3090 team** — `~/AI/2x-3090-GA102-300-A1-sglang-inference` (NVIDIA, AWQ_Marlin)
- **R9700 team** — `~/AI/2x-R9700-RDNA4-GFX1201-sglang-inference` (AMD RDNA4, ROCm)

Their patches are NOT directly portable (Marlin/HIP kernels), but their **eval
scripts, calibration insights, and chat-template fixes are**. Read their CLAUDE.md
+ patches/README.md when working on a model they've already characterized — they
have likely already found and fixed the silent quality regressions you're about
to discover.

## Working Mode
- **Operate autonomously.** User reads output and interrupts with feedback — do not
  stop for confirmation. Multi-hour benchmarks and quality sweeps are allowed
  without asking.
- **Commit + push as progress is made** — small, self-contained commits, not one
  giant batch. README.md is the status document the user reads first.
- **Never stop to ask for confirmation.** If the user wants a redirect they'll
  interrupt with new signal.
