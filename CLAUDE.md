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
scripts/setup.sh                                  # venv, SGLang v0.5.11, MLX deps, apply 17 patches
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
- **OOM guard MANDATORY for long-context (≥32K) work.** macOS doesn't have a
  Linux-style OOM killer; once a process touches a page past physical RAM, the
  whole system stalls until reboot. Never run a 32K+ bench without
  `bash scripts/common/oom_guard.sh &` running in the background — it pkill's
  the SGLang server when free+inactive drops below 8 GB. (Threshold updated
  from 64K → 32K on 2026-05-21 reflecting the new empirical ceiling; the
  guard itself is unchanged but its trigger should now fire on smaller
  contexts than before.)
- **DO NOT raise `MEM_FRAC` / `--mem-fraction-static` above 0.7 default.**
  On unified memory the flag is a fraction of TOTAL system RAM, not "GPU
  memory." 0.85 crashed the box on 2026-05-14 (macOS compressor + swap hit
  ~150 GB effective; jetsam reaped the server; required reboot). The lever
  moves DOWN for long-context: **use mem-fraction 0.4 for 32K+ work**
  (was 0.5; lowered 2026-05-21 to match the new ~32K ceiling). Right
  levers for memory pressure: `--max-total-tokens`,
  `--chunked-prefill-size`, `MAX_RUNNING`, request `max_tokens`,
  `chat_template_kwargs={"enable_thinking": false}`, `--kv-cache-dtype turboquant`.
  Note: 2026-05-21 found `--chunked-prefill-size` does NOT help with the
  current OOM regression — 1024 and 2048 produced identical OOM points
  at 60K. The lever is broken for activation memory; only
  `--max-total-tokens` (cap KV size) and `--mem-fraction-static` (cap
  total reservation) actually move the needle today.
- **Long-context launch flags — current state (2026-05-21 on v0.5.12):**
  The previously-claimed 128K recipe no longer fits on M4 today.
  Empirically-measured ceiling for qwen36 (35B-MoE-4bit) is now
  **~32K input tokens** of prefill with:
  `--kv-cache-dtype turboquant --chunked-prefill-size 2048 --mem-fraction-static 0.4`
  (note `mem-fraction 0.4`, was 0.5 — needed for the new ceiling).
  At 32K the request finishes with single-digit MB free RAM. 60K-64K
  OOM-kills the server mid-prefill regardless of chunked-prefill-size
  (1024 vs 2048 produced identical OOM points in 2026-05-21 testing —
  see `evals/swebench/runs/qwen36-long-context-*-2026-05-21/`).
  Memory consumption during prefill is empirically linear in
  tokens-processed at ~0.15-0.2 MB/token, mechanism unclear (candidates:
  MLX lazy page-touch accumulation, Python intermediate-buffer growth,
  MoE expert dispatch overhead). The lever is NOT `--chunked-prefill-size`
  or `--context-length`. Closing the gap to the aspirational 128K+
  needs MLX flash-attention or aggressive per-token memory reclaim —
  upstream-MLX work, not config-tuning. Bench tooling needs
  `urllib timeout=1800` for any long-context run.
  Historical claim (kept for reference): on v0.5.11 / May 2026, this
  same recipe fit 128K and prefilled in ~6.5 min. The regression
  arrived with the v0.5.12 rebase or co-installed MLX library updates;
  not yet isolated.

## Optimization Target
- **Aspirational primary:** single-user **256K context** performance (decode tok/s, TPOT).
  Measure at long context first — that is the workload Apple Silicon is uniquely good at.
  - **Current state (2026-05-21): NOT achievable on the present stack.** Practical
    ceiling is ~32K input prefill for a 35B-MoE-4bit model. Closing the gap needs
    MLX flash-attention or per-token memory reclaim — both are upstream-MLX work.
    See the "Long-context launch flags" note above for the regression context.
- **Secondary:** multi-user throughput. Do not sacrifice single-user latency to win
  batch benchmarks.
- **Tertiary (currently the most productive workload):** single-user agentic
  coding at moderate context (8-32K typical per turn). qwen36 + opencode is
  the validated primary recommendation here; SWE-bench Lite M4-scorable
  resolved rate = 5/13 = 38.5% on the v0.5.12 stack with N=3-confirmed
  outcome stability for the tested instances.

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
