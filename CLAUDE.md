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
scripts/setup.sh                                  # venv, SGLang from main, MLX deps, apply 5 patches
scripts/launch.sh devstral                        # Devstral 24B 4-bit
scripts/launch.sh coder-30b                       # Coder-30B MoE 4-bit
scripts/launch.sh coder-next                      # Coder-Next 80B 4-bit
scripts/launch.sh gemma4                          # Gemma 4 26B MoE 4-bit
scripts/launch.sh qwen35                          # Qwen3.5-27B 4-bit
# Quality + capability gates (run AFTER server is up)
python scripts/eval/validate_capabilities.py --port 23334    # basic + thinking gate
python scripts/eval/validate_chat_template.py --model <path> # static template check
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
  whole system stalls until reboot. Never run a 256K bench without
  `bash scripts/common/oom_guard.sh &` running in the background — it pkill's
  the SGLang server when free+inactive drops below 4 GB.
- **Long-context launch flags:** for ≥128K, prefer
  `--kv-cache turboquant` (4-bit, ~3.5x savings vs fp16 vs fp8's 2x) and
  `--chunked-prefill-size 2048` (halves attention scratch spikes vs 4096).
  At 64GB unified memory this is the difference between a clean run and a
  hard freeze.

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
