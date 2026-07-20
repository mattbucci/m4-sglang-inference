# M4 Pro Inference Project

SGLang with native MLX backend on Apple M4 Pro (64GB unified memory).

**All inference MUST use SGLang with the MLX backend.** Set `SGLANG_USE_MLX=1` for all operations.

**Stack: SGLang v0.5.15.post1 + 6 patches.** Text and VLM/hybrid paths are
production-validated. **`qwen36` is the primary agentic model** (codegen
STRONG, vision STRONG, video STRONG, thinking VERIFIED); `coder-30b` /
`qwen3-moe` / `qwen3-32b`, `qwen35`, `devstral`, and `nemotron-30b` all pass
their gates — full probe matrix in [patches/README.md](patches/README.md).
Hybrid (DeltaNet/Mamba2) presets run with the radix cache (`no_buffer`
strategy, greedy-determinism-validated prefix caching); the trade-off is that
radix-on-hybrid disables the overlap schedule. `gemma4*` is blocked by an
upstream sliding-window gap.

## Documentation

| File | Purpose |
|------|---------|
| [README.md](README.md) | Setup, benchmarks, model support, known issues |
| [rules-for-agents.md](rules-for-agents.md) | Apple Silicon constraints, launch rules, MLX specifics |
| [patches/README.md](patches/README.md) | Per-patch notes (6 patches on top of SGLang v0.5.15.post1) |

## Key Commands
```bash
scripts/setup.sh                                  # venv, SGLang v0.5.15.post1, MLX deps, apply 6 patches
# Presets — [OK] = gate-validated; [WIP] = blocked, see patches/README.md
scripts/launch.sh qwen36                          # [OK]  Qwen3.6-35B-A3B MoE+DeltaNet+VL (primary agentic)
scripts/launch.sh coder-30b                       # [OK]  Qwen3-Coder-30B-A3B-DWQ MoE
scripts/launch.sh qwen3-32b                       # [OK]  Qwen3-32B-DWQ Dense
scripts/launch.sh qwen3-moe                       # [OK]  Qwen3-30B-A3B-DWQ MoE
scripts/launch.sh qwen35                          # [OK]  Qwen3.5-27B DeltaNet hybrid+VL
scripts/launch.sh devstral                        # [OK]  Devstral 24B + image VLM
scripts/launch.sh nemotron-30b                    # [OK]  NemotronH Mamba2+Attn+MoE
scripts/launch.sh qwen35-9b-8bit                  # [unswept] Qwen3.5-9B 8-bit (same path as qwen35)
scripts/launch.sh qwen36-27b                      # [unswept] Qwen3.6-27B Dense+DeltaNet+VL (same path as qwen36)
scripts/launch.sh gemma4                          # [WIP] Gemma 4 26B MoE (sliding-window gap upstream)
scripts/launch.sh gemma4-31b                      # [WIP] Gemma 4 31B Dense (sliding-window gap upstream)
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

## Documentation Style
- **All .md docs describe CURRENT state only.** No datestamps in prose, no
  audit/verification markers (🅷, ⓡ), no "DONE"/strikethrough/"NEW" status
  markers, no "original"/"previous"/"post-rebase" framing, no was-vs-now
  comparisons ("up from X", "first time"). History lives in git history and
  commit messages — put the narrative of what changed there. Keep load-bearing
  *rationale* for rules (why a limit exists, what an incident cost) but strip
  the dates from it. Exception: `evals/*/runs/` and `benchmarks/` receipt
  files are dated run records by design and stay as-is.

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
  the SGLang server when free+inactive drops below 8 GB.
- **DO NOT raise `MEM_FRAC` / `--mem-fraction-static` above 0.7 default.**
  On unified memory the flag is a fraction of TOTAL system RAM, not "GPU
  memory." 0.85 has crashed the box (macOS compressor + swap hit ~150 GB
  effective; jetsam reaped the server; required reboot). The lever moves DOWN
  for long-context: **use mem-fraction 0.4 for 32K+ work**. Right levers for
  memory pressure: `--max-total-tokens`, `MAX_RUNNING`, request `max_tokens`,
  `chat_template_kwargs={"enable_thinking": false}`,
  `--kv-cache-dtype turboquant`, and `CHUNKED=2048` — chunk size sets the
  per-chunk transient floor: at the preset-default 4096 the transients swing
  free memory 1–11 GB per chunk and kill deep prefills at ~100-113K that
  complete at 2048 (`benchmarks/longctx-bisect/ATTRIBUTION.md`). Chunk size
  does NOT change steady per-token growth (that was the buffer-cache story,
  fixed by the patch-008 cap) — it changes the transient peak.
- **Long-context: 128K validated** for qwen36 (35B-MoE-4bit) with
  `CTX=140000 MEM_FRAC=0.5 CHUNKED=2048 EXTRA_ARGS="--disable-radix-cache" launch.sh
  qwen36 --kv-cache turboquant` — in=125,830 prefills in ~7 min, decode
  0.1 tok/s (receipts: `benchmarks/longctx-bisect/`). The prior ~32K
  ceiling was unbounded MLX buffer-cache accumulation across prefill chunks
  (~0.6 MB/token retained); patch 008 caps the cache
  (`SGLANG_MLX_CACHE_LIMIT_GB`, default 4). **192K+ still dies** at the
  contiguous-attention cache's 131K capacity-doubling boundary — that and
  decode TPOT at depth (13 s/token at 128K) are the open long-context
  constraints (see [experiments/](experiments/README.md)). Bench tooling
  needs a long urllib timeout for deep runs.

## Optimization Target
- **Aspirational primary:** single-user **256K context** performance (decode tok/s, TPOT).
  Measure at long context first — that is the workload Apple Silicon is uniquely good at.
  - Current: **128K validated**; 192K+ blocked at the contiguous-cache
    doubling boundary, and decode TPOT at depth is the other open constraint
    (see "Long-context" above).
- **Secondary:** multi-user throughput. Do not sacrifice single-user latency to win
  batch benchmarks.
- **Tertiary (currently the most productive workload):** single-user agentic
  coding at moderate context (8-32K typical per turn). qwen36 + opencode is
  the validated primary recommendation; SWE-bench Lite official Docker
  resolve rate = 9/26 = 34.6% (scored on the 3090's harness; measured on the
  previous stack pin — re-run queued in
  [experiments/](experiments/README.md)).

## Quality Rules
- **Every new MLX model must pass `validate_capabilities.py`** before its numbers
  are added to README. The validator catches silent regressions:
  - **Thinking** — Qwen-family models with greedy decode regress into infinite
    `<think>…</think><think>…</think>` loops. Must terminate before max_tokens.
  - **Basic** — short factual answer survives the chat template (catches Devstral
    `<unk>` from doubled BOS, Gemma4 `<pad>` token emission).
  - **Vision** — VLM image roundtrip works (probe_vision on the VLM presets;
    content-aware, catches placeholder-token hallucination).
- **Inspect chat templates BEFORE launching** any new community checkpoint with
  `validate_chat_template.py --model <path>`. Known hazard classes:
  - Devstral leading-BOS produces `<unk>` outputs → custom jinja template fix
    (`scripts/devstral_chat_template.jinja`).
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
