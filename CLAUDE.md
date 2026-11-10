# M4 Pro Inference Project

SGLang with native MLX backend on Apple M4 Pro (64GB unified memory).

**All inference MUST use SGLang with the MLX backend.** Set `SGLANG_USE_MLX=1` for all operations.

**Stack: SGLang v0.5.23 + 22 patches.** Text and VLM/hybrid paths are
production-validated. **`qwen36` is the primary agentic model** (codegen
STRONG, vision STRONG, video STRONG, thinking VERIFIED); `coder-30b` /
`qwen3-moe` / `qwen3-32b`, `qwen35`, `devstral`, `nemotron-30b`, and `flint`
(Recursive's coding-agent MoE, oQ8e default) all pass their gates — full probe matrix in [patches/README.md](patches/README.md).
Hybrid (DeltaNet/Mamba2) presets and `flint` run with the radix cache
(`no_buffer` strategy; hits and cold runs are each deterministic, and a hit
differs from a cold run only as much as two cold runs on different chunk sizes
do — chunk-grid bf16 noise); patch 028 keeps the overlap decode loop on.
`gemma4*` is blocked at boot
(mlx-vlm Gemma 4 attention is not recognised by the MLX attention contract).

## Documentation

| File | Purpose |
|------|---------|
| [README.md](README.md) | Setup, benchmarks, model support; known issues are GitHub issues |
| [rules-for-agents.md](rules-for-agents.md) | Apple Silicon constraints, launch rules, MLX specifics |
| [patches/README.md](patches/README.md) | Per-patch notes (22 patches on top of SGLang v0.5.23) |
| `docs/` | flint.md, quality.md (evals + checkpoint audits), long-context.md |

## Key Commands
```bash
scripts/setup.sh                                  # conda env, SGLang v0.5.23, MLX deps, apply 22 patches
# Presets — [OK] = gate-validated; [WIP] = blocked, see patches/README.md
scripts/launch.sh qwen36                          # [OK]  Qwen3.6-35B-A3B MoE+DeltaNet+VL (primary agentic)
scripts/launch.sh coder-30b                       # [OK]  Qwen3-Coder-30B-A3B-DWQ MoE
scripts/launch.sh qwen3-32b                       # [OK]  Qwen3-32B-DWQ Dense
scripts/launch.sh qwen3-moe                       # [OK]  Qwen3-30B-A3B-DWQ MoE
scripts/launch.sh qwen35                          # [OK]  Qwen3.5-27B DeltaNet hybrid+VL
scripts/launch.sh devstral                        # [OK]  Devstral 24B + image VLM
scripts/launch.sh nemotron-30b                    # [OK]  NemotronH Mamba2+Attn+MoE
scripts/launch.sh flint                           # [OK]  Recursive Flint 40B-A2.5B MoE coding agent (oQ8e default; MODEL= for oQ6e/oQ4e)
scripts/launch.sh qwen35-9b-8bit                  # [OK]  Qwen3.5-9B 8-bit (codegen/vision/video STRONG, validator 3/3; thinking always on)
scripts/launch.sh qwen36-27b                      # [OK]  Qwen3.6-27B Dense+DeltaNet+VL (codegen/vision/video STRONG, thinking VERIFIED; thinking always on)
MODEL=mlx-community/Qwen3.8-27B-4bit scripts/launch.sh qwen36-27b   # [OK]  Qwen3.8-27B Dense+DeltaNet+VL (validator 4/4, thinking/vision/codegen gates pass; thinking always on)
scripts/launch.sh gemma4                          # [WIP] Gemma 4 26B MoE (boot: mlx-vlm attention not recognised)
scripts/launch.sh gemma4-31b                      # [WIP] Gemma 4 31B Dense (same model class as gemma4)
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
for t in scripts/eval/test_*_unit.py scripts/eval/test_incremental_tokenizer.py; do python $t; done   # per-patch CPU tests (021-032), no server
python scripts/quantize/convert_qwen36_inhouse.py --dry-run     # in-house qwen36 4-bit, hazard classes BF16 (spec 07); drop --dry-run to convert
python scripts/eval/quant_fidelity.py --port 23334 --tokenizer <path> --tag <arm> [--ref-dump <8bit-dump.json>] --out <json>   # teacher-forced NLL/PPL, KL to a reference dump
python3 evals/swebench/audit_rollouts.py                       # answer-leak audit of archived SWE-bench sessions (web/network/git-history); the harness itself is sealed
bash evals/swebench/vendor_omp.sh && bash evals/swebench/smoke.sh   # sealed SWE-bench cell on the omp scaffold (PRESET= INSTANCES= TIMEOUT=; SCAFFOLD=opencode for the archived lane)
python scripts/eval/structured_reads.py --server http://<cuda-host>:30000 --tokenizer google/diffusiongemma-26B-A4B-it --question ... --kind noul   # DiffusionGemma reads (CUDA/ROCm host)

# Quality + benchmarks
python scripts/eval/eval_and_chart.py --run --port 23334 --tag "Coder-30B"
bash   scripts/eval/run_all_evals.sh              # full quality sweep across presets
bash   scripts/bench/bench_256k_all.sh            # 256K single-user context sweep
```

## Documentation Style
- **CI `docs-check` enforces these rules** (`.github/actions/docs-check/`, shared by all three rig repos): README skeleton and size, CLAUDE.md size, no dates or status markers. Open work and known issues are GitHub issues, not README sections. Run `python3 .github/actions/docs-check/docs_check.py --root .` before committing docs.
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
- **Never open pull requests against upstream packages** (SGLang, MLX, mlx-lm, mlx-vlm, or any dependency). Patches live in `patches/` and the local component tree only; no upstream PRs, no "upstream candidate" queue.
- **Real sampling supported** (upstream `--mlx-enable-sampling`, set by every
  preset) — per-request temperature/top-p/top-k/min-p; greedy (the default)
  stays bit-stable argmax, `--sampling-defaults model` applies checkpoint
  generation configs. Validate every Qwen-family model with
  `validate_capabilities.py` before publishing numbers — the 3090 confirmed
  Qwen3.x can enter `</think>` repetition loops at temperature=0, so
  thinking workloads should use model-recommended sampling.
- **MLX-format models required** — AWQ/GPTQ models from other platforms won't work; use `mlx_lm.convert` or `mlx-community/` builds
- **Always use full-size models for evals + benchmarks** — never substitute a smaller
  variant (e.g. Qwen3.5-9B) "because it loads faster" when the quality table tracks the
  full model; 64GB was chosen so the full models fit.
- Always source `scripts/common.sh` before launching
- **Model status and benchmarks** are in README.md (single source of truth)
- **OOM guard MANDATORY for long-context (≥32K) work.** macOS doesn't have a
  Linux-style OOM killer; once a process touches a page past physical RAM, the
  whole system stalls until reboot. Never run a 32K+ bench without
  `bash scripts/common/oom_guard.sh &` running in the background — it pkill's
  the SGLang server when the kernel-reported available memory
  (`kern.memorystatus_level`) drops below 8 GB.
- **DO NOT raise `MEM_FRAC` / `--mem-fraction-static` above 0.7 default.**
  On unified memory the flag is a fraction of TOTAL system RAM, not "GPU
  memory." 0.85 has crashed the box (macOS compressor + swap hit ~150 GB
  effective; jetsam reaped the server; required reboot). The lever moves DOWN
  for long-context: **use mem-fraction 0.4 for 32K+ work**. Right levers for
  memory pressure: `--max-total-tokens`, `MAX_RUNNING`, request `max_tokens`,
  `chat_template_kwargs={"enable_thinking": false}`, and `CHUNKED` (`--kv-cache-dtype`
  is inert on the MLX backend). Chunk size sets the per-chunk transient peak: 4096 kills deep
  prefills at ~100-113K that complete at 2048 or 1024.
- **Long-context: 256K validated** for qwen36 with `CTX=<label+64> MEM_FRAC=0.5
  CHUNKED=1024 EXTRA_ARGS="--disable-radix-cache"` (radix off, exact pool, buffer-cache cap,
  pre-sized cache, 1024-token chunks). Recipe, levers and the decode curve: `docs/long-context.md`.
  Bench tooling needs a long urllib timeout for deep runs.

## Optimization Target
- **Primary:** single-user **256K context** (decode tok/s, TPOT). Measure at long context
  first. Met: 256K validated, 6/6 recall at 245K, 36 tok/s true decode at 128K.
- **Secondary:** multi-user throughput; never sacrifice single-user latency for batch wins.
- **Tertiary (most productive workload):** single-user agentic coding at 8-32K per turn.
  qwen36 is the validated recommendation (SWE-bench Lite engagement 26/26; official
  Docker resolve 9/26 = 34.6%, scored on the 3090's harness).
- **Harness: one scaffold lane, omp (oh-my-pi) by default**, pinned to the 3090's version
  so cells compare across rigs. The 3090's scaffold bake-off shows every opencode /
  pi-family pair within noise (McNemar ties on 300 isolated instances), so harness
  variants are not a lever: spend cycles on models, not scaffolds. `SCAFFOLD=opencode`
  only to reproduce archived cells.

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
