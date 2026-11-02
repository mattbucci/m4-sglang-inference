# Apple Silicon Inference: SGLang + MLX on M4 Pro

Long-context LLM inference on an Apple M4 Pro Mac mini (64 GB unified memory) with SGLang's native MLX backend. Stack: **SGLang v0.5.23** (`297e54f`) + 17 patches ([patches/README.md](patches/README.md)).

- **`qwen36`** (Qwen3.6-35B-A3B MoE+DeltaNet+VL) is the primary agentic model: codegen, vision and video STRONG, thinking VERIFIED, 256K single-user context validated with 6/6 multi-needle recall at every rung to 245K realized tokens.
- `coder-30b`, `qwen3-moe`, `qwen3-32b`, `qwen35`, `qwen35-9b-8bit`, `qwen36-27b` (Qwen3.6-27B or Qwen3.8-27B), `devstral`, `nemotron-30b` and `flint` (Recursive's coding-agent MoE) pass their gates.
- Hybrid (DeltaNet/Mamba2) presets and `flint` run with the radix cache; the trade-off is that radix-on-hybrid turns the overlap schedule off.
- `gemma4` / `gemma4-31b` are blocked at boot (see Known Issues).

Experiment specs and the open queue live in [experiments/README.md](experiments/README.md).

## Primary target: long-context agentic coding

Single-user agentic coding at long context (tool-call-heavy multi-turn sessions, 10K–100K-token codebase prefixes) is what this stack is tuned for. Decode TPOT at depth matters more than peak batch throughput; the right model keeps decode flat as context grows.

### Recommended picks (SWE-bench Lite agentic coding)

**`qwen36` is the only configuration that completes the agentic loop on M4 reproducibly across all 12 ecosystems in the 26-instance test set.**

- **Patch engagement 26/26** (`evals/swebench/runs/qwen36-26inst-p023-2026-10-31/`): a non-empty patch targeting the canonical issue file on every instance.
- **Resolved 9/26 = 34.6%** on the official Docker harness (scored on the 3090 rig; the current-stack predictions in `evals/swebench/exports/` await re-scoring). Receipt: [`exports/qwen36-docker-summary.json`](evals/swebench/exports/qwen36-docker-summary.json).
- **Sealed harness:** web tools off, opencode and its bash children under a localhost-only network sandbox, ancestor-only checkout (no branch, tag or commit past the base commit). `evals/swebench/audit_rollouts.py` scans the archive for leak channels; no scored prediction came from a session that saw the fix. Under the seal Qwen3.8-27B writes the upstream change on all three smoke instances in 4–8 min each (`evals/swebench/runs/qwen38-27b-community-sealed-131k-2026-11-01/`) and Flint bash-only at 64K scores 1/3 with no runaway turns.
- **Calibration:** the engagement-to-resolve gap is the model characterization. qwen36 patches look right (correct file, valid syntax, usually no regressions) but miss the semantic intent in most cases. Trust it for first-pass exploration; verify with tests before merging. 34.6% sits above the 14–30% band typical of ~30B models on SWE-bench Lite.
- **Multi-instance sweeps** at CTX=131K hit macOS jetsam on a single server; use per-instance server restart (`bash evals/swebench/smoke.sh` per instance). Static HumanEval/MMLU does not predict agentic ability on this stack.

| Rank | Preset | Why | Agentic verdict |
|:----:|--------|-----|-----------------|
| **1** | **`qwen36`** (Qwen3.6-35B-A3B-4bit) | Only model to complete the agentic loop on M4 in a 9-model bake-off; MoE 3B-active decode speed appears load-bearing. Use with [`no_thinking_proxy`](evals/swebench/no_thinking_proxy.py). | engagement 26/26; **Docker 9/26 = 34.6%** |
| 2 | `qwen35` (Qwen3.5-27B-4bit) | Capability-equivalent to qwen36 at ~15× the wall time: same patches where qwen36 succeeds, same misses where it fails. | SWE-bench Lite 1/3 |
| 3 | `coder-30b` (Qwen3-Coder-30B-A3B-DWQ) | Best static HumanEval (95) and decode speed; use for direct chat-completion codegen. Under opencode the loop gives up after one `glob`. | 0 edits |
| 4 | `gemma4-31b` | Top MMLU (92). Blocked at boot; zero tokens under opencode tool-call prompts when it did run. | 0 tool calls |

**Session endurance (radix-on agentic serving):** memory stays flat for 150 turns under a 32K-rolling load (free+inactive 28.4 → 26.2 GB, no guard warning, cache hits throughout, 56 tok/s decode). Appending a tool result to a cached prefix costs the model's own short-prefill rate on the recomputed tokens (~35 ms + ~1.2 ms per token: the suffix plus the remainder past the last 64-token checkpoint), 80 ms at 8.6K and 260 ms at 34.5K to the first token, with 5 ms / 35 ms of hit-path overhead on top. The chat-shape follow-up, which the template makes diverge before the primed end, costs 0.2–0.5 s on qwen36 and ~1 s on Qwen3.8-27B at 8.7K–52K ([benchmarks/session-endurance/VERDICT.md](benchmarks/session-endurance/VERDICT.md)). Deep one-shot prefills (32K+ with a big pool) use the radix-off long-context recipe.

### Flint (`flint`)

[RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) is a 39.8B-total / ~2.5B-active MoE for coding agents: 49 layers (2 dense + 47 MoE, 256 routed experts top-6 + shared), 35 sliding-window (513) + 14 full-attention layers, learned attention sinks, NoPE on the full layers, 262K context, Apache-2.0. It runs through mlx-lm's `flint` model plus patch 019; the oMLX `oQ*e` checkpoints are converted through mlx-vlm's model class, hence the patch's checkpoint-layout adapter.

| Variant | bpw | Resident | Decode tok/s @83 / @3.5K / @13.9K / @27K | Prefill tok/s @3.5K / @13.9K / @27K | Validator / codegen / thinking | Greedy vs mlx-lm |
|---------|:---:|:--------:|:----------------------------------------:|:-----------------------------------:|:------------------------------:|:----------------:|
| **`flint-oQ8e`** (default) | 8.5 | 42.3 GB | 61.9 / 59.1 / 54.4 / 49.3 | 926 / 809 / 634 | 3/3 · STRONG · VERIFIED | identical |
| `flint-oQ6e` | 6.6 | 32.9 GB | 64.8 / 61.2 / 56.0 / — | 760 / 638 / — | 3/3 · STRONG · VERIFIED | identical |
| `flint-oQ4e` | 4.7 | 23.5 GB | 70.5 / 67.5 / 61.5 / — | 708 / 595 / — | 3/3 · STRONG · VERIFIED | identical |

Measured radix-off (oQ8e at the preset's `CHUNKED=1024`, 2 GB MLX buffer cache, `MAX_RUNNING=4`). The shipped preset runs the radix cache, which keeps prefill where it is and costs ~12% of decode (50.3 / 47.0 / 43.1 tok/s at 3.5K / 13.9K / 27K) because the hybrid `no_buffer` bookkeeping turns the overlap schedule off; turn 2 of a 13.4K-token conversation then starts in 0.6 s instead of 19 s (`benchmarks/flint/2026-10-31/prefix-cache/`). A 27K prefill at chunk 1024 leaves 17 GB kernel-available and never leaves pressure level "normal"; four concurrent requests decode at 123 tok/s aggregate.

- **Reasoning:** Flint decides per turn whether to emit `<think>…</think>`; there is no template toggle. `--reasoning-parser flint` returns it as `reasoning_content`; keep that field on assistant messages when sending the conversation back. Reasoning counts against `max_tokens`: give coding requests ≥ 4,096 (Recursive recommends ≥ 32,768 for hard problems).
- **Tool calls:** `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`; `--tool-call-parser flint` returns structured `tool_calls`. Recursive's constrained grammar backend is not ported. Under opencode Flint omits the `bash` tool's required `description` argument (see Known Issues); the SWE-bench proxy backfills it (`BACKFILL_BASH_DESCRIPTION=1`).
- **Sampling:** the preset applies the checkpoint's `generation_config` (temperature 1.0 / top-p 0.95 / top-k 20) to requests that omit sampling fields; greedy requests stay greedy and pass every gate.
- **Prefix cache:** the 14 full layers share the pool (28 KB/token, 3.7 GB at 131,072 slots) and the 35 windowed caches are snapshotted at the radix boundaries (patch 021). Set `MAX_TOTAL_TOKENS=131072` explicitly for CTX=131072 runs; `EXTRA_ARGS="--disable-radix-cache"` restores the pool-free path.
- **Agentic cell:** SWE-bench Lite 3-instance smoke, 1/3 with the default tool set and 1/3 bash-only (the `(?i)` fix on `astropy-14365`), 0 runaway turns in 521 radix-on turns; the other two instances explore to the 64K cap without an edit, where qwen36 patches all three. Three instances are a smoke, so `flint` stays out of the ranking.

### DiffusionGemma structured reads (CUDA/ROCm hosts)

SGLang serves `google/diffusiongemma-26B-A4B-it` with `--dllm-algorithm Gemma4Renoise` but rejects logprobs and structured output on it. Patch 025 adds the vLLM-style *structured read*: the answer template is written into the canvas (`seed_canvas`), template positions are `pinned`, `max_steps` denoising steps run, and a `read_only` request returns per-position logprobs for `token_ids_logprob`; `constrained` limits a slot to its labels. `scripts/eval/structured_reads.py` builds the read from a question and one-token labels. The model is CUDA/ROCm-only in SGLang and Gemma 4 is blocked on MLX, so this box carries the patch and its CPU test; the end-to-end run belongs to the R9700 (TP=2 holds the 51.6 GB BF16 checkpoint).

### Probe matrix (radix on)

| Preset | codegen | vision | video | thinking |
|--------|:-------:|:------:|:-----:|:--------:|
| `qwen36` | **STRONG** | **STRONG** | **STRONG** | **VERIFIED** |
| `qwen36-27b` (Qwen3.6-27B) | **STRONG** | **STRONG** | **STRONG** | **VERIFIED** |
| `qwen36-27b` + `MODEL=mlx-community/Qwen3.8-27B-4bit` | **STRONG** | **STRONG** | **STRONG** | **VERIFIED** |
| `qwen35` | **STRONG** | **STRONG** | PARTIAL | **VERIFIED** |
| `qwen35-9b-8bit` | **STRONG** | **STRONG** | **STRONG** | pass validate gate |
| `devstral` | **STRONG** | **STRONG** | PARTIAL | n/a |
| `nemotron-30b` | **STRONG** | n/a | n/a | **VERIFIED** |
| `flint` (oQ8e / oQ6e / oQ4e) | **STRONG**† | n/a | n/a | **VERIFIED** |
| `coder-30b` / `qwen3-moe` / `qwen3-32b` | **STRONG** | n/a | n/a | pass validate gate |
| `gemma4` / `gemma4-31b` | blocked at boot | | | |

Per-preset JSON: [`benchmarks/quality/probe-trio/`](benchmarks/quality/probe-trio/). † Flint's codegen probe runs with a 4,096-token budget (`probe_all.sh` sets it) because the model reasons before answering. Qwen3.8-27B, Qwen3.6-27B and Qwen3.5-9B-8bit chat templates ignore `enable_thinking` (thinking always on).

### Choosing a model

**MoE wins at long context.** Each decode token reads the model weights and the whole KV cache. At short context weight reads dominate (MoE 1.5 GB vs dense 14 GB). At 256K the KV read grows to several GB, so keeping the weight term small matters more, not less. **DeltaNet hybrids** (Qwen3.5/3.6/3.8) alternate full attention with linear attention whose cost does not grow with context, which is why they hold decode flat at depth; the full-attention layers still pay O(n).

## Quality table (100-sample MMLU + 20 HE + LAB-Bench + Needle@{1K,4K,16K})

Qwen3 family runs `--no-thinking`; Gemma 4 runs `--humaneval-mode chat`. The Qwen3.8-27B row is from the current stack (`benchmarks/quality/Qwen38-27B-community-2026-10-31.json`); the other rows predate it, and the re-run is queued in experiments.

| Model | MMLU | HumanEval | LAB-Bench | Needle |
|:------|:----:|:---------:|:---------:|:------:|
| Gemma 4 31B-it-mxfp4 | **92%** | 50%‡ | partial | **100%** |
| Qwen3.5-27B-4bit | **90%** | **100%** | 53/125 (clean cats) | **100%** |
| Qwen3-32B-4bit-DWQ | **90%** | 95% | 33.1% | 100% |
| Qwen3.6-27B-4bit | 86% | **100%** | 42/125 | **100%** |
| Qwen3.8-27B-4bit | 84% | 90% | 38.7% | **100%** |
| Qwen3-30B-A3B-4bit-DWQ | 85% | 70% | 31.4% | 100% |
| Gemma 4 26B-A4B-it-4bit | 85% | 60%‡ | partial | 100% |
| Coder-30B-A3B-4bit-DWQ | 84% | 95% | 30.9% | 100% |
| Qwen3.5-9B-MLX-8bit | 81% | 80% | 52/150 (clean cats) | **100%** |
| Qwen3.6-35B-A3B-4bit | 80% | 85% | 60/175 | 100% |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | 82% | 65% | 8.6% | 100% |
| NVIDIA Nemotron-3-Nano-30B-A3B-4bit | 77% | 10%¶ | 19.4%¶ | 100% |
| Devstral-24B-4bit | 71% | 55% | 34.3% | 100% |

‡ Chat-mode HumanEval, not comparable to base-completion HE. "N/M (clean cats)" and "partial" are LAB-Bench subsets where the server died mid-run (full LAB-Bench needs a restart between categories). ¶ Pre-reasoning-parser numbers; the `nemotron_3` parser is wired into the preset.

In-house 4-bit builds of Qwen3.6-35B-A3B and Qwen3.8-27B (`scripts/quantize/convert_qwen36_inhouse.py`, hazard classes kept BF16) are quality-equivalent to the community checkpoints on paired fidelity A/Bs against the 8-bit references (`benchmarks/quality/fidelity/`), so they stay opt-in local builds (`MODEL=~/AI/models/<name>-inhouse`).

## Checkpoint audits

Run both gates on every new checkpoint before adding numbers here:

```bash
python scripts/eval/check_mlx_quant_scales.py <repo>    # per-layer corruption (dead dequant output)
python scripts/eval/audit_mlx_quant_metadata.py         # recipe hazards (wrong module classes quantized), header-only
```

| Checkpoint | Recipe-level hazard |
|------------|---------------------|
| `Qwen3-Coder-30B-A3B-Instruct-4bit` (non-DWQ) | **10 dead layers** (`layers.{36,46}` attention + gate all-zero); the preset uses the DWQ variant. |
| `gemma-4-26b-a4b-it-4bit` / `gemma-4-31b-it-mxfp4` | `embed_vision.embedding_projection` quantized. |
| `Qwen3.5-27B-4bit` / `Qwen3.5-9B-MLX-8bit` / `Qwen3.6-27B-4bit` / `Qwen3.8-27B-4bit` | DeltaNet `linear_attn.in_proj_a/b` quantized on every layer; all pass their gates regardless. |
| `Qwen3.6-35B-A3B-4bit` | DeltaNet `in_proj_a/b` INT4; MoE router and shared-expert gate 8-bit. |
| `Qwen3-Coder-30B-A3B-4bit-DWQ` / `Qwen3-30B-A3B-4bit-DWQ` | MoE router `mlp.gate` INT4, bounded impact in practice. |
| `Qwen3.8-27B-8bit` | Ships a `processor_config.json` naming `Qwen3VLImageProcessor`, which `AutoProcessor` cannot load; serve from a mirror without that file (fidelity reference only). |
| `Devstral-24B`, `Qwen3-32B-DWQ`, `Nemotron-3-Nano-30B-A3B`, `flint-oQ*e`, in-house builds | **clean** |

DWQ variants are used where they measured better (`coder-30b`, `qwen3-moe`, `qwen3-32b`); `qwen36` stays on standard 4-bit (its DWQ trades −5.5 MMLU for +15 HE). **Never blind-swap DWQ: recipes vary per upload; measure MMLU + HumanEval per model.**

## Known Issues

- **Sampling.** Per-request temperature/top-p/top-k/min-p, logit_bias, grammar masks and logprobs are upstream behind `--mlx-enable-sampling` (every preset sets it); greedy stays bit-stable argmax; `--sampling-defaults model` applies checkpoint generation configs. `top_k==1` is served greedy; repetition/frequency/presence penalties are not applied.
- **`gemma4` / `gemma4-31b` blocked at boot.** mlx-vlm 0.6.5's Gemma 4 attention is not recognized by the MLX attention contract, so its layers are classed as auxiliary state and hybrid component extraction fails on `lm_head`. Needs a contract entry for that module (receipt: `benchmarks/quality/probe-trio/gemma4.json`).
- **Long-context ceiling is 256K** (in=251,659 tokens, ~23 min prefill, `benchmarks/longctx-bisect/`). Recipe: `CTX=<label+64> MEM_FRAC=0.5 CHUNKED=1024 EXTRA_ARGS="--disable-radix-cache"`. Four settings carry it: the patch-008 MLX buffer-cache cap (`SGLANG_MLX_CACHE_LIMIT_GB`, default 4; uncapped, chunked prefill retains ~0.6 MB/token and jetsams ~30K), patch-015 cache pre-sizing (no power-of-two overshoot), exact pool sizing (a 192K label at CTX=210K dies where CTX=196608 completes), and 1024-token chunks (4096 kills deep prefills at ~100–113K; 1024 buys ~4.3 GB transient margin over 2048 at no speed cost). There is no decode wall: streamed decode is 61.7 tok/s @32K, 36.0 @128K, 30.2 @151K (`benchmarks/longctx-bisect/decode-curve/DECISION.md`).
- **`--kv-cache-dtype` is inert on the MLX backend.** The `fp8`/`turboquant`/`fp16` aliases (patch 002) let presets pass argparse; the pool and per-request caches store KV in the attention compute dtype (bf16). Working memory levers: `--max-total-tokens`, `CHUNKED`, `MAX_RUNNING`, request `max_tokens`, `SGLANG_MLX_CACHE_LIMIT_GB`. 4-bit KV ([MLX discussion](https://github.com/ml-explore/mlx/discussions/3134)) is a prospect, not a shipped lever.
- **Dense-attention ceiling is structural.** `mx.fast.scaled_dot_product_attention` materializes the per-chunk score tensor, which reaches 30–90 GB at deep offsets; 32K is the practical decode ceiling for dense full-attention models. DeltaNet hybrids are the workaround.
- **Radix-on-hybrid disables the overlap schedule** (upstream `no_buffer` constraint); hybrids and `flint` run the normal event loop.
- **Flint omits opencode's required bash `description` argument** on nearly every call under opencode's real prompt, so opencode rejects the call and the agent retries until timeout; no prompt wording fixes it. Use the harness proxy's backfill (`BACKFILL_BASH_DESCRIPTION=1`) or disable the bash tool.
- **Coder-Next-80B infeasible**: 42 GB of weights alone exceed the safe budget.
- **GPU driver kernel panic on large prefill chunks** (`completeMemory() prepare count underflow` in IOGPUMemory, watchdog reset; also reported in [mlx#3186](https://github.com/ml-explore/mlx/issues/3186), [mlx#3346](https://github.com/ml-explore/mlx/issues/3346), [mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883)). Mechanism: a nonzero MLX wired limit puts every new Metal buffer into a residency set committed on each add and release. Mitigation: patch 024 skips the wired limit (`SGLANG_MLX_WIRED_LIMIT_GB`, `common.sh` default `0`; the identical workload then completes, decode within 1%), and `qwen36-27b` prefills in 2,048-token chunks. With no wired limit weights and KV are pageable, so the OOM guard stays mandatory; `SGLANG_MLX_WIRED_LIMIT_GB=64` restores upstream behaviour for an A/B. A reboot with `Boot faults: wdog` and this panic string is this issue.
- **`mlx-community/Qwen3.8-27B-8bit` does not boot as shipped** (`Unrecognized image processor` from its `processor_config.json`). Point `MODEL=` at a mirror of the snapshot without that file.
- **macOS has no OOM killer**: a process touching a page past physical RAM stalls the box until reboot. **OOM guard mandatory for ≥32K work:** `bash scripts/common/oom_guard.sh &` pkills the server when kernel-reported available memory (`kern.memorystatus_level`) drops below 8 GB. vm_stat free+inactive undercounts by the file pages a checkpoint load streams through the page cache, which is why the guard reads the kernel level.
- **Jetsam can silently reap the scheduler mid-eval** (`Connection refused`, no traceback). When an eval section drops to ~0%, suspect jetsam before model regression; relaunch and re-run the section.
- **`--mem-fraction-static` is a fraction of TOTAL system RAM** on unified memory. 0.85 has hard-locked the box (compressor + swap ~150 GB effective). The 0.7 default is the ceiling; long-context presets go down to 0.4–0.5.
- **HDMI display blackout** when heavy Metal compute starts: M4 Pro quirk, not an SGLang bug.
- **`/v1/responses` is disabled at boot**: the pinned `openai==2.6.1` lacks a type the Responses module imports; chat completions serve normally. `pip install -U openai` restores the endpoint at the cost of the pin.
- **Health check timeout**: `common.sh` sets `SGLANG_HEALTH_CHECK_TIMEOUT=120` because a 4K prefill chunk at depth takes 50–80 s on Apple Silicon.

## Quick Start

```bash
./scripts/setup.sh                          # conda env, SGLang v0.5.23, MLX deps, 17 patches

./scripts/launch.sh qwen36                  # PRIMARY: MoE+DeltaNet+VL
./scripts/launch.sh coder-30b               # MoE, fastest decode, direct chat-completion codegen
./scripts/launch.sh qwen3-moe               # Qwen3-30B MoE (DWQ)
./scripts/launch.sh qwen3-32b               # Dense (DWQ)
./scripts/launch.sh qwen35                  # Qwen3.5-27B DeltaNet hybrid+VL
./scripts/launch.sh qwen35-9b-8bit          # Qwen3.5-9B 8-bit, 10 GB resident
./scripts/launch.sh qwen36-27b              # Qwen3.6-27B (MODEL=mlx-community/Qwen3.8-27B-4bit for Qwen3.8-27B)
./scripts/launch.sh devstral                # Dense+VLM (Mistral3)
./scripts/launch.sh nemotron-30b            # NemotronH (Mamba2+Attn+MoE)
./scripts/launch.sh flint                   # Recursive Flint 40B-A2.5B (oQ8e; MODEL= for oQ6e/oQ4e)
# Blocked: gemma4, gemma4-31b (boot), coder-next (64 GB); smol-docling is a VLM smoke test only.

# Long-context (OOM guard mandatory for ≥32K)
bash scripts/common/oom_guard.sh &
CTX=140000 MEM_FRAC=0.5 CHUNKED=1024 EXTRA_ARGS="--disable-radix-cache" bash scripts/launch.sh qwen36

# Agentic coding (qwen36 + opencode): the Qwen3 <think> blocks break the opencode
# loop, so requests go through a proxy that injects enable_thinking=false
bash evals/swebench/smoke.sh                # server + proxy; defaults qwen36, 1 instance
python evals/swebench/no_thinking_proxy.py &   # manual: opencode → :23335 → SGLang :23334

# Capability gates (server up on PORT 23334)
python scripts/eval/validate_capabilities.py --port 23334   # basic + thinking gate
python scripts/eval/probe_thinking.py        --port 23334   # content-aware reasoning probe
python scripts/eval/probe_vision.py          --port 23334   # content-aware image probe
python scripts/eval/probe_codegen.py         --port 23334   # 2-task / 8-test codegen probe
bash   scripts/eval/probe_all.sh                            # probe trio across presets

# Pre-launch checkpoint audits (no server)
python scripts/eval/check_mlx_quant_scales.py   <repo-or-path>
python scripts/eval/audit_mlx_quant_metadata.py
python scripts/eval/validate_chat_template.py --model <path>

bash   scripts/bench/bench_256k_all.sh                      # long-context sweep (guard first)
scripts/bench/bench_regression.sh qwen36                    # decode tripwire vs benchmarks/baselines.json
```

## Prerequisites

- Apple Silicon Mac with **≥ 64 GB unified memory** (M4 Pro Mac mini is the reference rig)
- macOS 26+ (Tahoe), Xcode CLT; Python 3.12
- ~200 GB disk for models

## Model Support

| Preset | Checkpoint (mlx-community) | Type | Wts | Decode tok/s @1K / 8K / 32K | Context | Audit |
|--------|---------------------------|------|:---:|:---------------------------:|:-------:|:------|
| `qwen36` | `Qwen3.6-35B-A3B-4bit` | MoE+DeltaNet+VL | 17 GB | 79.4 / 68.0 / 56.9 | **256K validated** | DeltaNet INT4, router 8-bit |
| `coder-30b` | `Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ` | MoE (3B active) | 16 GB | 81.9 / 55.2 / 32.0 | 32K measured, 256K label | router INT4 |
| `qwen3-moe` | `Qwen3-30B-A3B-4bit-DWQ` | MoE (3B active) | 16 GB | 81.9 / 55.1 / 31.8 | 32K measured, 256K label | router INT4 |
| `qwen35` | `Qwen3.5-27B-4bit` | DeltaNet hybrid+VL | 15 GB | 14.5 / 14.1 / 12.8 | 32K measured, 256K label | DeltaNet INT4 |
| `qwen35-9b-8bit` | `Qwen3.5-9B-MLX-8bit` | DeltaNet hybrid+VL | 10 GB | 26.8 / 25.9 / 23.4 | 32K measured | DeltaNet 8-bit |
| `qwen36-27b` | `Qwen3.6-27B-4bit` | DeltaNet hybrid+VL | 14 GB | 14.5 / 14.1 / 12.9 | 32K measured, 256K label | DeltaNet INT4 |
| `qwen36-27b` + `MODEL=…/Qwen3.8-27B-4bit` | `Qwen3.8-27B-4bit` | DeltaNet hybrid+VL | 15 GB | 14.1 @1K | 131K served (SWE-bench), 256K label | DeltaNet INT4 |
| `flint` | `flint-oQ8e` (`MODEL=` for oQ6e / oQ4e) | MoE 40B-A2.5B, SWA + sinks | 42 GB | 61.9 / 54.4 @13.9K / 49.3 @27K | 32K preset, 262K label | **clean** |
| `devstral` | `Devstral-Small-2-24B-Instruct-2512-4bit` | Dense+VL (Mistral3) | 14 GB | 16.1 / 15.1 / 11.9 | 32K measured | **clean** |
| `qwen3-32b` | `Qwen3-32B-4bit-DWQ` | Dense | 18 GB | 12.0 / 10.5 / 7.5† | 32K at `MEM_FRAC=0.5` | **clean** |
| `nemotron-30b` | `NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` | NemotronH (Mamba2+Attn+MoE) | 17 GB | — | 32K probe | **clean** |
| `gemma4` / `gemma4-31b` | `gemma-4-26b-a4b-it-4bit` / `gemma-4-31b-it-mxfp4` | MoE / Dense | 15–17 GB | blocked | blocked | vision projection INT4 |

Decode columns are the regression baselines (`benchmarks/baselines.json`, schema v2): single user, 64 output tokens, server-verified depth (`actual_input_tokens` == label), served with `CTX=36000 MEM_FRAC=0.45 SGLANG_MLX_CACHE_LIMIT_GB=2 --disable-radix-cache`. † The 0.45 pool caps `qwen3-32b` requests at 25,241 tokens, so its 32K point is served at `MEM_FRAC=0.5`. **DeltaNet holds decode flat at depth** (the hybrids retain 85–88% of short-context speed at 32K) while full-attention models halve or worse. TTFT at 32K: qwen36 59 s, coder-30b 97 s, qwen3-moe 100 s, qwen35-9b-8bit 100 s, devstral 282 s, qwen35 310 s, qwen36-27b 301 s, qwen3-32b 494 s. Full curves (128→32K, 9 points) and concurrency rows: `benchmarks/<slug>/results.json`; regression run receipts: `benchmarks/regression/`.

### Memory budget at 256K

| Model class | Weights | KV @256K (bf16) | Fits? |
|-------|:-------:|:---------------:|:------|
| MoE 3B-active (Coder-30B, Qwen3-30B-DWQ) | 16 GB | 24 GB | yes, with the long-context recipe |
| MoE+DeltaNet (Qwen3.6-35B-A3B) | 17 GB | full-attention layers only | yes (256K validated) |
| Flint 40B-A2.5B (oQ8e) | 42 GB | 7.5 GB (14 full layers, 4 KV heads) | 32K preset validated; deeper runs need headroom measured per depth |
| Dense+VL (Devstral, Qwen3.5-27B) | 14–15 GB | 42 GB | no; the dense attention ceiling applies first |
| Dense (Qwen3-32B) | 18 GB | 66 GB | no |

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
pip install "mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git@736ab0d8c1d0edc79c3d798044fe6904e2fa23be"   # Flint model
pip install "mlx-vlm==0.6.5" --no-deps    # --no-deps keeps SGLang's transformers / mlx-lm pins
```

**Gate protocol (publish precondition):** every setup run, and any stack change before its numbers are published, must pass `scripts/test_patch_gates.sh` (pristine replay + byte-identity + double-apply rejection; `setup.sh` runs it and aborts on failure), `scripts/eval/import_smoke.py` (eager import of every patch-touched module + load-bearing symbols), and the behavioral probe gate (`PRESETS="devstral qwen36" scripts/eval/probe_all.sh`; verdicts must match `benchmarks/quality/probe-trio/`, and a STRONG→FAIL flip is a stop-and-investigate finding). Receipts: `benchmarks/patch-gates/`.

| Component | Version |
|-----------|---------|
| SGLang | **v0.5.23** (`297e54f`) + 17 patches |
| MLX | 0.32.4 |
| mlx-lm | git `736ab0d` (0.32.3.dev, carries the Flint model) |
| mlx-vlm | 0.6.5 |
| PyTorch | 2.13.0 (MPS) |
| transformers | 5.19.0 |
| Python | 3.12 |

## Patches

Seventeen patches on top of `v0.5.23`; rationale per patch in [patches/README.md](patches/README.md).

| # | Patch | What |
|:-:|-------|------|
| 002 | mps-backend-defaults | MPS multimodal default + MLX `--kv-cache-dtype` aliases (boot-critical). |
| 003 | mlx-skip-quantization-check | Accept MLX checkpoints' `quantization_config` shape. |
| 005 | mlx-attn-wrapper-varargs | Attention-wrapper vararg + attribute delegation to the wrapped module. |
| 007 | mlx-multimodal-and-mps-shim | cuda→cpu redirect in the MPS tensor shim; shm page-rounding for multimodal data. |
| 008 | mlx-vlm-hybrid-integration | The VLM/hybrid path: mlx_vlm loader, attention detection, radix-for-hybrids (`no_buffer`), vision plumbing, NemotronH, MLX buffer-cache cap. |
| 014 | mlx-hf-processor-fixes | Gemma 4 image-only processor; Mistral3/Devstral processor + tokenizer resolution. |
| 015 | mlx-presize-attention-cache | Exact per-request attention-cache pre-sizing (replaces the doubling ladder for known-length requests). |
| 018 | mlx-aux-slot-admission-eviction | Admission-time auxiliary-slot eviction for the MLX pool. |
| 019 | mlx-flint-support | Recursive Flint on the MLX path: checkpoint-layout adapter, per-layer window map, sinks + NoPE in batched decode, `flint` parsers. |
| 020 | mlx-prefill-last-token-logits | Final-chunk prefill computes logits for the last position only (drops a 1.2 GB `[chunk, vocab]` transient at chunk 4096). |
| 021 | mlx-swa-prefix-snapshots | Exact prefix cache for sliding-window models: windowed caches snapshotted through the auxiliary-state pool at radix boundaries and restored on a hit. |
| 022 | mlx-pool-backed-cache-append | The pool-backed attention cache keeps already-fetched chunks in front of the next forward, so a prefix hit's split final chunk sees every new token. |
| 023 | mlx-release-finished-request | Finished requests release their MLX caches at the scheduler's KV-release hook instead of at the next decode batch. |
| 024 | mlx-wired-limit-env | `SGLANG_MLX_WIRED_LIMIT_GB` (`0`: no wired limit, no residency-set traffic; `common.sh` default): the IOGPUFamily panic mitigation. |
| 025 | dllm-structured-reads | Structured reads on DiffusionGemma (seed canvas, pinned positions, step cap, read-only finish with logprobs, constrained labels); CUDA/ROCm hosts only. |
| 026 | mlx-extend-chunk-mamba-track | Continuation prefill chunks track the DeltaNet/Mamba state at their last 64-token grid boundary, so an agentic follow-up that diverges 1–2 tokens before the primed end restores that state instead of recomputing the last chunk. Radix off: no tracking. |
| 027 | incremental-tokenizer | A chat request re-encodes only the tail after the last special token it shares with a recent prompt (4–9 ms instead of 165–250 ms at 250K tokens), and a multimodal-flagged model decodes the prompt back to text only when an image or video is attached. Ported from the 3090's patch 072; `SGLANG_INCREMENTAL_TOKENIZE=0` disables. |

## Radix prefix cache

Agentic workloads re-send the same system prompt + history every turn; the radix cache reuses the shared-prefix KV so follow-up turns prefill only their new tokens. It is on for every preset, including hybrids (DeltaNet/Mamba2 state snapshotted per prefix) and Flint (patch 021). Greedy output is deterministic on both sides of the cache, and a hit differs from a cold run only as much as two cold runs on different chunk sizes do: over 24 positions of a 16.6K-token agent request the top token matches at 20–21 of 24 with a median top-1 shift of 0.02–0.05 nats ([benchmarks/flint/2026-10-31/prefix-cache/RESULTS.md](benchmarks/flint/2026-10-31/prefix-cache/RESULTS.md)). A follow-up turn re-rendered through the Qwen3.x template diverges 1–2 tokens before the primed prompt end, so it is served from the 64-token-grid DeltaNet checkpoint (patch 026) with `AUX_SLOTS=64` checkpoint slots per hybrid preset. `EXTRA_ARGS="--disable-radix-cache"` restores the uncached path for A/B runs. Models with native context below the requested `CTX` get linear RoPE interpolation (logged at boot).

## Repo layout

```
patches/                    # 17 numbered patches; see patches/README.md
experiments/                # Experiment specs and the open queue
benchmarks/                 # Per-model JSON, regression baselines, dated receipts
  quality/                  #   MMLU / HumanEval / Needle / probe-trio verdicts / fidelity
  <slug>/                   #   throughput + long-context sweeps
evals/swebench/             # Sealed agentic-coding harness + dated run receipts
scripts/
  launch.sh                 # Unified launcher: launch.sh <preset>
  common.sh                 # Shared MLX env setup
  setup.sh                  # Full setup (env, SGLang clone, patches, gates)
  common/oom_guard.sh       # MANDATORY for ≥32K work
  bench/                    # bench_long_context.py, bench_256k_all.sh, bench_regression.sh
  eval/                     # validate_capabilities, probes, eval_and_chart, audits, patch unit tests
components/sglang/          # SGLang checkout + applied patches (cloned by setup.sh)
```

## Test System

```
Mac mini (Mac16,11)
Apple M4 Pro, 14-core CPU, 20-core GPU
64 GB unified memory (LPDDR5, ~273 GB/s)
macOS 26.2 (Tahoe)
```
