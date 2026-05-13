# Apple Silicon Inference: SGLang + MLX on M4 Pro

256K-context LLM inference on Apple M4 Pro (Mac mini, 64 GB unified memory) using SGLang with a native MLX backend. SGLang **v0.5.11** (commit `612785ffd`) + 11 patches (see [patches/README.md](patches/README.md)) — upstream landed our patch 001 (`kv_cache/` subpackage) in v0.5.11, so the patch set is now 7 instead of 13.

## Current Focus (2026-05-11)

**Primary target: single-user 256K context** for agentic workloads. Decode TPOT at long context > peak batch throughput. Multi-user is secondary.

**Hard constraint: every new model must pass `scripts/eval/validate_capabilities.py`** before its numbers land in this README. Sister teams (3090, R9700) repeatedly found silent regressions in checkpoints that pass MMLU/HumanEval but emit `<unk>`, infinite `<think>` loops, or `<pad>` tokens. We adopt the same gate.

### Cross-team updates

- **All three teams now on SGLang v0.5.11** (2026-05-11). 3090 bumped at `d63e191`, R9700 at `0148071`, M4 at `ebe23bb` (this repo's rebase commit). Sister teams subsequently consolidated REAM-merger patches to top-level and trimmed their resolved-issues sections — see their READMEs for the cleanup. 3090 added `evals/bake-off-2026-05-11.md` (initial cross-cell quality snapshot) and `aggregate_bakeoff` emitting per-cell JSON to their `benchmarks/quality/`; worth pulling structure into our quality flow when we re-run MMLU/HumanEval on v0.5.11.
- **R9700 (2026-05-09):** Shipped [`mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ`](https://huggingface.co/mattbucci/Qwen3-Coder-30B-A3B-REAM-AWQ) — first in-house REAM merge from upstream BF16 ([`Qwen/Qwen3-Coder-30B-A3B-Instruct`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct), 128 → 96 experts via Samsung SAIL `merge.py`, ~23 B params / 3 B active). Native AWQ — MLX cannot serve it directly without `mlx_lm.convert`, but the BF16 REAM-merged base could be MLX-quantized fresh if Coder-30B-MoE REAM-pruning lands on the M4 roadmap. **Build-from-scratch rule extension:** R9700 now also prunes themselves from upstream BF16 via Samsung SAIL, not from third-party pre-pruned BF16 (Cerebras / atbender).
- **3090 (2026-04-30):** Cross-checked our patch 013 hybrid-cache-wiring finding (Qwen3.5-27B MMLU 16.7% → 93.0%). **Not a bug class on Ampere SGLang** — `is_hybrid` model config dispatches per-layer cache type before the outer wrapper resolves `make_cache`, so the Ampere path never hits the MLX bridge's fall-through-to-uniform path. Their `qwen3-ream` and `qwen36-27b CT v3` both pass basic+thinking through `validate_capabilities.py` on TP=1 / 4–8K context without a patch-013 equivalent. Confirms the bug is MLX-bridge-specific, not quantization or upstream SGLang.

### Active work

-1. **Reasoning + tool-call parsers wired into launch presets (2026-05-13).** Sister-team gap audit: 3090 ships `--tool-call-parser qwen3_coder` on every Qwen3-Coder preset and `--reasoning-parser gemma4` on Gemma 4; R9700 does `--reasoning-parser gemma4` too. M4 was missing all three.
   - `coder-30b` + `coder-next` → `--tool-call-parser qwen3_coder`. Verified end-to-end on `coder-30b-DWQ`: a tool-spec prompt returns `finish_reason="tool_calls"` with a structured `tool_calls[{function:{name,arguments}}]` array and `content=None` (the raw `<tool_call>` XML is now parsed out). probe_codegen still STRONG 8/8 with the parser wired in — no regression on plain codegen.
   - `gemma4` + `gemma4-31b` → `--reasoning-parser gemma4`. Required so `chat_template_kwargs={"enable_thinking": true}` output gets split into `reasoning_content` instead of leaving raw `<think>...</think>` blocks in `content`. **Sub-finding:** gemma4 currently crashes on first prefill with `ValueError: [broadcast_shapes] Shapes (2,128) and (1,8,64) cannot be broadcast` inside `_sync_new_kv_to_pool` (patch 012 path). The reasoning-parser flag is accepted (server_args confirms `reasoning_parser='gemma4'`), but the KV-pool sync bug blocks runtime validation — separate Gemma 4 bug to fix.
   - `nemotron-30b` → `--reasoning-parser nemotron_3`. README quality table footnote ¶ explicitly flagged this: the model emits verbose `<think>` traces that burn the 1024-tok MC eval budget, dragging HumanEval to 10% and LAB-Bench to 19.4%. With nemotron_3 wired, the thinking trace gets parsed into `reasoning_content` and answer extraction stops consuming budget on the channel marker. Re-eval pending.

0. **Content-aware probe trio surfaced VLM image regression (2026-05-13).** Ported the 3090 team's `probe_thinking` / `probe_vision` / `probe_codegen` (deeper than `validate_capabilities.py`'s loose keyword grep — STRONG / DEGRADED / FAIL classification). First probe sweep findings:
   - **`coder-30b` (DWQ): codegen STRONG 8/8.** Matches 3090's `coder-reap-25b` baseline.
   - **`devstral`: vision FAIL.** Red-circle-on-white prompt returns `"A diagram of a circular flow chart with a central circle labeled '1' surrounded by 12 smaller circles numbered 2 through 13..."` — total fabrication. `prompt_tokens=117` is too low for a 256×256 image to be image-tokenized; image isn't reaching the model. The April 18 commit `4bb3c53` shows Devstral describing the same image as `"I see a red circle with a black outline."` (11 tok) — clear regression since v0.5.11 rebase + patches 010/011/012. Crucially `validate_capabilities.py:check_vision` keyword grep **would have passed** this (response contains "circle") — the probe is what catches the silent regression.
   - **`qwen35-9b-8bit`: vision FAIL with a different signature.** Returns `"This is a solid image of a pale pink color."` — the reasoning trace narrates "solid block of color, very light pale pink, no objects/text/patterns visible". Vision tower *is* producing features, but they're degraded to a single low-saturation color (normalization / channel-order bug suspected, not image-token misroute like Devstral).
   - **`gemma4` boot crashed on first attempt** (BrokenPipeError in scheduler init) — was 2/2 PASS on the v0.5.11 gate at `ebe23bb`, so this is flaky rather than uniformly broken; pull the boot diagnostic before flagging.
   - Per-cell JSON in `benchmarks/quality/probe-trio/*.json`. Investigate VLM regression next (bisect 010/011/012 — patch 010 resets `_position_ids` / `_rope_deltas` on every prefill, which may break image-bearing flows that need cache persistence).

1. **Memory tuning at high `MAX_RUNNING` on Qwen3.5/3.6** — patch 011 (2026-05-12) landed both the hybrid batched-decode path in `decode_batch_start` *and* the MLXAttentionWrapper extensions for Qwen3_5's gated multimodal attention. **Verified across the Qwen3.5/3.6 hybrid family at MR=2 and MR=4:** the 8-prompt random bench completes cleanly on `qwen35`, `qwen36`, and `qwen36-27b` with correct outputs to parallel math queries (`17×23`→`391`, `100-47`→`53`, `7²`→`49`, `1024÷4`→`256`); validate_capabilities matches the pre-patch README (qwen35 / qwen35-9b-8bit 1/2 PASS — known Qwen3 thinking loop, unchanged; qwen36 / qwen36-27b 2/2 PASS). Throughput at MR=2: **qwen35** (27B dense+DeltaNet) peak 34 tok/s (2.3× single-user); **qwen36-27b** (27B dense+DeltaNet) peak 34 tok/s; **qwen36** (35B-A3B MoE+DeltaNet) peak **148 tok/s** — MoE active-params win compounds with batched decode. **Wrapper rework is also backward-compatible on non-hybrid mlx_lm models** at both dense and MoE scales: **Devstral 24B (dense)** at MR=4 gets peak 40 tok/s (2.4× single-user), 8/8 successful; **Qwen3-30B-A3B (MoE 3B-active)** at MR=8 gets peak **160 tok/s** with 16-prompt queue, concurrency 15.11, 16/16 successful — proves the gated-Q detection / RoPE dispatch / args type-discrimination doesn't regress the standard-attention path even under production-style multi-user load. Tested recipe: `MAX_RUNNING={2,4} MEM_FRAC=0.4 EXTRA_ARGS="--disable-radix-cache --chunked-prefill-size 1024 --max-total-tokens 32768"`. The `--max-total-tokens 32768` cap unblocks `MR=4` at 8-prompt depth (the earlier 16-prompt bench at MR=4 still trips the OOM guard — larger queues with full 16-concurrent decode need additional headroom from `mf=0.3` or smaller chunked-prefill).
2. **Gemma 4 + Qwen3.6 multimodal** — architecturally support image/video/audio, but the mlx-community 4-bit checkpoints ship without `preprocessor_config.json`, so SGLang can't load any multimodal processor. Text-only on M4 until a re-uploaded checkpoint with the preprocessor lands; harness in `scripts/eval/test_audio.py` runs end-to-end the moment one does.
3. **Chunked-prefill scratch memory at 128K+** — direct measurement (2026-05-11) shows the Python import surface is only ~547 MB; the OOMs at 128K/256K are from chunked-prefill activation tensors, not import bloat. Knobs: drop `--chunked-prefill-size` from 4096 → 2048 (halves the per-chunk scratch) and `--mem-fraction-static` from 0.7 → 0.4. Each tradeoff pushes 256K prefill into the 30+ min regime; long-context is bandwidth-bound either way on a 64 GB Mac.
4. **64K perf data appears structurally infeasible on M4 with current MLX attention.** Targeted single-user 64K probes on 2026-05-12 hit the OOM guard mid-prefill on every hybrid tested — qwen35 (27B dense, fp8 KV) made it to ~32K, qwen36 (35B-A3B MoE, turboquant) to ~50K, qwen35-9b-8bit (9B at 8-bit weights) to ~50K. The wall isn't the model weights or the KV pool; it's the **per-chunk attention-score tensor** that `mx.fast.scaled_dot_product_attention` materializes in full (shape `chunked × current_offset × n_heads × n_layers`). At chunk=1024, offset=50K, with ~30–60 layers × 16 heads × 4 bytes that buffer alone is 30–90 GB — well past the M4's activation headroom even with `mf=0.4`. Lower `chunked` doesn't help much because the dominant factor is `current_offset`. Without a flash-attention-style block-streaming SDPA in MLX, the 64K column is genuinely out of reach for this hardware. The pre-rebase 0.2 tok/s @64K number archived below was likely an artifact of the broken inference path that patch 013 fixed. **256K is reachable only because DeltaNet hybrids' O(1) linear layers carry the inference forward; the full-attention scratch peaks once per chunk and the bench just has to absorb a long prefill time.** Decode at 32K is the practical M4 long-context ceiling for dense + standard-attention models.

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

### Quality table (v0.5.11, 100-sample MMLU + 20 HE + 25×7 LAB-Bench + Needle@{1K,4K,16K})

Full sweep completed 2026-05-11 — 10 mlx-community models on the v0.5.11 stack with `--disable-radix-cache`. Qwen3 family uses `--no-thinking` (CLAUDE.md gate, avoids infinite-think loops on greedy decode); Gemma 4 family uses `--humaneval-mode chat` (IT-tuned Gemma 4 doesn't respond to bare base completions, so HE goes through `/v1/chat/completions` with an explicit "complete this function" instruction).

| Model | MMLU | HumanEval | LAB-Bench | Needle |
|:------|:----:|:---------:|:---------:|:------:|
| Gemma 4 31B-it-mxfp4 | **92%** | 50%‡ | **41.1%** | 0%† |
| Qwen3.5-27B-4bit | **90%** | **100%** | **41.1%** | 100% |
| Qwen3-32B-4bit-DWQ | **90%** | 95% | 33.1% | 100% |
| Qwen3.6-27B-4bit | 86% | **100%** | 40.0% | 100% |
| Qwen3.6-35B-A3B-4bit | 86% | 85% | 34.3% | 100% |
| Gemma 4 26B-A4B-it-4bit | 85% | 60%‡ | 36.0% | 100% |
| Qwen3-30B-A3B-4bit-DWQ | 85% | 70% | 31.4% | 100% |
| Coder-30B-A3B-4bit-DWQ | 84% | 95% | 30.9% | 100% |
| Qwen3.5-9B-MLX-8bit | 80% | 75% | 33.7% | 100% |
| NVIDIA Nemotron-3-Nano-30B-A3B-4bit | 77% | 10%¶ | 19.4%¶ | 100% |
| Devstral-24B-4bit | 71% | 55% | 34.3% | 100% |

Sorted by MMLU (descending). Chart: `benchmarks/quality/quality_comparison.png`.

‡ Gemma 4 HumanEval ran in `--humaneval-mode chat` (not directly comparable to the other rows' base-completions HE — chat-mode prompts the model with an explicit instruction). Going through completions gives Gemma 4 0% / 5% because the IT-tuned chat template intercepts the bare function-signature prefix; the chat-mode path lifts that to 60% / 50%.
† Gemma 4 31B Needle 0% under `enable_thinking=false`. Short MC questions ("Answer with just A/B/C/D") work; long-context retrieval requires thinking. Re-eval with thinking enabled is the next Gemma-specific improvement.
¶ Nemotron-3-Nano emits verbose reasoning traces (the model's nano_v3_reasoning_parser isn't yet wired in our launch preset). The 1024-token MC budget gets consumed by `<think>` blocks, so HumanEval (base completions) and LAB-Bench (multi-letter answers) under-score; MMLU (single-letter A/B/C/D) tolerates a brief preamble and lands at 77. Chat-mode HE + a reasoning-parser flag should both bump significantly.

Standouts: Qwen3.5-27B (DeltaNet hybrid) hits MMLU 90 / HE 100 / Needle 100 — and as of 2026-05-12 the concurrent-prefill broadcast crash documented in `patches/HYBRID_CONCURRENT_TRACE_PLAN.md` is **resolved by patch 010**, lifting the preset's `MAX_RUNNING` cap from 1 to 4; Qwen3.6-27B also hits HE 100 under greedy decode without thinking budget; Gemma 4 31B leads MMLU at 92% and ties Qwen3.5-27B for top LAB-Bench at 41.1%.

## Quantization scan: 10 dead layers in coder-30b mlx-community upload (2026-05-11)

Ported the [3090 team's `check_awq_scales.py` pattern](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference) to MLX. The scanner reads every `*.safetensors` shard of an mlx-community checkpoint, groups `weight`/`scales`/`biases` triples per quantized layer, and flags layers where the combination dequantizes to a dead output:

```bash
python scripts/eval/check_mlx_quant_scales.py mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
```

**Result across the full cross-team model set:** 9 of 10 checkpoints are clean; **`mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` has 10 broken layers** — both `model.layers.36.*` and `model.layers.46.*` have their `self_attn.{q,k,v,o}_proj` and `mlp.gate` quantized as `weight` payload all-zero AND `biases` all-zero. Dequant produces identically zero output through those layers' attention + routing gate. The capability gate still passes (basic factual answers survive thanks to the surrounding 46 layers and DeltaNet/MoE redundancy), but MMLU 86.7% — slightly below the Qwen3.6-27B at 88% despite Coder-30B being a larger architecture — is consistent with degraded attention at two layers.

This is the kind of silent regression the 3090 team caught on Gemma 4 26B v3 in 16 hours; the MLX analog catches it in 30 seconds. Raw scan output in [`benchmarks/quality/v0.5.11-quant-scan-2026-05-11.txt`](benchmarks/quality/v0.5.11-quant-scan-2026-05-11.txt). Make `check_mlx_quant_scales.py` part of every new-checkpoint gate before adding numbers to the README.

**Quality lift after swapping to the DWQ variant** (`mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ`, clean: 386/386 layers healthy):

| Metric | Broken 4bit (old) | 4bit-DWQ (new) | Lift |
|--------|:----------------:|:--------------:|:----:|
| MMLU (100 samples) | 86.7% | **89.5%** | +2.8 pp |
| HumanEval (20 samples) | 75.0% | **95.0%** | **+20.0 pp** |
| Needle 1K | PASS | PASS | — |
| Decode @128 tok/s (turboquant) | 58.3 | 58.5 | flat (dead layers don't cost compute, only quality) |

The 20-percentage-point lift on HumanEval is the load-bearing data point: dead attention layers at depths 36 + 46 of a 48-layer model degrade code generation roughly twice as badly as factual recall. The `coder-30b` launch preset now points at the DWQ variant by default.

**DWQ recipes vary per upload — always measure.** Probed the DWQ variants of all 4 models with non-multimodal mlx-community DWQ uploads:

| Preset | Standard 4bit | 4bit-DWQ | Δ MMLU | Δ HumanEval | Decision |
|--------|:-------------:|:--------:|:------:|:-----------:|:--------:|
| coder-30b | 86.7 / 75.0 | **89.5 / 95.0** | **+2.8** | **+20.0** | SWAP (was broken; DWQ fixes dead layers AND specializes for code) |
| qwen3-moe (Qwen3-30B-A3B) | 83.3 / 75.0 | **91.2 / 70.0** | **+7.9** | -5.0 | SWAP (MMLU lift outweighs HE drop for general agentic work) |
| qwen3-32b | 86.7 / 87.5 | **89.5 / 95.0** | **+2.8** | **+7.5** | SWAP (wins both axes — cleanest case) |
| qwen36 (Qwen3.6-35B-A3B) | 88.0 / 80.0 | 82.5 / 95.0 | -5.5 | +15.0 | **SKIP** (5.5-pp MMLU loss not worth it for general agentic flagship) |

DWQ (Distillation Weight Quantization) optimizes the quantization against a distillation-teacher's output distribution. Different mlx-community uploads use different teacher recipes — some code-heavy (qwen36, coder-30b), some general-knowledge-heavy (qwen3-moe). **Never blind-swap DWQ; always measure both MMLU and HumanEval and decide per-model.**

Qwen3.5-27B-4bit-DWQ exists but the mlx-community upload ships without `preprocessor_config.json`; SGLang multimodal-aware launch path requires it, so the preset can't load. Probe blocked until upstream fixes (same class of bug as Gemma 4 documented in Known Issues).

`MODEL="mlx-community/Qwen3.6-35B-A3B-4bit-DWQ" launch.sh qwen36` is the override if you specifically want the code-specialist behavior on qwen36.

## Known Issues

- **Radix cache (patch 001) corrupts repeated prompts.** Identical-prompt cache hits return deterministic garbage on the 2nd+ request. **Workaround:** `EXTRA_ARGS="--disable-radix-cache"` (now the default in `run_all_evals.sh` and `test_thinking.sh`).
- **Greedy-only sampling.** MLX backend uses `mx.argmax`; temperature/top-p/top-k unsupported. On Qwen3 family this causes `<think>` loops on reasoning-heavy prompts (`validate_capabilities.py` includes a loop-detector).
- **Qwen3.5-27B / Qwen3-30B-MoE / Qwen3-32B infinite `<think>` loops.** Greedy decode + Qwen3 chat template that always emits `<think>` → loop. Short factual prompts return cleanly; fix blocked on real sampling support.
- **MLX VLM image path regressed under v0.5.11 stack** (re-discovered 2026-05-13 via content-aware probe). Devstral's red-circle prompt now returns fabricated content (`"A diagram of a circular flow chart..."` — model doesn't see the image, `prompt_tokens=117` is too low for image-tokenization). Qwen3.5-9B-8bit returns `"a solid image of pale pink color"` — vision tower produces features but they're degraded. April-18 commit `4bb3c53` shows Devstral previously described the same image correctly. Likely a v0.5.11 rebase or patch-010/011/012 regression. `validate_capabilities.py` keyword-grep passed both because "circle" / color words appeared — pure substring matching can't distinguish real recognition from fabrication. Use `scripts/eval/probe_vision.py` going forward. Sub-bug: a second image request often crashes the SGLang MLX bridge entirely (server health → 000).
- **VLM warmup crash on Devstral** — set `--skip-server-warmup` automatically in the preset.
- **Gemma 4 first-prefill crash (2026-05-13).** Both `gemma4` and `gemma4-31b` boot cleanly but die on the first prefill request with `ValueError: [broadcast_shapes] Shapes (2,128) and (1,8,64) cannot be broadcast` in `_sync_new_kv_to_pool` (patch 012 path). Symptom: SGLang scheduler SIGQUITs, `/health` returns 000, BrokenPipeError surfaces in process tree. The v0.5.11 rebase capability gate at `ebe23bb` had gemma4 at 2/2 PASS, so this regressed between rebase and 2026-05-13 — possibly a side effect of patches 010/011/012 (same window as the [VLM image regression](#mlx-vlm-image-path-regressed-under-v0511-stack)). Patch 012 specifically tries to skip non-`ContiguousKVCache` layers; Gemma 4's sliding-window attention exposes a KV head-dim mismatch (2 KV heads × 128 head_dim vs the pool's expected 1 × 8 × 64) the skip logic doesn't cover. Bisect candidate alongside the VLM regression.
- **Coder-Next-80B infeasible on current toolchain.** 42 GB weights alone exceed the M4's safe budget — model load itself OOMs (not chunked-prefill scratch). Sister R9700 (2× 32 GB total via TP=2) runs it cleanly. No path forward on a single 64 GB Mac.
- **macOS has no OOM killer** — once a process touches a page past physical RAM, the system stalls until reboot. **OOM guard mandatory for ≥64K work:** `bash scripts/common/oom_guard.sh &` pkills the SGLang server when free+inactive drops below 8 GB.
- **HDMI display blackout** — brief screen blank when the server starts heavy Metal compute. M4 Pro HDMI quirk, not an SGLang bug.

## Quick Start

```bash
./scripts/setup.sh                          # venv, SGLang clone, MLX deps, apply patches

./scripts/launch.sh coder-30b               # MoE — peak throughput, 256K
./scripts/launch.sh devstral                # Dense — image-VLM verified
./scripts/launch.sh qwen35                  # DeltaNet hybrid (32K preset)
./scripts/launch.sh gemma4                  # MoE 26B (4K preset, tight 64GB)
./scripts/launch.sh qwen3-moe               # Qwen3-30B MoE
./scripts/launch.sh qwen3-32b               # Dense
./scripts/launch.sh gemma4-31b              # Dense
./scripts/launch.sh qwen36                  # Qwen3.6-35B-A3B (DeltaNet+MoE+VL, sister-team flagship)
./scripts/launch.sh qwen36-27b              # Qwen3.6-27B Dense+DeltaNet+VL (new)

# Long-context (128K) — qwen36 validated, prefill ~6.5 min, decode ~0.10 tok/s
CTX=140000 EXTRA_ARGS="--disable-radix-cache --kv-cache-dtype turboquant \
    --chunked-prefill-size 2048 --mem-fraction-static 0.5" \
    bash scripts/launch.sh qwen36

python scripts/eval/validate_capabilities.py --port 23334   # basic + thinking gate (loose keyword grep)
python scripts/eval/probe_thinking.py --port 23334          # content-aware reasoning probe
python scripts/eval/probe_vision.py    --port 23334         # content-aware image probe (STRONG/DEGRADED/FAIL)
python scripts/eval/probe_codegen.py   --port 23334         # 8-test code-synthesis probe
bash   scripts/eval/probe_all.sh                            # sweep probe trio across all presets
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
| Devstral-24B (Mistral3) | ⚠️ | ❌ | ❌ | **Regressed 2026-05-13:** image-bearing requests return fabricated content (probe_vision FAIL); was working on Apr 18 stack (`4bb3c53`). Bisect 010/011/012/v0.5.11. |
| Qwen3.5-27B / 9B-8bit | ⚠️ | ✅ | ❌ | **9B-8bit regressed 2026-05-13:** vision features degrade to "pale pink solid". Video supported by arch, needs end-to-end test. |
| Qwen3.6-35B-A3B | ⚠️ | ✅ | ❌ | Text path validated end-to-end (capability gate PASS/PASS, MMLU 86, full perf sweep); image probe pending — sibling Qwen3.5-9B-VL on same code path is regressed, expect similar. |
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

Single-user decode speed at 128 / 4K / 16K context, fp16 KV (the default; `--kv-cache-dtype fp8|turboquant` activates the wired-up `KVQuantizer` for 256K work). Output 64 tokens, radix cache disabled.

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

### v0.5.11 long-context turboquant sweep (refreshed 2026-05-12)

Decode tok/s on the v0.5.11 stack with the long-context-tuned recipe (`--chunked-prefill-size 1024 --mem-fraction-static 0.4 --disable-radix-cache`, single user, 64 output tokens). Bench restart between models, OOM guard active. (The earlier `v0.5.11-longctx-turboquant.png` chart at `benchmarks/quality/` reflects pre-refresh numbers and will be regenerated next sweep.)

| Preset | KV | @128 | @4K | @8K | @16K | @32K |
|--------|----|:----:|:---:|:---:|:----:|:----:|
| coder-30b (Qwen3-Coder-30B MoE) | turboquant | **73.8** | 66.6 | 55.2 | 43.0 | — \* |
| gemma4 (Gemma 4 26B MoE) | turboquant | 58.9 | 55.2 | 52.8 | 49.8 | **44.7** |
| devstral (24B Dense) | turboquant | 17.4 | 17.1 | 16.2 | — \* | — \* |
| qwen35 (Qwen3.5-27B DeltaNet) | fp8 | 14.7 | 14.3 | 14.0 | 13.5 | **12.6** |
| gemma4-31b (Gemma 4 31B Dense) | turboquant | 13.5 | 12.7 | 12.4 | 11.7 | — \* |

\*Cells marked `—` are not measurement gaps from the run but indicate the OOM-guard tripped at that context probe — the static pool plus the per-chunk attention scratch (proportional to context × chunked-prefill) exceeded the activation budget before the prefill completed. Both Gemma 4 26B and Qwen3.5 carried through to 32K; the others bottomed out earlier. Raw JSON in `benchmarks/<slug>/results.json` per model (re-run 2026-05-12 03:34–05:01).

**Two patterns emerge:**

- **MoE shapes the short-context win.** Coder-30B (3B active) opens at 73.8 tok/s @128, falls to 43 by 16K. Gemma 4 26B (4B active) opens lower (58.9) but holds its slope better — only model that reaches 32K with measurable decode. Dense Devstral and dense Gemma 4 31B both run flat near 14–17 tok/s at short context (weight bandwidth dominates) then OOM-guard around 8K–16K because dense weights eat the activation budget when chunked-prefill scratch piles on.
- **DeltaNet keeps decode flat.** Qwen3.5-27B (DeltaNet hybrid + Dense full-attn) starts slow (14.7 @128) but stays nearly flat across the entire sweep — 14.7→12.6 from 128 to 32K. TPOT moves from 68 ms to 79 ms while TTFT scales 0.6 s → 272 s. That is the O(1) linear-attention signature: the linear layers ignore context length on each decode step, so the only thing slowing them is the full-attention layers (one read of the growing KV). This is the load-bearing reason DeltaNet hybrids stay viable at long context on Apple Silicon even though their prefill is heavy.

**Why the old numbers were misleadingly low.** The pre-rebase table in the archive section below shows 1.8 tok/s @16K for coder-30b. The 2026-05-12 re-run shows 43 tok/s — a 24× refresh, not a regression we recovered from. Root cause: the earlier sweeps ran with `mem-fraction-static=0.7` and `chunked-prefill-size=2048`, which together starved the activation budget and forced MLX into very slow recompute paths from 8K onward. The recipe in this table (chunked=1024 + mf=0.4 + radix off) is the empirically-tuned configuration that keeps all five models on a stable activation footprint up to the OOM-guard wall.

**Headline: turboquant works.** Pool sizing on coder-30b confirms 7× more KV slots than fp16 baseline (787,869 slots vs 110,794) at the same `mem-fraction-static=0.7`. Validation `2/2 PASS` — output identical to fp16 within tolerance. Decode tok/s at short context is within 1% of fp16 (58.3 vs 57.9 on coder-30b @128). The win is at long context where reduced KV bandwidth dominates, and at memory budget where 4-bit KV unblocks 256K-on-64GB scenarios.

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
for p in ../../patches/0[01][0-9]-*.patch; do git apply "$p"; done
cd python && cp pyproject_other.toml pyproject.toml
pip install -e ".[srt_mps]"
```

| Component | Version |
|-----------|---------|
| SGLang | **v0.5.11** (`612785ffd`) + 11 patches |
| MLX | 0.31.1 |
| mlx-lm | 0.31.2 |
| PyTorch | 2.9.1 (MPS) |
| Python | 3.12 |

## Patches

11 patches on top of SGLang `v0.5.11` (commit `612785ffd`). Upstream landed patch 001 (the `kv_cache/` subpackage) — we dropped it. The old in-tree mods 008–015 are now folded into proper patch files (006 / 008 / and inside 004). Patches 010–012 are 2026-05-12 follow-ups: mlx_vlm position-cache reset for MR>1, hybrid batched decode + Qwen3.5 gated multimodal wrapper, and pool-sync hardening. All patches apply via `git apply` against a clean v0.5.11 — no more `post_apply.py`. See [patches/README.md](patches/README.md) for the per-patch breakdown and [patches/REBASE-v0.5.11-NOTES.md](patches/REBASE-v0.5.11-NOTES.md) for the rebase narrative.

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
