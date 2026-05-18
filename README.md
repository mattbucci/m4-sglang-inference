# Apple Silicon Inference: SGLang + MLX on M4 Pro

256K-context LLM inference on Apple M4 Pro (Mac mini, 64 GB unified memory) using SGLang with a native MLX backend. SGLang **v0.5.11** (commit `612785ffd`) + 17 patches (see [patches/README.md](patches/README.md)) — upstream landed our patch 001 (`kv_cache/` subpackage) in v0.5.11.

## Primary target: long-context agentic coding

Single-user agentic coding at long context (tool-call-heavy multi-turn sessions, 10K–100K-token codebase prefixes) is what this stack is tuned for. Decode TPOT at long context matters more than peak batch throughput; the right model is one whose decode stays flat as context grows.

### Recommended picks (2026-05-18, ranked by SWE-bench Lite agentic-coding ability)

The headline: **`qwen36` is the only configuration verified to produce
real SWE-bench Lite patches on M4, reproducibly, at scale, across all
12 ecosystems present in our test set.** 2026-05-18 cumulative with
proxy + hardened harness: **19/21 (90.5%) patches across 12 ecosystems**
(astropy, Django, flask, matplotlib, pytest, requests, scikit-learn,
seaborn, sphinx, sympy, pylint, xarray) — all targeting canonical
upstream-fix files. Wall time 73-908 s/instance. Used with
`no_thinking_proxy`. The 2 misses both involve **adding behavior**
(django-11019 = N-way topological-sort rewrite; flask-4045 = adding
`name` validation) rather than fixing visible bugs — a real model-class
ceiling, not a tuning issue.

**Important method note for multi-instance sweeps:** At CTX=131K on M4,
multi-instance single-server sweeps hit recurring macOS jetsam (the
SGLang scheduler gets reaped silently). The hardened
`evals/swebench/run_rollouts.py` (per-instance preflight, landed
2026-05-18) detects and aborts cleanly on dead upstream. For sweeps
beyond 1-2 instances use the **per-instance server restart** pattern:
one fresh `bash evals/swebench/smoke.sh` invocation per instance. Adds
~30s/instance overhead and fully sidesteps the jetsam floor.
Static HE/MMLU scores do not predict agentic-coding capability on this
stack. See
[`evals/swebench/runs/4pick-scorecard-2026-05-18/`](evals/swebench/runs/4pick-scorecard-2026-05-18/)
for the model bake-off,
[`qwen36-3instance-2026-05-18/`](evals/swebench/runs/qwen36-3instance-2026-05-18/)
for within-domain reproducibility, and
[`qwen36-crossrepo-2026-05-18/`](evals/swebench/runs/qwen36-crossrepo-2026-05-18/)
for cross-ecosystem generalization.

| Rank | Preset | Why | Agentic verdict |
|:----:|--------|-----|-----------------|
| **1** | **`qwen36` (Qwen3.6-35B-A3B-4bit MoE+DeltaNet)** | Only model to complete the agentic loop. MoE keeps decode fast; DeltaNet keeps it flat at long context — **at CTX=131072 the same patch produces in 123 s vs 122 s at CTX=32768 (no perf cliff at 4× context)**. Vision-capable too. Use with [`no_thinking_proxy`](evals/swebench/no_thinking_proxy.py). For multi-instance sweeps use per-instance server restart (avoids jetsam at 131K). | **SWE-bench Lite 19/21 (90.5%) across 12 ecosystems** (astropy 6/6, django 4/5, flask 0/1, matplotlib 1/1, pylint 1/1, pytest 1/1, requests 1/1, scikit-learn 1/1, seaborn 1/1, sphinx 1/1, sympy 1/1, xarray 1/1), 73-908 s/instance, 506-2563 B patches targeting canonical issue locations; **131K-context validated**. The 2 misses are model-class ceiling cases (add-behavior, not fix-behavior). |
| 2 | `qwen35` (Qwen3.5-27B-4bit DeltaNet) | **Capability-equivalent to qwen36, NOT higher-capability.** On qwen36's only real miss (`django__django-11019` — a 4929-byte topological-sort algorithmic rewrite), qwen35 at `TIMEOUT=1800` ALSO produced 0 bytes after timing out at 1803 s with 4 tool calls. Static MMLU 90 (hardened) does not convert to agentic-coding ceiling above qwen36's MoE. qwen35 succeeds where qwen36 succeeds (same 506-byte patch on astropy-12907 at 15× the wall) and fails where qwen36 fails. **No agentic value over qwen36.** Use only when DeltaNet 27B-dense is required for non-agentic reasons. | SWE-bench Lite 1/3 (1 success at TIMEOUT=1800; 2 fails including the algorithmic-instance retest) |
| 3 | `coder-30b` (Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ) | Best static HumanEval (95) and decode speed (73 tok/s @128) — use for **direct chat-completion code generation**, NOT agentic flows. Under greedy MLX + opencode the agent loop gives up after one `glob`, regardless of thinking config. | 1 glob then asks user, 0 edits |
| 4 | `gemma4-31b` (gemma-4-31b-it-mxfp4) | Top MMLU (92) + Needle 100. Vision/video STRONG via patches 014+018. **Not usable through opencode for agentic coding** — under tool-call-augmented prompts the model emits zero tokens before timeout (re-tested 2026-05-18 after `tool_call: true` config fix; same 0/0/603s result). Use for direct chat-completion code generation and reasoning, not tool-call flows. | 0 tool calls, 0 emission under opencode tool prompts |

Everything else in [Model Support](#model-support) is either smaller-variant, untested in the latest sweep, or has known regressions (Devstral video FAIL on greedy MLX, Qwen3-32B 16K ceiling, etc.).

### Active work (2026-05-17)

Resolved items moved to [patches/README.md → Shipped narrative](patches/README.md). The list below is the *current* backlog only.

1. **Eval harness hardening — server-death detection across all eval functions.** Patched `needle_eval` 2026-05-17 to tag connection-class exceptions with `server_dead=True` after root-causing the apparent Qwen3.5/3.6 Needle regression (turned out to be macOS jetsam silently reaping the sglang scheduler mid-LAB-Bench when a 7K+ token CloningScenarios prompt pushed peak memory past the threshold). `mmlu_eval`, `humaneval_eval`, `labbench_eval` still have bare `except: return False` patterns and may produce the same silent ghost regressions. `run_all_evals.sh` should also restart the server between phases so a mid-eval scheduler death doesn't poison every subsequent metric. Memory: [project_eval_jetsam_artifact.md](~/.claude/projects/-Users-letsrtfm-AI/memory/project_eval_jetsam_artifact.md).

2. **LAB-Bench cross-family drift verification.** Median -5 pp drop on LAB-Bench from the mlx-vlm 0.4.4 → 0.5.0 bundled with the patch cycle. Currently treated as real tokenization drift, but the Needle false alarm proved jetsam can contaminate sequential evals — the LAB drift could be partial server-degradation too. Won't know until #1 is hardened and the comparisons re-run.

3. **Chunked-prefill scratch memory at 128K+.** Python import surface is only ~547 MB; the OOMs at 128K/256K are from chunked-prefill activation tensors, not import bloat. Working knobs: `--chunked-prefill-size 2048` (halves per-chunk scratch) + `--mem-fraction-static 0.4` (long-context preset default). Each push deeper into context costs prefill time — 256K prefill is ~30 min on the patched stack; long-context is bandwidth-bound either way on a 64 GB Mac.

4. **64K dense-attention ceiling is structural.** The per-chunk attention-score tensor `mx.fast.scaled_dot_product_attention` materializes (shape `chunked × current_offset × n_heads × n_layers`) hits 30–90 GB before completing at offset=50K. Without flash-attention-style block-streaming SDPA in MLX, **decode at 32K is the practical M4 ceiling for dense + standard-attention models.** 256K is reachable only because DeltaNet hybrids' O(1) linear layers carry inference forward; the full-attention scratch peaks once per chunk and the bench just absorbs the long prefill. Picking a DeltaNet model (`qwen35`, `qwen36`, `qwen36-27b`) is the practical workaround for now.

### Full probe-trio+video sweep (2026-05-16, 12 presets)

| Preset | codegen | vision | video | thinking |
|--------|:-------:|:------:|:-----:|:--------:|
| `coder-30b` | **STRONG** | n/a | n/a | n/a |
| `devstral` | PARTIAL† | **STRONG** | FAIL¶ | n/a |
| `gemma4` | **STRONG** | **STRONG**§ | PARTIAL§§§ | **VERIFIED** |
| `gemma4-31b` | **STRONG** | **STRONG**§ | **STRONG**§ | **VERIFIED** |
| `qwen3-moe` | **STRONG** | n/a | n/a | n/a |
| `qwen3-32b` | **STRONG** | n/a | n/a | n/a |
| `qwen35` | **STRONG** | **STRONG** | DEGRADED⁂ | (Qwen3 greedy-loop) |
| `qwen35-9b-8bit` | **STRONG** | **STRONG** | **STRONG** | (Qwen3 greedy-loop) |
| `qwen36` | **STRONG** | **STRONG** | **STRONG** | **VERIFIED** |
| `qwen36-27b` | **STRONG** | **STRONG** | **STRONG** | **VERIFIED**‡ |
| `nemotron-30b` | **STRONG** | n/a | n/a | **VERIFIED** |
| `nemotron-omni` | **STRONG** | **STRONG**§§ | **STRONG**※ | **VERIFIED** |

† Devstral codegen PARTIAL on the 8-test set; `merge_intervals` SyntaxError'd because the model emitted a U+2014 em-dash inside a comment. Probe-side hardening issue (`strip` non-ASCII from extracted code), not a model regression.

‡ `probe_thinking` default raised 600 → 2000 tokens (2026-05-17) so Qwen3.6-27B's verbose dense greedy-MLX trace reaches `</think>` cleanly. Real long-thinking workloads on this family should still budget `max_tokens ≥ 2000` per request, or pass `chat_template_kwargs={"enable_thinking": false}` for budget-bound MC evals.

§ Gemma 4 vision+video unblocked 2026-05-16 by patch 018 (`mlx-gemma4-unpatch`). SGLang's transformers `Gemma4ImageProcessor` returns pre-patched `(B, max_patches, patch_pixels)` 3D pixel_values (via `siglip2.convert_image_to_patches`: `reshape(C, npH, pH, npW, pW).permute(1, 3, 2, 4, 0).reshape(npH·npW, pH·pW·C)`) plus an `image_position_ids` `(B, max_patches, 2)` tensor with `-1` padding markers. mlx_vlm's gemma4 vision tower expects raw `(B, C, H, W)` and does its own patchification. Patch 018 detects Gemma 4 by pixel_values shape `(B, max_patches, 768)` + presence of `image_position_ids`, filters padding, sorts patches by row-major grid order, reverses the permute+reshape to reconstruct raw pixels, and drops `image_position_ids` from `mm_kwargs`. Verified: `gemma4-31b` STRONG on both vision (the dense 31B-mxfp4 variant) and video (both "right" / "down" 2-token completions correct); `gemma4` (26B-A4B MoE) vision STRONG, video PARTIAL (says "down" on both probes — model-quality artifact at this smaller variant, not a plumbing issue).

¶ Devstral video FAIL: model echoes the prompt + emits gibberish for the "down" probe. Multi-image input reaches the tokenizer (471 prompt tokens for 6 frames is plausible), but Mistral-Small-3.1's expected per-image token-frame separators differ from what SGLang's processor + our axis-0 concat produces. Separate Mistral-specific gap; single-image vision STRONG.

⁂ qwen35 (full 27B) video DEGRADED: model recognizes motion but the dense decode is verbose enough that the one-word answer the probe wants doesn't make it to the first 80 completion tokens. qwen35-9b-8bit (smaller, sparser) and qwen36 (35B-A3B MoE) both nail STRONG on the same probe.

§§ `nemotron-omni` (`Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit`) loads end-to-end via mlx-vlm 0.5.0 + patch 015 multi-image plumbing + patch 016 bfloat16→float32 numpy conversion + the librosa runtime dep for `ParakeetExtractor`. SGLang's `nano_nemotron_vl.py` processor returns `(3, H, W)` bfloat16 tensors; NumPy can't represent bfloat16, so patch 015's `.numpy()` raised `TypeError: Got unsupported ScalarType BFloat16` and the outer `Exception` block silently dropped pixel_values to None — model ran text-only on image requests, producing the "the word 'Terror' repeated in a grid" fabrication for a red-circle-on-white prompt. Patch 016 catches the TypeError and falls back to `t.to(torch.float32).numpy()`. **codegen STRONG 8/8** + **vision STRONG** ("A red circle with a black outline is centered on a white background") + **thinking VERIFIED**.

※ `nemotron-omni` video unblocked 2026-05-16 by patch 017. Two related fixes in `nano_nemotron_vl.py`: (a) force `dynamic_resolution=False` on the MLX backend so the static tile path runs (per-image num_tokens = tiles × 256 exactly matches mlx_vlm's vision-tower feature count); (b) replace each `<image>` placeholder with its own rendered image one at a time — upstream used `replace(..., "".join(rendered_images), 1)` which put the full concatenation into the first `<image>` and left the remaining N-1 `<image>` tokens as stray IMG_CONTEXT placeholders. For 6-frame video that produced 1541 placeholders (1536 inserted + 5 stray) vs 1536 vision-tower features → `_merge_features` ValueError. Per-placeholder replacement gives exact token=feature alignment. Verified STRONG: 3-token completions ("right" / "down"), 1632 prompt tokens (matches expected 6 × 256 + chat-template overhead).

§§§ `gemma4` (26B-A4B MoE) video PARTIAL: vision plumbing fully works (1598-token prompt accepted, 2-token completions), but the smaller MoE variant gave "down" for both right and down probes. Same probe `gemma4-31b` nails both directions — this is a quality regression at the smaller variant, not a plumbing failure.

Per-preset JSON: [`benchmarks/quality/probe-trio/`](benchmarks/quality/probe-trio/). Headline: **8/8 VLMs probe_vision STRONG** (full M4 model set), **6/8 VLMs probe_video STRONG** (Qwen3.5/3.6 + Nemotron-Omni + Gemma 4 31B); only `gemma4` (PARTIAL — quality at small MoE) and `devstral` (Mistral multi-image format gap) remain non-STRONG on video.

### Quality table (v0.5.11+17 patches, 100-sample MMLU + 20 HE + 25×7 LAB-Bench + Needle@{1K,4K,16K})

Numbers below are from the **2026-05-18 hardened-harness audit** (6 of 12 models re-evaluated with the post-#45 jetsam-detect across MMLU/HE/LAB/Needle — see `benchmarks/quality/<Model>.hardened-pre.json` for the contaminated runs the hardened version replaces). Untagged rows are still the 2026-05-11 baseline pending refresh.

Qwen3 family uses `--no-thinking`; Gemma 4 family uses `--humaneval-mode chat` (IT-tuned Gemma 4 doesn't respond to bare base completions).

| Model | MMLU | HumanEval | LAB-Bench | Needle |
|:------|:----:|:---------:|:---------:|:------:|
| Gemma 4 31B-it-mxfp4 🅷 | **92%** | 50%‡ | partial⁂ | **100%**† |
| Qwen3.5-27B-4bit 🅷 | **90%** | **100%** | 53/125‡‡ (clean cats) | **100%** |
| Qwen3-32B-4bit-DWQ | **90%** | 95% | 33.1% | 100% |
| Qwen3.6-27B-4bit 🅷 | 86% | **100%** | 42/125‡‡ (-8.8 pp real drift) | **100%** |
| Qwen3-30B-A3B-4bit-DWQ | 85% | 70% | 31.4% | 100% |
| Gemma 4 26B-A4B-it-4bit 🅷 | 85% | 60%‡ | partial⁂ | 100% |
| Coder-30B-A3B-4bit-DWQ | 84% | 95% | 30.9% | 100% |
| Qwen3.5-9B-MLX-8bit 🅷 | 81% | 80% | 52/150‡‡ (clean cats) | **100%** |
| Qwen3.6-35B-A3B-4bit 🅷 | 80% | 85% | 60/175 (flat vs pre) | 100%★ |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit ⓡ | 82% | 65%❋ | 8.6%❋ | 100% |
| NVIDIA Nemotron-3-Nano-30B-A3B-4bit | 77% | 10%¶ | 19.4%¶ | 100% |
| Devstral-24B-4bit | 71% | 55% | 34.3% | 100% |

Sorted by MMLU (descending). Chart (still showing 2026-05-11 numbers): `benchmarks/quality/quality_comparison.png` — regeneration pending.

🅷 **Hardened-audit** verified (2026-05-18). The previous "ⓡ" row's reported numbers were partially eval-harness artifacts; the audited number above is from a re-run with jetsam-detect tagging. Pre-hardening snapshots at `<Model>.hardened-pre.json` for direct comparison.

ⓡ Re-evaluated 2026-05-17 on patches-015-018 + mlx-vlm-0.5.0 but not yet re-run on the hardened harness. Numbers may shift on hardened re-eval.

‡ Gemma 4 HumanEval runs in `--humaneval-mode chat` (not comparable to base-completions HE — the IT-tuned chat template intercepts the bare function-signature prefix).

‡‡ **LAB-Bench partial-data score** — the original 175-sample full run hit jetsam mid-eval on CloningScenarios / SeqQA, so the hardened result is over only the cleanly-completed categories. Direct comparison to pre-patch 175-sample numbers is unfair; the listed N/M shows the cleanly-completed subset.

⁂ LAB-Bench partial: server died early in the run (during DbQA on both Gemma 4 variants). Only LitQA2 + partial DbQA ran clean. Not enough data for a meaningful comparison vs pre-patch. Full LAB-Bench requires restarting the server between categories — a follow-up patch on `scripts/eval/run_all_evals.sh`.

† **Gemma 4 31B Needle 0% → 100%** is the real headline upgrade of the mlx-vlm 0.4.4 → 0.5.0 cycle. Confirmed twice now (2026-05-17 manual fresh-server retest + 2026-05-18 hardened full-eval). Same `enable_thinking=false` config; the model just retrieves correctly now.

★ Qwen3.6-35B-A3B MMLU dropped 86 → 80 in the hardened audit. This is REAL model-output drift (MMLU runs first, before jetsam fires, on simple single-letter MC questions). The only confirmed cross-family MMLU regression from mlx-vlm 0.4.4 → 0.5.0. LAB-Bench at 60/175 is identical to pre-patch (per-category shifts cancel out).

**Audit summary** (6 of 6 picks with reported drift re-verified):

| Model | Real Δ from mlx-vlm 0.5.0 | Was reported as |
|-------|---------------------------|-----------------|
| Qwen3.6-35B-A3B | **MMLU -6 pp** (real) | MMLU -4 pp ✓ |
| Qwen3.6-27B | **LAB -8.8 pp** (real, smaller) | LAB -16.6 pp (×2 inflated) |
| **Gemma 4 31B** | **Needle 0% → 100%** (real gain) | Same ✓ |
| Qwen3.5-9B-8bit | flat | -3.4 pp LAB (jetsam) |
| Qwen3.5-27B | flat (+2.4 pp clean cats) | -6.9 pp LAB (jetsam) |
| Gemma 4 26B | flat (partial data) | -5.1 pp LAB (jetsam) |

**3 real changes, 3 jetsam artifacts.** The "Qwen family cross-family LAB regression" narrative is dead. The `#45` hardening permanently prevents the silent-zero-section pattern from re-occurring. Real drift from mlx-vlm 0.4.4 → 0.5.0 is small and per-model: one MMLU regression (qwen36), one LAB regression (qwen36-27b), one Needle gain (gemma4-31b). Everything else is flat within run-to-run variance.

★ Nemotron-3-Nano-Omni-30B-A3B-Reasoning (new row, no prior baseline). Patches 016+017 unblocked end-to-end. vs the text-only nemotron-30b sibling: MMLU +5pp, HumanEval +55pp (10%→65%), LAB-Bench -10.8pp (19.4%→8.6%). The reasoning-mode wrapper consumes a chunk of the LAB-Bench answer budget on multi-letter biology QA — HE/MMLU gain comes from the same reasoning capability.

¶ Nemotron-3-Nano emits verbose reasoning traces (the model's nano_v3_reasoning_parser isn't yet wired in our launch preset). The 1024-token MC budget gets consumed by `<think>` blocks, so HumanEval (base completions) and LAB-Bench (multi-letter answers) under-score; MMLU (single-letter A/B/C/D) tolerates a brief preamble and lands at 77. Chat-mode HE + a reasoning-parser flag should both bump significantly.

Standouts (post-refresh): **Gemma 4 31B** leads MMLU at 92% and is now the top long-context retrieval model (Needle 100% confirmed); **Qwen3.5-27B** (DeltaNet hybrid) preserves MMLU 90 / HE 100 / LAB 34.3; **Qwen3.6-27B** holds HE 100 under greedy decode without thinking budget; **Nemotron-Omni** is the new HE leader among text-only-comparable models at 65% with full multimodal support.

The headline gain of the patch cycle: **Gemma 4 31B Needle now works** (the dominant prior footnote regression is closed). The originally-reported "Qwen3.5/3.6 Needle 0%" turned out to be a measurement bug (server jetsam reaping mid-eval) and is corrected to 100% across all three Qwen variants on the patched stack. **No real Needle regression exists.** LAB-Bench cross-family -5 pp drift remains and tracks the mlx-vlm 0.4.4 → 0.5.0 version bump's effect on long-passage tokenization.

## Quantization scan: 10 dead layers in coder-30b mlx-community upload (2026-05-11)

Ported the [3090 team's `check_awq_scales.py` pattern](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference) to MLX. The scanner reads every `*.safetensors` shard of an mlx-community checkpoint, groups `weight`/`scales`/`biases` triples per quantized layer, and flags layers where the combination dequantizes to a dead output:

```bash
python scripts/eval/check_mlx_quant_scales.py mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
```

**Result across the M4 mlx-community model set:** 9 of 10 checkpoints are clean; **`mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` has 10 broken layers** — both `model.layers.36.*` and `model.layers.46.*` have their `self_attn.{q,k,v,o}_proj` and `mlp.gate` quantized as `weight` payload all-zero AND `biases` all-zero. Dequant produces identically zero output through those layers' attention + routing gate. The capability gate still passes (basic factual answers survive thanks to the surrounding 46 layers and DeltaNet/MoE redundancy), but MMLU 86.7% — slightly below the Qwen3.6-27B at 88% despite Coder-30B being a larger architecture — is consistent with degraded attention at two layers.

This is the kind of silent regression the 3090 team caught on Gemma 4 26B v3 in 16 hours; the MLX analog catches it in 30 seconds. Raw scan output in [`benchmarks/quality/v0.5.11-quant-scan-2026-05-11.txt`](benchmarks/quality/v0.5.11-quant-scan-2026-05-11.txt). Make `check_mlx_quant_scales.py` part of every new-checkpoint gate before adding numbers to the README.

### Calibration metadata audit: 10 latent recipe issues across the model set (2026-05-13)

Ported the 3090 team's [`audit_calib_quality.py`](https://github.com/mattbucci/2x-3090-GA102-300-A1-sglang-inference/blob/main/scripts/eval/audit_calib_quality.py) (commit `6f7f2ae`) — pure HF metadata audit, Range-fetches safetensors headers only (no weight download, no model load), flags recipe-level mistakes invisible to the validator. The 3090 version inspects AWQ's `qweight / scales / qzeros` triple; ours [`audit_mlx_quant_metadata.py`](scripts/eval/audit_mlx_quant_metadata.py) inspects MLX's `weight / scales / biases` (4-bit/8-bit) plus mxfp4's `weight / scales`.

First sweep across the 12 mlx-community checkpoints we ship surfaces problems sister teams have lost 16h calibrations to:

| Checkpoint | Recipe-level finding |
|------------|---------------------|
| `mlx-community/gemma-4-26b-a4b-it-4bit` | `embed_vision.embedding_projection` is INT4 — **exactly the layer sister teams' 2026-05-06 disaster zero-scaled**. Likely degrades image features even after the missing-preprocessor block resolves. |
| `mlx-community/gemma-4-31b-it-mxfp4` | Same `embed_vision.embedding_projection` is mxfp4 — same hazard, different format. |
| `mlx-community/Qwen3.5-27B-4bit` | DeltaNet `linear_attn.in_proj_a`/`in_proj_b` INT4 across all 48 layers (96 total). Sister teams' rule: these are recurrent-state gate scalars that **must** stay BF16; error accumulates under INT4 → recurrent state diverges. **Strong candidate for the root of the known Qwen3.5 infinite-`<think>` loop** (previously attributed solely to greedy decode). |
| `mlx-community/Qwen3.5-9B-MLX-8bit` | Same DeltaNet quantization at 8-bit (still violates the BF16 rule). |
| `mlx-community/Qwen3.6-35B-A3B-4bit` | DeltaNet in_proj_a/b INT4 (60 layers) **and** MoE `mlp.gate` router INT4 (40 layers — top-k routing under INT4). |
| `mlx-community/Qwen3.6-27B-4bit` | DeltaNet in_proj_a/b INT4 (96 layers). |
| `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ` | MoE router `mlp.gate` INT4 (48 layers). Still scores MMLU 89.5 / HE 95 — bounded impact. |
| `mlx-community/Qwen3-30B-A3B-4bit-DWQ` | Same router quantization (48 layers). |
| `mlx-community/Qwen3-Coder-Next-4bit` | Same router quantization (48 layers — 80B infeasible on M4 regardless). |

**Clean: Devstral-24B, Qwen3-32B-DWQ, Nemotron-3-Nano-30B-A3B** (no DeltaNet, no MoE routers in the model, vision tower fully BF16 on Devstral).

Raw output: [`benchmarks/quality/mlx-metadata-audit-2026-05-13.txt`](benchmarks/quality/mlx-metadata-audit-2026-05-13.txt) (+ `.json`). Bake this into every new-checkpoint gate alongside `check_mlx_quant_scales.py` — the two are complementary: metadata audit catches **recipe** mistakes (wrong things quantized), scale scanner catches **per-layer** corruption (right things quantized badly).

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
- **VLM warmup crash on Devstral** — set `--skip-server-warmup` automatically in the preset.
- **Gemma 4 needs `--disable-radix-cache`** (baked into both gemma4 presets). `MlxKVPool` assumes homogeneous attention shapes that Gemma 4's heterogeneous sliding+full layout doesn't satisfy. Workaround unblocks one-shot evals; agentic prefix reuse loses the radix cache. Full root-cause + fix options in [patches/README.md → Gemma 4 radix-cache](patches/README.md#gemma-4-radix-cache-root-cause-2026-05-13) + [`patches/RADIX_CACHE_GEMMA4_ROOT_CAUSE.md`](patches/RADIX_CACHE_GEMMA4_ROOT_CAUSE.md).
- **Coder-Next-80B infeasible on current toolchain.** 42 GB weights alone exceed the M4's safe budget — model load itself OOMs (not chunked-prefill scratch). Sister R9700 (2× 32 GB total via TP=2) runs it cleanly. No path forward on a single 64 GB Mac.
- **macOS has no OOM killer** — once a process touches a page past physical RAM, the system stalls until reboot. **OOM guard mandatory for ≥64K work:** `bash scripts/common/oom_guard.sh &` pkills the SGLang server when free+inactive drops below 8 GB.
- **macOS jetsam can silently reap the sglang scheduler mid-eval.** Root-caused 2026-05-17: a long LAB-Bench `CloningScenarios` prompt (7K+ tokens) pushed peak memory past jetsam's threshold on Qwen3.5/3.6 under mlx-vlm 0.5.0. No traceback, no log line — just `Connection refused` on every subsequent request. `needle_eval` now tags `server_dead=True` on `URLError`/`RemoteDisconnected` to distinguish this from a real model miss; `mmlu_eval`/`humaneval_eval`/`labbench_eval` still don't. When you see a Needle/LAB-Bench section drop to ~0% in `benchmarks/quality/<Tag>.json`, suspect jetsam before suspecting model regression: re-launch the server fresh and re-run just that eval section. Memory: [project_eval_jetsam_artifact.md](~/.claude/projects/-Users-letsrtfm-AI/memory/project_eval_jetsam_artifact.md).
- **`--mem-fraction-static` is a fraction of TOTAL system RAM on unified memory, not "GPU memory".** Discrete-GPU intuition does not transfer. `MEM_FRAC=0.85` means MLX takes 85% of the *whole* 64 GB pool, leaving the OS itself ~10 GB for kernel + Metal compile buffers + page cache + transient activation + everything else; tested 2026-05-14 → macOS compressor + swap hit ~150 GB effective usage and jetsam reaped the server, hard-locked the box. **The default 0.7 ceiling is load-bearing.** Long-context presets override it *down* to 0.4-0.5 because activation scratch dominates — that direction is the validated lever; the *up* direction is not.
- **HDMI display blackout** — brief screen blank when the server starts heavy Metal compute. M4 Pro HDMI quirk, not an SGLang bug.

## Quick Start

```bash
./scripts/setup.sh                          # venv, SGLang clone, MLX deps, apply 17 patches

# Production presets (all in the probe-trio sweep; see Recommended picks for highlights)
./scripts/launch.sh coder-30b               # MoE — codegen STRONG 8/8, 68 tok/s peak
./scripts/launch.sh devstral                # Dense+VLM — probe_vision STRONG
./scripts/launch.sh gemma4                  # MoE 26B — codegen STRONG + thinking VERIFIED (4K preset)
./scripts/launch.sh gemma4-31b              # Dense (sliding+full) — 4K preset, MMLU 92
./scripts/launch.sh qwen35                  # DeltaNet hybrid+VL (32K preset)
./scripts/launch.sh qwen35-9b-8bit          # Tight-memory variant — 10 GB resident, probe_vision STRONG
./scripts/launch.sh qwen3-moe               # Qwen3-30B MoE (DWQ, MMLU 91)
./scripts/launch.sh qwen3-32b               # Dense (DWQ, clean audit, no DeltaNet/MoE)
./scripts/launch.sh qwen36                  # Qwen3.6-35B-A3B — peak long-ctx throughput, 148 tok/s @ MR=2
./scripts/launch.sh qwen36-27b              # Qwen3.6-27B Dense+DeltaNet+VL
./scripts/launch.sh nemotron-30b            # NemotronH (Mamba2+Attn+MoE) — codegen STRONG 8/8

# Not in production rotation:
# ./scripts/launch.sh coder-next            # Coder-Next-80B — infeasible on M4 (see Known Issues)
# ./scripts/launch.sh smol-docling          # 256M VLM smoke test for the multimodal bridge

# Long-context (128K) — qwen36 validated, prefill ~6.5 min, decode ~0.10 tok/s
CTX=140000 EXTRA_ARGS="--disable-radix-cache --kv-cache-dtype turboquant \
    --chunked-prefill-size 2048 --mem-fraction-static 0.5" \
    bash scripts/launch.sh qwen36

# Agentic coding recipe (VERIFIED 2026-05-18: first M4 SWE-bench Lite patch)
#
# The Qwen3 family's <think>...</think> chat-template blocks break the
# opencode agent loop (model reasons into reasoning_content where opencode
# can't see it, OR </think> tags leak into content and confuse the tool-call
# parser). Workaround: run requests through evals/swebench/no_thinking_proxy.py
# which injects chat_template_kwargs={"enable_thinking": false} server-side.
#
# Primary pick for SWE-bench-style coding: qwen36 (Qwen3.6-35B-A3B
# MoE+DeltaNet). With the proxy, produced a real 506-byte patch on
# astropy__astropy-12907 — the first M4 success.
CTX=32768 EXTRA_ARGS="--disable-radix-cache --kv-cache-dtype turboquant \
    --chunked-prefill-size 2048 --mem-fraction-static 0.5 --enable-multimodal \
    --tool-call-parser qwen3_coder" bash scripts/launch.sh qwen36 &
python evals/swebench/no_thinking_proxy.py &     # opencode → 23335 → SGLang :23334
# (Edit ~/.config/opencode/opencode.jsonc to point baseURL at :23335.)
# Or use the wrapped smoke script which orchestrates all of this:
bash evals/swebench/smoke.sh   # defaults: qwen36, 1 instance, proxy on

# coder-30b alternative — fast single-user decode, BUT under greedy MLX +
# opencode the agent loop gives up after 1 glob call (2026-05-18 finding).
# Use coder-30b for direct chat-completion code generation (HumanEval 95);
# use qwen36 for tool-call-driven agentic flows.

# Capability gates (run AFTER server is up on PORT 23334)
python scripts/eval/validate_capabilities.py --port 23334   # basic + thinking gate (loose keyword grep)
python scripts/eval/probe_thinking.py        --port 23334   # content-aware reasoning probe
python scripts/eval/probe_vision.py          --port 23334   # content-aware image probe (STRONG/DEGRADED/FAIL)
python scripts/eval/probe_codegen.py         --port 23334   # 2-task / 8-test code-synthesis probe
bash   scripts/eval/probe_all.sh                            # sweep probe trio across all presets
PRESETS="nemotron-30b qwen3-32b" bash scripts/eval/probe_all.sh   # single-preset / subset sweep

# Pre-launch checkpoint audits (no server needed)
python scripts/eval/check_mlx_quant_scales.py   <repo-or-path>    # per-layer scale corruption
python scripts/eval/audit_mlx_quant_metadata.py                    # recipe-level hazards across the M4 set
python scripts/eval/validate_chat_template.py --model <path>       # static jinja template check

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

| Preset | Checkpoint (mlx-community) | Type | Wts | 1-user tok/s | Max ctx | Audit hazards |
|--------|---------------------------|------|:---:|:------------:|:-------:|:-------------|
| `coder-30b` | `Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ` | MoE (3B active) | 16 GB | 68.4 | **256K** (3.2) | router INT4 |
| `qwen3-moe` | `Qwen3-30B-A3B-4bit-DWQ` | MoE (3B active) | 16 GB | 69.0 | **64K** (6.3) | router INT4 |
| `qwen36` | `Qwen3.6-35B-A3B-4bit` | MoE+DeltaNet+VL | 17 GB | 51.8 (148 MR=2) | **256K** (0.1) | router INT4 + DeltaNet INT4 |
| `gemma4` | `gemma-4-26b-a4b-it-4bit` | MoE (4B active) | 15 GB | 58.8 | **256K** (1.5) | `embed_vision.embedding_projection` INT4 |
| `qwen35` | `Qwen3.5-27B-4bit` | DeltaNet hybrid+VL | 15 GB | 14.3 (34 MR=2) | 256K (decode timeout) | DeltaNet `in_proj_a/b` INT4 |
| `qwen35-9b-8bit` | `Qwen3.5-9B-MLX-8bit` | DeltaNet hybrid+VL | 10 GB | — | 32K | DeltaNet INT4 (8-bit) |
| `qwen36-27b` | `Qwen3.6-27B-4bit` | DeltaNet hybrid+VL | 14 GB | — (34 MR=2) | 256K | DeltaNet INT4 |
| `devstral` | `Devstral-Small-2-24B-Instruct-2512-4bit` | Dense+VL (Mistral3) | 14 GB | 17.0 (40 MR=4) | **256K** (1.8) | **clean** |
| `qwen3-32b` | `Qwen3-32B-4bit-DWQ` | Dense | 18 GB | 12.1 | 16K (bench timeout) | **clean** |
| `gemma4-31b` | `gemma-4-31b-it-mxfp4` | Dense (sliding+full) | 17 GB | 11.7 (16K) | 16K | `embed_vision.embedding_projection` INT4 |
| `nemotron-30b` | `NVIDIA-Nemotron-3-Nano-30B-A3B-4bit` | NemotronH (Mamba2+Attn+MoE) | 17 GB | — | 32K probe | **clean** |

All checkpoints from [`mlx-community/`](https://huggingface.co/mlx-community). MR=N numbers are batched-decode peaks from patch 011 (2026-05-12). Audit hazards from [the metadata sweep](#calibration-metadata-audit-10-latent-recipe-issues-across-the-model-set-2026-05-13) — `clean` means vision tower / MoE router / DeltaNet `in_proj_a/b` recipe ignores are correct; hazard names are the specific module class quantized when it shouldn't be. Coder-Next-80B is **not** in the active table — see Known Issues.

DWQ variants in 4 presets (`coder-30b`, `qwen3-moe`, `qwen3-32b`, plus `gemma4-31b`'s mxfp4) replaced the standard 4bit uploads after the [DWQ measurement sweep](#quantization-scan-10-dead-layers-in-coder-30b-mlx-community-upload-2026-05-11) — broken-layer fix + MMLU/HE lifts. `qwen36`'s 4bit-DWQ was **not** swapped (-5.5 pp MMLU).

### Multimodal capability matrix (2026-05-17, end-to-end probe-verified)

| Preset | Image | Video | Status |
|--------|:-----:|:-----:|:-------|
| `devstral` (Mistral-Small-3.1) | ✅ | ⚠️ | probe_vision STRONG (single-image). Video FAIL — model echoes prompt under greedy MLX decode on 6-frame input (model-side, not plumbing). |
| `qwen35` (Qwen3.5-27B DeltaNet+VL) | ✅ | ⚠️ | probe_vision STRONG, probe_video DEGRADED (recognizes motion, picks wrong direction). |
| `qwen35-9b-8bit` (Qwen3.5-9B) | ✅ | ✅ | probe_vision + probe_video STRONG. |
| `qwen36` (Qwen3.6-35B-A3B+DeltaNet+VL) | ✅ | ✅ | probe_vision + probe_video STRONG (both directions). |
| `qwen36-27b` (Qwen3.6-27B Dense+DeltaNet+VL) | ✅ | ✅ | probe_vision + probe_video STRONG. |
| `gemma4-31b` (gemma-4-31b-it-mxfp4) | ✅ | ✅ | probe_vision + probe_video STRONG via patches 014 + 018. |
| `gemma4` (gemma-4-26b-a4b-it-4bit) | ✅ | ⚠️ | probe_vision STRONG, probe_video PARTIAL (1/2 directions — quality at the smaller MoE variant; plumbing works). |
| `nemotron-omni` (NemotronH_Omni 30B-A3B) | ✅ | ✅ | probe_vision + probe_video STRONG via patches 016 + 017. |
| `coder-30b` / `qwen3-moe` / `qwen3-32b` / `nemotron-30b` | ❌ | ❌ | Text-only by architecture. |

Headline: **8/8 VLMs probe_vision STRONG, 6/8 probe_video STRONG.** Per-preset JSON in [`benchmarks/quality/probe-trio/`](benchmarks/quality/probe-trio/).

### Choosing a model

**MoE wins at long context.** Each decode token must (1) read model weights and (2) read the entire KV cache. At short context, weight loading dominates → MoE reads 1.5 GB vs Dense 14 GB (4× faster). At 256K with fp8, the KV read climbs to ~5–10 GB — comparable to dense weights — so MoE keeps the weight component small and the KV penalty proportionally less painful. Coder-30B is the best overall: fastest decode, lowest KV pool usage, highest concurrent throughput.

**DeltaNet hybrids** (Qwen3.5, Coder-Next) alternate standard attention (O(n)) with linear attention (O(1)). Linear layers don't slow with context — architecturally suited for very long context — but the standard layers in the hybrid still pay full O(n).

## Performance

> Mac mini M4 Pro (64 GB), SGLang + MLX, `sglang.bench_serving`.
> **Context sweep**: single user, 64 output tokens, radix cache disabled, FP8 or TurboQuant KV cache.
> **Concurrency sweep**: 256 in / 256 out, 8 K context, scaling concurrent users.

### v0.5.11 short-sweep decode tok/s (2026-05-11, fp16 KV)

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
| nemotron-30b | NVIDIA-Nemotron-3-Nano-30B-A3B-4bit (NemotronH) | — | — | — |

\*Gemma 4 presets ship with `CTX=4096` (tight 64 GB budget) — 16K requests rejected. Raise via `CTX=16384 bash scripts/launch.sh gemma4` for the longer-context numbers. Raw bench logs: `/tmp/perf_<preset>_bench.log`. Nemotron added to launch.sh post-sweep — short-sweep perf TBD.

### v0.5.11 long-context turboquant sweep (refreshed 2026-05-12)

Decode tok/s on the v0.5.11 stack with the long-context-tuned recipe (`--chunked-prefill-size 1024 --mem-fraction-static 0.4 --disable-radix-cache`, single user, 64 output tokens). Bench restart between models, OOM guard active. (The earlier `v0.5.11-longctx-turboquant.png` chart at `benchmarks/quality/` reflects pre-refresh numbers and will be regenerated next sweep.)

| Preset | KV | @128 | @4K | @8K | @16K | @32K |
|--------|----|:----:|:---:|:---:|:----:|:----:|
| coder-30b (Qwen3-Coder-30B MoE) | turboquant | **73.8** | 66.6 | 55.2 | 43.0 | — \* |
| gemma4 (Gemma 4 26B MoE) | turboquant | 58.9 | 55.2 | 52.8 | 49.8 | **44.7** |
| devstral (24B Dense) | turboquant | 17.4 | 17.1 | 16.2 | — \* | — \* |
| qwen35 (Qwen3.5-27B DeltaNet) | fp8 | 14.7 | 14.3 | 14.0 | 13.5 | **12.6** |
| gemma4-31b (Gemma 4 31B Dense) | turboquant | 13.5 | 12.7 | 12.4 | 11.7 | — \* |

\*Cells marked `—` are not measurement gaps from the run but indicate the OOM-guard tripped at that context probe — the static pool plus the per-chunk attention scratch (proportional to context × chunked-prefill) exceeded the activation budget before the prefill completed. Both Gemma 4 26B and Qwen3.5 carried through to 32K; the others bottomed out earlier. Raw JSON in `benchmarks/<slug>/results.json` per model (re-run 2026-05-12 03:34–05:01). `qwen36`, `qwen36-27b`, `qwen35-9b-8bit`, `qwen3-32b`, `qwen3-moe`, `nemotron-30b` not in this sweep — added or characterized post-2026-05-12; long-context turboquant rerun is TBD.

**Two patterns emerge:**

- **MoE shapes the short-context win.** Coder-30B (3B active) opens at 73.8 tok/s @128, falls to 43 by 16K. Gemma 4 26B (4B active) opens lower (58.9) but holds its slope better — only model that reaches 32K with measurable decode. Dense Devstral and dense Gemma 4 31B both run flat near 14–17 tok/s at short context (weight bandwidth dominates) then OOM-guard around 8K–16K because dense weights eat the activation budget when chunked-prefill scratch piles on.
- **DeltaNet keeps decode flat.** Qwen3.5-27B (DeltaNet hybrid + Dense full-attn) starts slow (14.7 @128) but stays nearly flat across the entire sweep — 14.7→12.6 from 128 to 32K. TPOT moves from 68 ms to 79 ms while TTFT scales 0.6 s → 272 s. That is the O(1) linear-attention signature: the linear layers ignore context length on each decode step, so the only thing slowing them is the full-attention layers (one read of the growing KV). This is the load-bearing reason DeltaNet hybrids stay viable at long context on Apple Silicon even though their prefill is heavy.

**Headline: turboquant works.** Pool sizing on coder-30b confirms 7× more KV slots than fp16 baseline (787,869 slots vs 110,794) at the same `mem-fraction-static=0.7`. Validation `2/2 PASS` — output identical to fp16 within tolerance. Decode at short context is within 1% of fp16 (58.3 vs 57.9 on coder-30b @128). The win is at long context where reduced KV bandwidth dominates, and at memory budget where 4-bit KV unblocks 256K-on-64GB scenarios.

### Batched-decode peaks (patch 011, single-server multi-prompt)

Per-preset MR=N peak tok/s, measured 2026-05-12 with the patch-011 batched-decode path on the 8-prompt random bench (recipe in Active work item 1):

| Preset | Single user | Peak @ MR | Notes |
|--------|:-----------:|:---------:|:------|
| `qwen3-moe` (Qwen3-30B-A3B-DWQ) | 69 | **160 @ MR=8** | 16-prompt queue, 16/16 successful, concurrency 15.11 |
| `qwen36` (35B-A3B MoE+DeltaNet) | 52 | **148 @ MR=2** | MoE active-params × batched decode compound |
| `devstral` (24B Dense) | 17 | **40 @ MR=4** | 8/8 successful — wrapper backward-compat for dense |
| `qwen35` / `qwen36-27b` (DeltaNet+attn) | 12–14 | **34 @ MR=2** | DeltaNet batched-state stacking unblocks MR>1 |

`coder-30b`, `qwen3-32b`, `gemma4`, `gemma4-31b`, `qwen35-9b-8bit`, `nemotron-30b` not yet benched at MR>1 — patch 011 path is backward-compatible (wrapper rework verified on Devstral and Qwen3-30B as non-hybrid regression checks), so batched decode should work on these too; sweep TBD.

### Memory budget at 256K (64 GB Mac)

Radix cache pre-allocates the KV pool at startup; on unified memory it competes with Metal compute buffers. Use `--mem-fraction-static 0.7` (default).

| Model | Weights | KV @256K fp8 | Fits? |
|-------|:-------:|:------------:|:------|
| MoE 3B-active (Coder-30B, Qwen3-30B-DWQ) | 16 GB | 12 GB | **fp8** comfortably |
| MoE 4B-active (Gemma 4 26B) | 15 GB | 31 GB | **fp8** (tight, mf=0.5) |
| MoE+DeltaNet (Qwen3.6-35B-A3B) | 17 GB | varies† | **turboquant** for 256K |
| Dense+VL (Devstral, Qwen3.5-27B) | 14–15 GB | 21 GB | **fp8** |
| Dense (Qwen3-32B) | 18 GB | 33 GB | **turboquant** required |
| Dense (Gemma 4 31B, sliding+full) | 17 GB | varies | **4K preset** (16K OOMs) |

† DeltaNet layers don't hold KV (recurrent state instead); MoE only stores KV for full-attention layers. Coder-Next 80B (42 GB weights) doesn't fit on a single 64 GB Mac — see Known Issues.

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
| SGLang | **v0.5.11** (`612785ffd`) + 17 patches |
| MLX | 0.31.1 |
| mlx-lm | 0.31.2 |
| PyTorch | 2.9.1 (MPS) |
| Python | 3.12 |

## Patches

17 patches on top of SGLang `v0.5.11` (commit `612785ffd`). Upstream landed patch 001 (the `kv_cache/` subpackage) — we dropped it. The old in-tree mods 008–015 are now folded into proper patch files (006 / 008 / and inside 004). Patches 010–012 are 2026-05-12 follow-ups (mlx_vlm position-cache reset, hybrid batched decode + Qwen3.5 gated multimodal wrapper, pool-sync hardening); patch 013 (2026-05-13) restores the v0.5.10 VLM image-bearing inference path that was silently lost in the v0.5.11 rebase; patch 014 (2026-05-15) unblocks Gemma 4 image+text serving by bypassing transformers' strict `feature_extractor` + `video_processor` requirement; patches 015–018 (2026-05-16) unblock the full multi-image / video pipeline: 015 concatenates `mm_items[*]` features for multi-image input; 016 fixes the bf16-numpy TypeError swallow that left `nemotron-omni` running text-only on image requests; 017 forces `dynamic_resolution=False` on MLX + per-placeholder rendered-image replacement so `nemotron-omni` multi-image token/feature counts align; 018 reverses SGLang's transformers `siglip2.convert_image_to_patches` patchification so Gemma 4's mlx_vlm vision tower receives the `(B, C, H, W)` raw pixels it expects (`gemma4-31b` probe_video STRONG, both directions). See [patches/README.md](patches/README.md) for full per-patch forensics. All patches apply via `git apply` against a clean v0.5.11. See [patches/RADIX_CACHE_GEMMA4_ROOT_CAUSE.md](patches/RADIX_CACHE_GEMMA4_ROOT_CAUSE.md) for the Gemma 4 heterogeneous-attention analysis.

| # | Patch | What |
|:-:|-------|------|
| 002 | mps-backend-defaults | Disable CUDA graph & piecewise CUDA on MPS, force `torch_native` attention, multimodal off by default. |
| 003 | mlx-skip-quantization-check | Skip SGLang's quantization verify when MLX backend is active. |
| 004 | mlx-lifecycle-and-hybrid-fixes | Lifecycle (clear-on-idle, drop-on-finish) + hybrid-model bookkeeping + **patch 013** hybrid cache via `language_model.make_cache()` (Qwen3.5/3.6 MMLU 16.7%→93%) + **patch 015** keep `RotatingKVCache` native (Gemma 4 sliding) + VLM-detect-first `_load_model` with image-aware shim + RoPE auto-scaling. Hybrid-aware `find_attention_layers` so DeltaNet-first layer orderings don't crash; `_get_attn_config` accepts both `n_kv_heads` (mlx_lm) and `num_key_value_heads` (mlx_vlm). |
| 005 | mlx-attn-wrapper-varargs | Devstral / Ministral3 `attn_scale` positional-arg compat. |
| 006 | mlx-offsetcache-and-make-mask | `OffsetCache.__getitem__`/`__setitem__`/`__len__`/`lengths`/`advance` stubs for hybrid decode + **patch 014** explicit `(N, offset+N)` `make_mask` when `offset>0` (chunked prefill). |
| 007 | mlx-multimodal-and-mps-shim | `_mps_stub` cuda→cpu redirect, `mm_utils` shm page-rounding (macOS 16 KB pages), `Modality.MULTI_IMAGES` enum member. |
| 008 | mlx-kv-quant-module | New `kv_quant.py` — `KVCacheMode`, `KVQuantizer`, `bytes_per_element`, `parse_kv_cache_mode` (fp8/mxfp8/turboquant/tq/4bit aliases). Wired into `MlxModelRunner.__init__` via `kv_cache_mode` + `context_length` kwargs; ContiguousKVCache + MlxKVPool wire-up TBD. |
| 009 | mlx-nemotron-h-support | NemotronH hybrid (Mamba2 + Attention + MoE) support — `find_attention_layers` skips non-attention mixers, wrapper accepts both naming conventions, RoPE skipped when absent. |
| 010 | mlx-vlm-position-cache-reset | Clear `_position_ids` / `_rope_deltas` on `LanguageModel` at every new-request prefill — unblocks `MAX_RUNNING>1` on mlx_vlm Qwen3.5/3.6 family. |
| 011 | mlx-hybrid-batched-decode-gated-attn | True batched DeltaNet decode + Qwen3.5 gated multimodal attention in `MLXAttentionWrapper` — replaces serial-per-request fallback. |
| 012 | mlx-sync-pool-skip-non-contiguous | `_sync_new_kv_to_pool` filters to `ContiguousKVCache` only — `ArraysCache` / `RotatingKVCache` skipped (different shapes/types). |
| 013 | mlx-vlm-pixel-values | **Restored** the v0.5.10 VLM image-bearing path (Apr-18 commit `f20ee6e`) silently lost in the v0.5.11 rebase: `pixel_values` + `mm_kwargs` threaded `tp_worker → prefill → TextOnlyVLMShim`. Devstral + Qwen3.5-9B-8bit now probe_vision STRONG. |
| 014 | mlx-gemma4-image-only-processor | Bypass transformers' strict `feature_extractor` + `video_processor` type check for upstream/community Gemma 4 checkpoints — replays `Gemma4Processor.__init__` body with image+tokenizer only. |
| 015 | mlx-multi-image-concat | Patch 013 only forwarded `mm_items[0]`; multi-image requests (probe_video sends 6 frames as 6 image_url items) crashed with `tokens: 384, features 64`. Patch 015 iterates all `mm_items`, concatenates features + per-image kwargs along axis 0. |
| 016 | mlx-mm-bfloat16-numpy | SGLang's `nano_nemotron_vl` processor returns `(3, H, W)` bfloat16 tensors; NumPy can't represent bfloat16, so patch 015's `.numpy()` raised `TypeError` and the outer `except` silently dropped `pixel_values=None`, leaving the model running text-only on image requests. Patch 016 catches the TypeError + falls back to `t.to(torch.float32).numpy()`. Fixes `nemotron-omni` vision from FAIL fabrication to STRONG. |
| 017 | mlx-nemotron-omni-video-alignment | Two co-dependent fixes in SGLang's `nano_nemotron_vl.py` for multi-image alignment. (a) Force `dynamic_resolution=False` on the MLX backend (env-gated by `SGLANG_USE_MLX=1`) so the static tile path runs — `tiles × num_image_token` placeholders exactly matches mlx_vlm's vision-tower feature count. (b) Replace each `<image>` placeholder with its own rendered image one at a time — upstream's `replace(..., "".join(rendered_images), 1)` left N-1 stray IMG_CONTEXT tokens, producing 1541 placeholders vs 1536 features on 6-frame video. Verified `nemotron-omni` probe_video STRONG with 3-token "right"/"down" completions. |
| 018 | mlx-gemma4-unpatch | SGLang's transformers `Gemma4ImageProcessor` outputs pre-patched pixel_values `(B, max_patches, patch_size² × C)` via `siglip2.convert_image_to_patches` (`reshape(C, npH, pH, npW, pW).permute(1, 3, 2, 4, 0).reshape(npH·npW, pH·pW·C)`), but mlx_vlm's gemma4 vision tower destructures `B, C, H, W = pixel_values.shape` — incompatible. In `tp_worker`, detect Gemma 4 by `pixel_values.ndim == 3` + `shape[-1] in (768, 1536)` + presence of `image_position_ids` (only Gemma 4 ships these), filter padding (positions = `-1`), sort patches by row-major grid order from position_ids, reverse the permute/reshape to reconstruct raw `(B, C, H, W)` pixels, drop `image_position_ids` from `mm_kwargs` before forwarding. No extra normalization needed (both processors leave pixels in [0, 1]). Verified: `gemma4-31b` (dense mxfp4) probe_vision STRONG + probe_video STRONG (both "right"/"down" directions correct); `gemma4` (26B-A4B MoE) probe_vision STRONG, probe_video PARTIAL (1/2 directions — quality artifact at smaller variant, plumbing works). |

## Repo layout

```
patches/                    # SGLang patches — see patches/README.md
  00*.patch                 #   17 numbered patches (002-018, sans the upstreamed 001)
  REBASE-v0.5.11-NOTES.md   #   v0.5.11 rebase strategy & lineage
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
