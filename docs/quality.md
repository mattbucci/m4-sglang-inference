# Quality: evals, checkpoint audits, fidelity

## Quality table (100-sample MMLU + 20 HE + LAB-Bench + Needle@{1K,4K,16K})

Qwen3 family runs `--no-thinking`; Gemma 4 runs `--humaneval-mode chat`. The Qwen3.8-27B row
and the Qwopus3.8-27B-Coder-Flash row are from the current stack
(`benchmarks/quality/Qwen38-27B-community-2026-10-31.json`, full 333-sample LAB-Bench;
`Qwopus38-27B-Coder-Flash-2026-11-02.json`, 50 per LAB-Bench category); the other rows predate it (re-run tracked in the issue tracker).

| Model | MMLU | HumanEval | LAB-Bench | Needle |
|:------|:----:|:---------:|:---------:|:------:|
| Gemma 4 31B-it-mxfp4 | **92%** | 50%‡ | partial | **100%** |
| Qwen3.5-27B-4bit | **90%** | **100%** | 53/125 (clean cats) | **100%** |
| Qwen3-32B-4bit-DWQ | **90%** | 95% | 33.1% | 100% |
| Qwen3.6-27B-4bit | 86% | **100%** | 42/125 | **100%** |
| Qwen3.8-27B-4bit | 84% | 90% | 38.7% | **100%** |
| Qwopus3.8-27B-Coder-Flash MLX-4bit (in-house, published) | 83% | 90% | 37.5% | **100%** |
| Qwen3-30B-A3B-4bit-DWQ | 85% | 70% | 31.4% | 100% |
| Gemma 4 26B-A4B-it-4bit | 85% | 60%‡ | partial | 100% |
| Coder-30B-A3B-4bit-DWQ | 84% | 95% | 30.9% | 100% |
| Qwen3.5-9B-MLX-8bit | 81% | 80% | 52/150 (clean cats) | **100%** |
| Qwen3.6-35B-A3B-4bit | 80% | 85% | 60/175 | 100% |
| Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | 82% | 65% | 8.6% | 100% |
| NVIDIA Nemotron-3-Nano-30B-A3B-4bit | 77% | 10%¶ | 19.4%¶ | 100% |
| Devstral-24B-4bit | 71% | 55% | 34.3% | 100% |

‡ Chat-mode HumanEval, not comparable to base-completion HE. "N/M (clean cats)" and
"partial" are LAB-Bench subsets where the server died mid-run (full LAB-Bench needs a restart
between categories). ¶ Pre-reasoning-parser numbers; the `nemotron_3` parser is wired into
the preset. Static HumanEval/MMLU does not predict agentic-coding ability on this stack.

## In-house 4-bit builds

`scripts/quantize/convert_qwen36_inhouse.py` builds Qwen3.6-35B-A3B and Qwen3.8-27B
(`--hf-path Qwen/Qwen3.8-27B`) with the hazard classes kept BF16: MoE router, shared-expert
gate, DeltaNet `in_proj_a/b` and the vision tower. Both are quality-equivalent to the
community checkpoints: paired KL to the 8-bit reference +0.0004 ± 0.0006 and the same greedy
token at 1,506/1,524 positions on the 3.8 build; MMLU 83 vs 85, HumanEval 17/20 both,
SWE-bench smoke 3/3 byte-identical, decode within 2.4% on the 3.6 build
(`benchmarks/quality/fidelity/`). The same converter builds `Jackrong/Qwopus3.8-27B-Coder-Flash` (a Qwen3.8-27B coding fine-tune
with no other MLX release), published as `mattbucci/Qwopus3.8-27B-Coder-Flash-MLX-4bit`
(recipe `benchmarks/quality/qwopus38-coder-flash-quant-recipe-2026-11-02.json`, scan clean,
validator 3/3, probe trio STRONG, sealed SWE-bench smoke 3/3). The two Qwen builds stay opt-in local builds:
`MODEL=~/AI/models/Qwen3.6-35B-A3B-4bit-inhouse scripts/launch.sh qwen36` and
`MODEL=~/AI/models/Qwen3.8-27B-4bit-inhouse scripts/launch.sh qwen36-27b`.

## Checkpoint audits

Run both gates on every new checkpoint before adding numbers to the README:

```bash
python scripts/eval/check_mlx_quant_scales.py <repo>    # per-layer corruption (dead dequant output)
python scripts/eval/audit_mlx_quant_metadata.py         # recipe hazards (wrong module classes quantized), header-only
python scripts/eval/validate_chat_template.py --model <path>
```

| Checkpoint | Recipe-level hazard |
|------------|---------------------|
| `Qwen3-Coder-30B-A3B-Instruct-4bit` (non-DWQ) | **10 dead layers** (`layers.{36,46}` attention + gate all-zero); the preset uses the DWQ variant. |
| `gemma-4-26b-a4b-it-4bit` / `gemma-4-31b-it-mxfp4` | `embed_vision.embedding_projection` quantized, the layer class the sister teams' calibration disaster zero-scaled. |
| `Qwen3.5-27B-4bit` / `Qwen3.5-9B-MLX-8bit` / `Qwen3.6-27B-4bit` / `Qwen3.8-27B-4bit` | DeltaNet `linear_attn.in_proj_a/b` quantized on every layer (sister-team rules keep these BF16); all pass their gates regardless. |
| `Qwen3.6-35B-A3B-4bit` | DeltaNet `in_proj_a/b` INT4 (60 layers); MoE router and shared-expert gate 8-bit. |
| `Qwen3-Coder-30B-A3B-4bit-DWQ` / `Qwen3-30B-A3B-4bit-DWQ` | MoE router `mlp.gate` INT4, bounded impact in practice. |
| `Qwen3.8-27B-8bit` | Ships a `processor_config.json` naming `Qwen3VLImageProcessor`, which `AutoProcessor` cannot load; serve from a mirror without that file (fidelity reference only). |
| `Devstral-24B`, `Qwen3-32B-DWQ`, `Nemotron-3-Nano-30B-A3B`, `flint-oQ*e`, in-house builds (incl. `Qwopus3.8-27B-Coder-Flash-MLX-4bit`) | **clean** (Flint: router, sinks and norms bf16; 180 shared-expert projections use group size 128, honoured per tensor). |

DWQ variants are used where they measured better (`coder-30b` +2.8 MMLU / +20 HE over the
broken 4-bit, `qwen3-moe` +7.9 MMLU / −5 HE, `qwen3-32b` +2.8 / +7.5); `qwen36` stays on
standard 4-bit (its DWQ trades −5.5 MMLU for +15 HE; `MODEL=mlx-community/Qwen3.6-35B-A3B-4bit-DWQ`
overrides). **Never blind-swap DWQ: recipes vary per upload; measure MMLU + HumanEval per model.**

Chat-template hazards the validator catches: Devstral leading BOS produces `<unk>` (fixed by
`scripts/devstral_chat_template.jinja`); Qwen3.5 thinking tags without calibrated thinking
data loop on `<think>`; Gemma 4 thinking needs `chat_template_kwargs={"enable_thinking": true}`;
Qwen3.8-27B, Qwen3.6-27B and Qwen3.5-9B-8bit templates ignore `enable_thinking` (always on).

## DiffusionGemma structured reads (CUDA/ROCm hosts)

SGLang serves `google/diffusiongemma-26B-A4B-it` with `--dllm-algorithm Gemma4Renoise` but
rejects logprobs and structured output on it. Patch 025 adds the vLLM-style *structured
read*: the answer template is written into the canvas (`seed_canvas`), template positions
are `pinned`, `max_steps` denoising steps run, and a `read_only` request returns per-position
logprobs for `token_ids_logprob`; `constrained` limits a slot to its labels.
`scripts/eval/structured_reads.py` builds the read from a question and one-token labels. The
model is CUDA/ROCm-only in SGLang and Gemma 4 is blocked on MLX, so this box carries the
patch and its CPU test; the end-to-end run belongs to the R9700 (TP=2 holds the 51.6 GB BF16
checkpoint).
