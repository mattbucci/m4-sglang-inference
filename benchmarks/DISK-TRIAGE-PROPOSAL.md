# Disk triage proposal — HF cache

State: 12 GiB free of 926 GiB; `~/.cache/huggingface/hub` holds ~362 GB across
26 repos (full inventory: `hf-cache-sizes.txt` alongside this file). Low disk
blocks the in-house qwen36 quant (~70+ GB needed) and squeezes bisect-arm
builds (~25 GB each). Nothing below is deleted — awaiting Matt's sign-off.

## Tier 1 — safe deletions (no preset uses these): ~147 GB

| Repo | Size | Why safe |
|------|-----:|----------|
| `mlx-community/Qwen3-Coder-Next-4bit` | 42 GB | Documented infeasible on 64 GB (Known Issues); preset commented out. |
| `mlx-community/Qwen3-32B-4bit` | 17 GB | Preset uses the DWQ variant (kept). |
| `mlx-community/gemma-4-31b-4bit` | 17 GB | Base (non-it) model; documented garbage output — preset uses it-mxfp4 (kept). |
| `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` | 16 GB | The 10-dead-layers upload; preset uses the DWQ variant (kept). |
| `mlx-community/Qwen3-30B-A3B-4bit` | 16 GB | Preset uses the DWQ variant (kept). |
| `mlx-community/Qwen3.5-27B-4bit-DWQ` | 14 GB | Cannot load (upload ships no preprocessor_config.json). |
| Devstral stale revision | ~13 GB | `Devstral-Small-2-24B-Instruct-2512-4bit` holds TWO full snapshots (27 GB total); prune the non-current revision. |
| `mlx-community/Devstral-Small-2507-4bit-DWQ` | 12 GB | Superseded 2507 variant; no preset references it. |

## Tier 2 — judgment calls: ~37 GB

| Repo | Size | Consideration |
|------|-----:|---------------|
| `mlx-community/Qwen3.6-35B-A3B-4bit-DWQ` | 19 GB | Measured SKIP for the flagship (−5.5 MMLU); kept only as a documented code-specialist override. Delete if the override is never used. |
| `mlx-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit` | 18 GB | nemotron-omni preset exists but is unswept on the current stack. Keep if omni validation is planned. |

## Keep

Everything a validated or upstream-blocked preset points at: qwen36 4bit,
coder-30b DWQ, qwen3-moe DWQ, qwen3-32b DWQ, qwen35 4bit, qwen35-9b-8bit,
qwen36-27b, devstral (current revision), nemotron-30b, gemma-4-26b-it-4bit +
gemma-4-31b-it-mxfp4 (return when the upstream sliding-window gap closes),
smol-docling + Qwen2-VL-2B (smoke tests, ~1.7 GB), the three tiny
config-only repos (~30 MB).

## Execution (after sign-off)

Tier 1 via `hf cache delete` (or `rm -rf` of the specific
`models--*` dirs / stale snapshot revision). Expected free after Tier 1:
~159 GiB — comfortably clears the quant build and bisect arms.
