#!/bin/bash
# Unified model launcher for SGLang MLX on Apple M4 Pro
#
# Usage:
#   ./scripts/launch.sh <model> [options]
#   ./scripts/launch.sh devstral
#   ./scripts/launch.sh coder-30b --context-length 16384
#   ./scripts/launch.sh devstral --context-length 262144 --kv-cache fp8
#   ./scripts/launch.sh devstral --context-length 262144 --kv-cache turboquant
#   MODEL=/path/to/weights ./scripts/launch.sh coder-next
#
# Models:
#   devstral       Devstral-24B 4-bit (32K context, best all-round)
#   coder-30b      Qwen3-Coder-30B MoE 4-bit (32K, best throughput)
#   coder-next     Qwen3-Coder-Next-80B MoE 4-bit (8K, largest model)
#   gemma4         Gemma 4 26B MoE 4-bit (4K)
#   gemma4-31b     Gemma 4 31B 4-bit (4K)
#   qwen35         Qwen3.5-27B DeltaNet 4-bit (32K)
#   qwen35-9b-8bit Qwen3.5-9B DeltaNet 8-bit (32K, smaller+higher-precision)
#   qwen36         Qwen3.6-35B-A3B-4bit MoE+DeltaNet+VL (sister-team flagship)
#   qwen36-27b     Qwen3.6-27B-4bit Dense DeltaNet+VL (new — no MoE variant)
#
# KV cache modes (--kv-cache):
#   fp8            MXFP8 quantized (default, ~2x memory savings)
#   turboquant     Affine 4-bit quantized (~3.5x savings, large KV heads)
#   fp16           Full precision float16 (no quantization)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Capture genuine env-var overrides BEFORE we apply any script-level
# defaults. Without this split, the line below that does
# `MAX_RUNNING="${MAX_RUNNING:-8}"` materializes the default into the
# shell variable, so the later ENV_MAX_RUNNING capture (~line 223)
# reads "8" and applies it as if the user had set it — silently
# overriding whatever the preset chose. Same bug existed for CTX and
# CHUNKED; capturing here keeps preset values intact when the user
# didn't actually pass an env var.
ENV_CTX="${CTX:-}"
ENV_MAX_RUNNING="${MAX_RUNNING:-}"
ENV_CHUNKED="${CHUNKED:-}"

# --- Defaults (overridden by model preset, then env vars, then CLI flags) ---
MODEL="${MODEL:-}"
TOKENIZER=""
CTX="${CTX:-32768}"
MAX_RUNNING="${MAX_RUNNING:-8}"
CHUNKED="${CHUNKED:-4096}"
CHAT_TEMPLATE=""
REASONING=""
WARMUP=""
WATCHDOG=600
KV_CACHE="fp8"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# --- Model presets (tuned for 64GB unified memory) ---
apply_preset() {
    case "$1" in
        devstral)
            # Mistral3ForConditionalGeneration — vision-capable.
            # --enable-multimodal exposes the image path (patches 007/009/010
            # /011/012 + VLM detection make it actually work).
            # --tool-call-parser mistral matches 3090 sister-team mapping for
            # Mistral-arch models (Devstral emits `[TOOL_CALLS]` tags); without
            # the parser SGLang serves the raw tag as assistant text and any
            # coding harness silently drops the call.
            MODEL="${MODEL:-mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit}"
            CTX=32768; MAX_RUNNING=16; CHUNKED=8192
            CHAT_TEMPLATE="--chat-template \$SCRIPT_DIR/devstral_chat_template.jinja"
            EXTRA_ARGS="$EXTRA_ARGS --enable-multimodal --tool-call-parser mistral"
            WARMUP="--skip-server-warmup"
            ;;
        coder-30b)
            # 4bit-DWQ (Distillation Weight Quantization) is the clean variant
            # — the original 4bit upload has 10 dead layers (model.layers.36/46
            # weight + biases all-zero, attention output collapses through those
            # two layers). check_mlx_quant_scales.py catches it; DWQ is 9/9.
            # --tool-call-parser qwen3_coder matches the 3090 sister-team
            # convention on every Qwen3-Coder preset (XML <tool_call> tag
            # parsing into tool_calls[]); operates on output tokens only so
            # safe under greedy MLX decode.
            MODEL="${MODEL:-mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ}"
            CTX=32768; MAX_RUNNING=8; CHUNKED=4096
            EXTRA_ARGS="$EXTRA_ARGS --tool-call-parser qwen3_coder"
            ;;
        coder-next)
            # Qwen3-Coder-Next is a Qwen3-Next DeltaNet hybrid (~80B). See the
            # "Coder-Next-80B infeasible on M4" memory before retrying — the
            # weights + PyTorch/mlx_vlm/torchcodec import surface exceed 64GB
            # in practice. --disable-radix-cache required like other hybrids.
            # qwen3_coder tool-call parsing matches the family convention.
            MODEL="${MODEL:-mlx-community/Qwen3-Coder-Next-4bit}"
            CTX=8192; MAX_RUNNING=1; CHUNKED=4096
            EXTRA_ARGS="$EXTRA_ARGS --disable-radix-cache --tool-call-parser qwen3_coder"
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            ;;
        gemma4)
            # Gemma4ForConditionalGeneration — architecturally supports
            # IMAGE + VIDEO + AUDIO, but the mlx-community 4bit checkpoint
            # is missing preprocessor_config.json so SGLang can't load any
            # of the multimodal processors. Text-only for now; multimodal
            # needs a re-upload from the original Gemma weights with the
            # preprocessor + audio config.
            # --reasoning-parser gemma4 matches both sister teams' convention:
            # without it, chat_template_kwargs={"enable_thinking": true}
            # output stays as a raw <think>...</think> block in content
            # instead of being split into reasoning_content. Required for
            # downstream probes / evals to see the parsed thinking trace.
            # --tool-call-parser gemma4 matches both sister teams — Gemma 4
            # emits `<|tool>` tags which need the gemma4 parser to surface as
            # tool_calls[] instead of plain assistant text.
            MODEL="${MODEL:-mlx-community/gemma-4-26b-a4b-it-4bit}"
            CTX=4096; MAX_RUNNING=4; CHUNKED=2048
            REASONING="--reasoning-parser gemma4"
            # --disable-radix-cache is REQUIRED — Gemma 4 has heterogeneous
            # per-layer attention shapes (25 sliding @ (8,256) + 5 full @
            # (2,512)). MlxKVPool is sized for ONE shape sampled from layer 0
            # (sliding), but only full-attention layers write to the pool
            # (sliding layers stay on native RotatingKVCache). First
            # full-attention prefill broadcasts (2,128) packed-KV into
            # (1,8,64) pool slots and crashes with ValueError. See
            # patches/RADIX_CACHE_GEMMA4_ROOT_CAUSE.md for the full trace and
            # fix-A/B/C options. Workaround applies to both 26B and 31B.
            # --enable-multimodal: patch 002 forces multimodal=False by default;
            # patch 014 unblocked the processor construction (Gemma4ImageOnlyProcessor
            # bypasses missing preprocessor_config.json / video_preprocessor_config.json)
            # but the SGLang multimodal gate still drops image bytes without this flag.
            # Same pattern as Qwen3.5/3.6 + Devstral.
            EXTRA_ARGS="$EXTRA_ARGS --enable-multimodal --disable-radix-cache --tool-call-parser gemma4"
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            ;;
        gemma4-31b)
            # mlx-community/gemma-4-31b-4bit (no `-it`) is a BASE model:
            # no chat template, no special tokens, no instruction tuning.
            # Chat completions returned garbage start tokens. Switched to
            # the instruction-tuned variant which speaks Gemma turn format.
            # --reasoning-parser gemma4 — see gemma4 preset comment for why.
            # --tool-call-parser gemma4 — see gemma4 preset comment for why.
            MODEL="${MODEL:-mlx-community/gemma-4-31b-it-mxfp4}"
            CTX=4096; MAX_RUNNING=4; CHUNKED=2048
            REASONING="--reasoning-parser gemma4"
            # --disable-radix-cache REQUIRED — same heterogeneous-attention bug
            # as gemma4 (50 sliding @ (16,256) + 10 full @ (4,512)).
            # --enable-multimodal — see gemma4 preset comment.
            EXTRA_ARGS="$EXTRA_ARGS --enable-multimodal --disable-radix-cache --tool-call-parser gemma4"
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            ;;
        qwen35)
            # Qwen3_5ForConditionalGeneration — DeltaNet hybrid + vision + VIDEO.
            # video_grid_thw / second_per_grid_ts already flow through tp_worker
            # mm_extra_kwargs path (see SGLang commit 2a327f0 for upstream fix).
            # Radix cache ENABLED as of v0.5.15.post1 — DeltaNet state is
            # snapshotted via MlxAuxiliaryStatePool (no_buffer strategy), so
            # the old ArraysCache/_sync_new_kv_to_pool crash is gone. Same
            # machinery greedy-determinism-validated on qwen36 2026-07-19.
            MODEL="${MODEL:-mlx-community/Qwen3.5-27B-4bit}"
            # MAX_RUNNING=4: patch 010 reset of mlx_vlm position cache +
            # patch 011 batched DeltaNet decode + Qwen3_5-aware
            # MLXAttentionWrapper unblock concurrent serving on this
            # gated multimodal hybrid (2026-05-12).
            # --tool-call-parser qwen3_coder per 3090 mapping: every Qwen3.5/3.6
            # family member emits `<function=NAME>...</function>` XML tool calls
            # that need the qwen3_coder parser to surface as structured tool_calls.
            CTX=32768; MAX_RUNNING=4; CHUNKED=8192
            REASONING="--reasoning-parser qwen3"
            EXTRA_ARGS="$EXTRA_ARGS --enable-multimodal --tool-call-parser qwen3_coder"
            WARMUP="--skip-server-warmup"
            ;;
        qwen35-9b-8bit)
            # Smaller Qwen3.5 (9B) at higher precision (8-bit). Same DeltaNet
            # hybrid + vision architecture as qwen35; needs patch 013 for
            # correctness. Better quality/memory tradeoff than 27B-4bit for
            # most workloads (~10 GB resident vs ~14 GB). Radix cache ENABLED
            # as of v0.5.15.post1 (same aux-state snapshot path as qwen35).
            MODEL="${MODEL:-mlx-community/Qwen3.5-9B-MLX-8bit}"
            CTX=32768; MAX_RUNNING=1; CHUNKED=8192
            REASONING="--reasoning-parser qwen3"
            EXTRA_ARGS="$EXTRA_ARGS --enable-multimodal --tool-call-parser qwen3_coder"
            WARMUP="--skip-server-warmup"
            ;;
        qwen3-32b)
            # 4bit-DWQ variant: scanner-clean (449/449 layers healthy),
            # MMLU 89.5% / HumanEval 95% vs 86.7% / 87.5% on standard 4bit.
            # Wins on both axes — clean swap.
            # --tool-call-parser qwen25 per 3090 mapping: Qwen3 base/generalist
            # (non-Coder, non-3.5/3.6) emits JSON-in-tag `<tool_call>{json}</tool_call>`
            # which the qwen25 parser handles.
            MODEL="${MODEL:-mlx-community/Qwen3-32B-4bit-DWQ}"
            CTX=32768; MAX_RUNNING=4; CHUNKED=8192
            REASONING="--reasoning-parser qwen3"
            EXTRA_ARGS="$EXTRA_ARGS --tool-call-parser qwen25"
            ;;
        qwen3-moe)
            # 4bit-DWQ variant: scanner-clean, MMLU 91.2% vs 83.3% on standard 4bit
            # (+7.9 pp), HumanEval -5 pp. Net win for general-knowledge agentic work.
            # --tool-call-parser qwen25 — see qwen3-32b for rationale.
            #
            # Concurrent batched decode verified 2026-05-17: MAX_RUNNING=4 +
            # default chunk=4096 + mem-fraction-static=0.7 handles 16 in-flight
            # prompts cleanly. Peak observed throughput ~158 tok/s at
            # 13 concurrent decode on Qwen3-30B-A3B-4bit-DWQ. MAX_RUNNING
            # bumped 8 → 16 since SGLang's cap doesn't actually limit MLX's
            # batched decode path (patch 011); the real ceiling is set by
            # what fits in the activation budget, and 16 fits.
            MODEL="${MODEL:-mlx-community/Qwen3-30B-A3B-4bit-DWQ}"
            CTX=32768; MAX_RUNNING=16; CHUNKED=4096
            REASONING="--reasoning-parser qwen3"
            EXTRA_ARGS="$EXTRA_ARGS --tool-call-parser qwen25"
            ;;
        smol-docling)
            # VLM smoke test: smallest available MLX VLM (256M params).
            # Used for MLX vision investigation — patch 002 normally disables
            # multimodal but `--enable-multimodal` on the CLI overrides
            # (the patch only forces False when enable_multimodal is None).
            MODEL="${MODEL:-ds4sd/SmolDocling-256M-preview-mlx-bf16}"
            CTX=8192; MAX_RUNNING=1; CHUNKED=4096
            EXTRA_ARGS="$EXTRA_ARGS --enable-multimodal --disable-radix-cache"
            WARMUP="--skip-server-warmup"
            ;;
        qwen36)
            # Qwen3.6-35B-A3B MoE+DeltaNet+VL. Sister teams (3090/R9700) use
            # this as their flagship 256K agentic model. Vision + VIDEO capable
            # (qwen_vl.preprocess_video, video_grid_thw / second_per_grid_ts).
            # Radix cache ENABLED as of v0.5.15.post1: the native MLX cache
            # layout snapshots DeltaNet state via MlxAuxiliaryStatePool, so the
            # old "hybrid ArraysCache" _sync_new_kv_to_pool crash is gone.
            # Validated 2026-07-19: greedy outputs identical with/without
            # prefix-cache hit.
            MODEL="${MODEL:-mlx-community/Qwen3.6-35B-A3B-4bit}"
            # --enable-multimodal required: patch 002 forces multimodal=False
            # when unset; without this flag the SGLang multimodal gate drops
            # incoming images before they reach the model. Probe sweep
            # 2026-05-16 showed Qwen3.6 vision FAIL with fabricated reasoning
            # content (e.g. "cartoon character with blue skin and pointy ears"
            # for a red-circle-on-white prompt). Adding --enable-multimodal
            # routes image bytes into the patch 013 pixel_values plumbing —
            # same path that's been STRONG on Qwen3.5-9B-8bit since 2026-05-13.
            CTX=32768; MAX_RUNNING=1; CHUNKED=4096
            REASONING="--reasoning-parser qwen3"
            EXTRA_ARGS="$EXTRA_ARGS --enable-multimodal --tool-call-parser qwen3_coder"
            WARMUP="--skip-server-warmup"
            ;;
        qwen36-27b)
            # Qwen3.6-27B (Dense, no MoE). Pure DeltaNet+VL variant of the
            # Qwen3.6 family — smaller weights than 35B-A3B, no MoE indirection
            # so decode is dense-bound. Same hybrid-cache + VLM-wrapper path
            # as qwen35 / qwen36 (patches 013/015 load-bearing).
            # Radix cache ENABLED as of v0.5.15.post1 (same aux-state
            # snapshot path as qwen36).
            # --enable-multimodal required for vision — see qwen36 preset
            # comment. probe_vision FAIL on 2026-05-16 without it.
            MODEL="${MODEL:-mlx-community/Qwen3.6-27B-4bit}"
            CTX=32768; MAX_RUNNING=1; CHUNKED=8192
            REASONING="--reasoning-parser qwen3"
            EXTRA_ARGS="$EXTRA_ARGS --enable-multimodal --tool-call-parser qwen3_coder"
            WARMUP="--skip-server-warmup"
            ;;
        nemotron-omni)
            # NVIDIA Nemotron-3-Nano-Omni-30B-A3B-Reasoning (NemotronH_Nano_Omni_Reasoning_V3).
            # Vision-capable + reasoning-mode variant of nemotron-30b — potentially
            # replaces it if load + probes pass. Same memory footprint (19.6 GB,
            # 4 safetensors), conversion via mlx-vlm 0.4.5. Has both
            # processor_config.json + preprocessor_config.json so AutoProcessor
            # construction works without the gemma4-style bypass patch.
            #
            # Risk: NemotronH_Nano_Omni_Reasoning_V3 is a newer architecture class
            # than what patch 009 wrapped (Mamba2+Attn+MoE without vision tower).
            # May need additional wrapping for the vision side.
            MODEL="${MODEL:-mlx-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit}"
            CTX=32768; MAX_RUNNING=1; CHUNKED=4096
            REASONING="--reasoning-parser nemotron_3"
            EXTRA_ARGS="$EXTRA_ARGS --enable-multimodal --disable-radix-cache"
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            ;;
        nemotron-30b)
            # NVIDIA Nemotron-3-Nano-30B-A3B (NemotronH hybrid: Mamba2 +
            # Attention + MoE). 30B total, 3B active (128 experts top-6,
            # 1 shared), 52 layers, 262K native context. No RoPE — position
            # info comes from interleaved Mamba layers.
            #
            # Block pattern "MEMEM*EMEMEM*..." → ArraysCache for M (Mamba)
            # layers, KVCache for * (attention), nothing for E (MoE) / -
            # (MLP). _detect_hybrid catches it via ArraysCache; patch 013
            # routes make_cache() through language_model. Smoke-test at
            # MAX_RUNNING=1 like the other hybrids.
            MODEL="${MODEL:-mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit}"
            CTX=32768; MAX_RUNNING=1; CHUNKED=4096
            # Radix cache ENABLED as of v0.5.15.post1: Mamba2 state rides
            # the same MlxAuxiliaryStatePool snapshot path as DeltaNet.
            # --reasoning-parser nemotron_3 — Nemotron-3-Nano emits verbose
            # thinking traces; without the parser they consume the 1024-tok
            # MC eval budget before the model answers. Initial M4 quality
            # eval landed at MMLU 77 / HE 10 / LAB-Bench 19.4 specifically
            # because of this (README quality table footnote ¶). Wiring
            # nemotron_3 should bump HE + LAB-Bench substantially.
            EXTRA_ARGS="$EXTRA_ARGS"
            REASONING="--reasoning-parser nemotron_3"
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            ;;
        *)
            echo "Unknown model: $1"
            echo "Run with -h for available models."
            exit 1
            ;;
    esac
}

# --- Parse arguments (save CLI overrides to apply after preset) ---
PRESET=""
CLI_CTX="" CLI_PORT="" CLI_MAX_RUNNING="" CLI_CHUNKED="" CLI_WATCHDOG="" CLI_KV_CACHE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            head -23 "$0" | tail -22
            exit 0
            ;;
        --context-length) CLI_CTX="$2"; shift 2 ;;
        --port) CLI_PORT="$2"; shift 2 ;;
        --max-running) CLI_MAX_RUNNING="$2"; shift 2 ;;
        --chunked-prefill) CLI_CHUNKED="$2"; shift 2 ;;
        --watchdog) CLI_WATCHDOG="$2"; shift 2 ;;
        --kv-cache) CLI_KV_CACHE="$2"; shift 2 ;;
        -*)
            echo "Unknown option: $1"; exit 1 ;;
        *)
            if [[ -z "$PRESET" ]]; then
                PRESET="$1"; shift
            else
                echo "Unexpected argument: $1"; exit 1
            fi
            ;;
    esac
done

if [[ -z "$PRESET" ]]; then
    echo "Usage: $0 <model> [options]"
    echo "Run with -h for available models."
    exit 1
fi

# Env-var overrides take precedence over the preset (so callers can do
# `CTX=80000 MAX_RUNNING=4 ./scripts/launch.sh coder-30b` without editing
# the preset). CLI flags still beat env vars below. The ENV_* values
# were captured at script entry, BEFORE the script-level defaults were
# materialised into the shell — see the top-of-file note for why.
apply_preset "$PRESET"
# REASONING_OFF=1 strips --reasoning-parser, useful for agent harnesses
# (opencode) that don't surface reasoning_content into the agent loop.
[[ "${REASONING_OFF:-}" = "1" ]] && REASONING=""
[[ -n "$ENV_CTX" ]] && CTX="$ENV_CTX"
[[ -n "$ENV_MAX_RUNNING" ]] && MAX_RUNNING="$ENV_MAX_RUNNING"
[[ -n "$ENV_CHUNKED" ]] && CHUNKED="$ENV_CHUNKED"
[[ -n "$CLI_CTX" ]] && CTX="$CLI_CTX"
[[ -n "$CLI_PORT" ]] && PORT="$CLI_PORT"
[[ -n "$CLI_MAX_RUNNING" ]] && MAX_RUNNING="$CLI_MAX_RUNNING"
[[ -n "$CLI_CHUNKED" ]] && CHUNKED="$CLI_CHUNKED"
[[ -n "$CLI_WATCHDOG" ]] && WATCHDOG="$CLI_WATCHDOG"
[[ -n "$CLI_KV_CACHE" ]] && KV_CACHE="$CLI_KV_CACHE"

# Resolve chat template (deferred $MODEL / $SCRIPT_DIR expansion)
CHAT_TEMPLATE=$(eval echo "$CHAT_TEMPLATE")

# --- Setup environment ---
activate_venv
setup_mlx_env

# Check memory pressure before launching large models
MEM_GB=$(get_memory_gb)
echo "=============================================="
echo "$PRESET — SGLang MLX on M4 Pro"
echo "Python  $(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "MLX     $(python3 -c 'import mlx.core as mx; print(mx.__version__)')"
echo "SGLang  $(python3 -c 'import sglang; print(sglang.__version__)')"
echo "Model:  $MODEL"
echo "Context: $CTX  Port: $PORT  Memory: ${MEM_GB}GB  KV: $KV_CACHE"
echo "=============================================="

# --- Memory fraction ---
# Radix cache pre-allocates the full KV pool at startup. On Apple Silicon
# UNIFIED memory, `--mem-fraction-static` caps the fraction of the WHOLE
# pool MLX takes — there is no separate VRAM. 0.7 → ~45 GB to MLX,
# ~19 GB for kernel + Metal compile buffers + page cache + activation
# scratch + everything else. Pushing this to 0.85 was tested 2026-05-14
# and crashed the box (compressor + swap effective usage hit ~150 GB
# before macOS jetsam reaped). DO NOT raise without per-preset
# validation. Override with MEM_FRAC env var or --max-total-tokens for
# exact control.
MEM_FRAC="${MEM_FRAC:-0.7}"

# --- Build command ---
# Serve the model under a clean preset name (e.g. `coder-30b`) instead of
# the full HuggingFace path. Clients (opencode, etc.) can refer to the
# model as `sglang/coder-30b` rather than
# `sglang/mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ`. Override
# via `SERVED_NAME=<custom>` if a specific name is needed.
SERVED_NAME="${SERVED_NAME:-$PRESET}"

CMD=(python3 -m sglang.launch_server
    --model-path "$MODEL"
    --served-model-name "$SERVED_NAME"
    --context-length "$CTX"
    --max-running-requests "$MAX_RUNNING"
    --chunked-prefill-size "$CHUNKED"
    --mem-fraction-static "$MEM_FRAC"
    --trust-remote-code
    --watchdog-timeout "$WATCHDOG"
    --port "$PORT"
    --host 0.0.0.0
    --enable-metrics
    --kv-cache-dtype "$KV_CACHE"
)

[[ -n "$TOKENIZER" ]] && CMD+=($TOKENIZER)
[[ -n "$CHAT_TEMPLATE" ]] && CMD+=($CHAT_TEMPLATE)
[[ -n "$REASONING" ]] && CMD+=($REASONING)
[[ -n "$WARMUP" ]] && CMD+=($WARMUP)
[[ -n "$EXTRA_ARGS" ]] && CMD+=($EXTRA_ARGS)

exec "${CMD[@]}"
