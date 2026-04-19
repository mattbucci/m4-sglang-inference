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
#
# KV cache modes (--kv-cache):
#   fp8            MXFP8 quantized (default, ~2x memory savings)
#   turboquant     Affine 4-bit quantized (~3.5x savings, large KV heads)
#   fp16           Full precision float16 (no quantization)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# --- Defaults (overridden by model preset, then by CLI flags) ---
MODEL="${MODEL:-}"
TOKENIZER=""
CTX=32768
MAX_RUNNING=8
CHUNKED=4096
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
            MODEL="${MODEL:-mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit}"
            CTX=32768; MAX_RUNNING=16; CHUNKED=8192
            CHAT_TEMPLATE="--chat-template \$SCRIPT_DIR/devstral_chat_template.jinja"
            EXTRA_ARGS="$EXTRA_ARGS --enable-multimodal"
            WARMUP="--skip-server-warmup"
            ;;
        coder-30b)
            MODEL="${MODEL:-mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit}"
            CTX=32768; MAX_RUNNING=8; CHUNKED=4096
            ;;
        coder-next)
            MODEL="${MODEL:-mlx-community/Qwen3-Coder-Next-4bit}"
            CTX=8192; MAX_RUNNING=1; CHUNKED=4096
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            ;;
        gemma4)
            # Gemma4ForConditionalGeneration — architecturally vision-capable
            # but the mlx-community 4bit checkpoint is missing
            # preprocessor_config.json, so SGLang can't load the image
            # processor. Text-only for now; vision needs a re-upload from
            # the original Gemma weights with the preprocessor.
            MODEL="${MODEL:-mlx-community/gemma-4-26b-a4b-it-4bit}"
            CTX=4096; MAX_RUNNING=4; CHUNKED=2048
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            ;;
        gemma4-31b)
            # Same situation as gemma4 — text-only on this checkpoint.
            MODEL="${MODEL:-mlx-community/gemma-4-31b-4bit}"
            CTX=4096; MAX_RUNNING=4; CHUNKED=2048
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            ;;
        qwen35)
            # Qwen3_5ForConditionalGeneration — DeltaNet hybrid + vision.
            MODEL="${MODEL:-mlx-community/Qwen3.5-27B-4bit}"
            # MAX_RUNNING=1: DeltaNet batched decode crashes on cache shape
            # mismatch (see project_qwen35_deltanet_decode_crash). Serial
            # decode path works.
            CTX=32768; MAX_RUNNING=1; CHUNKED=8192
            REASONING="--reasoning-parser qwen3"
            EXTRA_ARGS="$EXTRA_ARGS --enable-multimodal"
            WARMUP="--skip-server-warmup"
            ;;
        qwen3-32b)
            MODEL="${MODEL:-mlx-community/Qwen3-32B-4bit}"
            CTX=32768; MAX_RUNNING=4; CHUNKED=8192
            REASONING="--reasoning-parser qwen3"
            ;;
        qwen3-moe)
            MODEL="${MODEL:-mlx-community/Qwen3-30B-A3B-4bit}"
            CTX=32768; MAX_RUNNING=8; CHUNKED=4096
            REASONING="--reasoning-parser qwen3"
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
            # this as their flagship 256K agentic model. Vision-capable.
            MODEL="${MODEL:-mlx-community/Qwen3.6-35B-A3B-4bit}"
            CTX=32768; MAX_RUNNING=1; CHUNKED=4096
            REASONING="--reasoning-parser qwen3"
            WARMUP="--skip-server-warmup"
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

# Apply preset first, then CLI overrides
apply_preset "$PRESET"
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
# Radix cache pre-allocates the full KV pool at startup. On unified
# memory (Apple Silicon), this competes with Metal compute buffers.
# Use 0.7 to leave ~30% headroom for attention intermediates.
# Override with MEM_FRAC env var or --max-total-tokens for exact control.
MEM_FRAC="${MEM_FRAC:-0.7}"

# --- Build command ---
CMD=(python3 -m sglang.launch_server
    --model-path "$MODEL"
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
