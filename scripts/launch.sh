#!/bin/bash
# Unified model launcher for SGLang MLX on Apple M4 Pro
#
# Usage:
#   ./scripts/launch.sh <model> [options]
#   ./scripts/launch.sh devstral
#   ./scripts/launch.sh coder-30b --context-length 16384
#   ./scripts/launch.sh gemma4 --port 8000
#   MODEL=/path/to/weights ./scripts/launch.sh coder-next
#
# Models:
#   devstral       Devstral-24B 4-bit (32K context, best all-round)
#   coder-30b      Qwen3-Coder-30B MoE 4-bit (32K, best throughput)
#   coder-next     Qwen3-Coder-Next-80B MoE 4-bit (8K, largest model)
#   gemma4         Gemma 4 26B MoE 4-bit (4K)
#   qwen35         Qwen3.5-27B DeltaNet 4-bit (32K)

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
EXTRA_ARGS=""

# --- Model presets (tuned for 64GB unified memory) ---
apply_preset() {
    case "$1" in
        devstral)
            MODEL="${MODEL:-mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit}"
            CTX=32768; MAX_RUNNING=16; CHUNKED=8192
            CHAT_TEMPLATE="--chat-template \$SCRIPT_DIR/devstral_chat_template.jinja"
            WARMUP="--skip-server-warmup"
            ;;
        coder-30b)
            MODEL="${MODEL:-mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit}"
            CTX=32768; MAX_RUNNING=8; CHUNKED=4096
            ;;
        coder-next)
            MODEL="${MODEL:-mlx-community/Qwen3-Coder-Next-4bit}"
            CTX=8192; MAX_RUNNING=1; CHUNKED=4096
            WATCHDOG=1800
            ;;
        gemma4)
            MODEL="${MODEL:-mlx-community/gemma-4-26b-a4b-it-4bit}"
            CTX=4096; MAX_RUNNING=4; CHUNKED=2048
            WARMUP="--skip-server-warmup"; WATCHDOG=1800
            ;;
        qwen35)
            MODEL="${MODEL:-mlx-community/Qwen3.5-27B-4bit}"
            CTX=32768; MAX_RUNNING=4; CHUNKED=8192
            CHAT_TEMPLATE="--chat-template \$MODEL/chat_template.jinja"
            REASONING="--reasoning-parser qwen3"
            ;;
        *)
            echo "Unknown model: $1"
            echo "Run with -h for available models."
            exit 1
            ;;
    esac
}

# --- Parse arguments ---
PRESET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            head -16 "$0" | tail -15
            exit 0
            ;;
        --context-length) CTX="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --max-running) MAX_RUNNING="$2"; shift 2 ;;
        --chunked-prefill) CHUNKED="$2"; shift 2 ;;
        --watchdog) WATCHDOG="$2"; shift 2 ;;
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

apply_preset "$PRESET"

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
echo "Context: $CTX  Port: $PORT  Memory: ${MEM_GB}GB"
echo "=============================================="

# --- Build command ---
CMD=(python3 -m sglang.launch_server
    --model-path "$MODEL"
    --context-length "$CTX"
    --max-running-requests "$MAX_RUNNING"
    --chunked-prefill-size "$CHUNKED"
    --trust-remote-code
    --watchdog-timeout "$WATCHDOG"
    --port "$PORT"
    --host 0.0.0.0
    --enable-metrics
)

[[ -n "$TOKENIZER" ]] && CMD+=($TOKENIZER)
[[ -n "$CHAT_TEMPLATE" ]] && CMD+=($CHAT_TEMPLATE)
[[ -n "$REASONING" ]] && CMD+=($REASONING)
[[ -n "$WARMUP" ]] && CMD+=($WARMUP)
[[ -n "$EXTRA_ARGS" ]] && CMD+=($EXTRA_ARGS)

exec "${CMD[@]}"
