#!/bin/bash
# Common configuration for M4 Pro SGLang MLX inference
#
# SGLang with native MLX backend on Apple Silicon.
# No CUDA, no ROCm — pure Metal via MLX.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# --- Python environment ---
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
SGLANG_DIR="${SGLANG_DIR:-$REPO_DIR/components/sglang}"
MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"
PORT="${PORT:-23334}"
BASE_URL="http://localhost:${PORT}"

activate_venv() {
    if [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate"
    else
        echo "ERROR: Virtual environment not found at $VENV_DIR"
        echo "Run scripts/setup.sh first."
        exit 1
    fi
}

# MLX environment setup
setup_mlx_env() {
    # Activate MLX backend
    export SGLANG_USE_MLX=1

    # HuggingFace token (for model downloads)
    if [ -f "$HOME/.secrets/hf-token" ]; then
        export HF_TOKEN="$(cat "$HOME/.secrets/hf-token")"
    fi

    # Silence warnings
    export TOKENIZERS_PARALLELISM=false
    export PYTHONWARNINGS="ignore::UserWarning"

    # Metal performance hints
    export MLX_USE_DEFAULT_STREAM=1

    # Long-context support: increase health check timeout.
    # Default 20s is too short — a single prefill chunk at 64K+ context
    # can take 50-90s on Apple Silicon, blocking the scheduler heartbeat.
    export SGLANG_HEALTH_CHECK_TIMEOUT=${SGLANG_HEALTH_CHECK_TIMEOUT:-120}

    # Allow context length override beyond model's default max.
    # Required for 256K context on models with shorter native limits.
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
}

# System info
get_memory_gb() {
    sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1024/1024/1024}'
}

get_chip_name() {
    sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon"
}

print_system_info() {
    echo "Chip:   $(get_chip_name)"
    echo "Memory: $(get_memory_gb) GB unified"
    echo "OS:     $(sw_vers -productName 2>/dev/null) $(sw_vers -productVersion 2>/dev/null)"
}
