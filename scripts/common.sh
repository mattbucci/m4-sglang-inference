#!/bin/bash
# Common configuration for M4 Pro SGLang MLX inference
#
# SGLang with native MLX backend on Apple Silicon.
# No CUDA, no ROCm — pure Metal via MLX.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# --- Python environment ---
# Primary environment is the conda env (miniforge). A VENV_DIR virtualenv
# takes precedence when it exists so bisect arms / scratch stacks can still
# point VENV_DIR at a throwaway env.
CONDA_ENV="${CONDA_ENV:-sglang-0516}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
SGLANG_DIR="${SGLANG_DIR:-$REPO_DIR/components/sglang}"
MODELS_DIR="${MODELS_DIR:-$HOME/AI/models}"
PORT="${PORT:-23334}"
BASE_URL="http://localhost:${PORT}"

activate_venv() {
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
        return
    fi
    local conda_base
    conda_base="$(conda info --base 2>/dev/null)"
    if [ -n "$conda_base" ] && [ -d "$conda_base/envs/$CONDA_ENV" ]; then
        source "$conda_base/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
        return
    fi
    echo "ERROR: no Python environment found."
    echo "Expected conda env '$CONDA_ENV' or a virtualenv at $VENV_DIR."
    echo "Run scripts/setup.sh first."
    exit 1
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

    # torchcodec ships libtorchcodec_core{4..8}.dylib that link against the
    # libavutil from FFmpeg {4..8}. We have brew FFmpeg 7 (libavutil.59);
    # torchcodec finds it only when DYLD_LIBRARY_PATH points at brew's lib
    # dir. Without this, every server boot spams a stack trace per dylib.
    # macOS strips DYLD_* from inherited env in some paths, so set both.
    local FFMPEG_LIB="/opt/homebrew/opt/ffmpeg/lib"
    if [ -d "$FFMPEG_LIB" ]; then
        export DYLD_LIBRARY_PATH="$FFMPEG_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
        export DYLD_FALLBACK_LIBRARY_PATH="$FFMPEG_LIB${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
    fi
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
