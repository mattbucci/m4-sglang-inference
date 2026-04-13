#!/bin/bash
# SGLang + MLX setup for Apple M4 Pro
#
# Installs SGLang from main branch (includes MLX backend) plus MLX dependencies.
#
# Prerequisites:
#   - macOS with Apple Silicon (M1 or later)
#   - Python 3.12+ (system or Homebrew)
#   - Xcode Command Line Tools (xcode-select --install)
#
# Usage:
#   ./scripts/setup.sh
#   ./scripts/setup.sh --skip-env   # Skip venv creation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

SGLANG_REPO="https://github.com/sgl-project/sglang.git"
SGLANG_BRANCH="main"

SKIP_ENV=false
for arg in "$@"; do
    case $arg in
        --skip-env) SKIP_ENV=true ;;
        -h|--help) head -13 "$0" | tail -11; exit 0 ;;
    esac
done

echo "=============================================="
echo "M4 Pro Inference — MLX Backend Setup"
echo "=============================================="
echo "SGLang:  $SGLANG_BRANCH (includes MLX backend)"
echo "Backend: MLX (native Apple Silicon)"
echo "Venv:    $VENV_DIR"
echo "=============================================="

print_system_info
echo ""

# Validate
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "ERROR: This setup requires Apple Silicon (arm64). Got: $(uname -m)"
    exit 1
fi

PYTHON_CMD="${PYTHON_CMD:-python3}"
if ! command -v "$PYTHON_CMD" &>/dev/null; then
    echo "ERROR: $PYTHON_CMD not found. Install Python 3.12+."
    exit 1
fi

PY_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python:  $PY_VERSION ($PYTHON_CMD)"

# -------------------------------------------------------------------
# Step 1: Clone SGLang (main branch with MLX support)
# -------------------------------------------------------------------
echo ""
if [ ! -d "$SGLANG_DIR" ] || [ ! -d "$SGLANG_DIR/.git" ]; then
    echo "[1/3] Cloning SGLang (main branch with MLX backend)..."
    rm -rf "$SGLANG_DIR"
    mkdir -p "$(dirname "$SGLANG_DIR")"
    git clone --depth 1 "$SGLANG_REPO" "$SGLANG_DIR"

    # Apply patches in order (001 excludes test file that may conflict)
    if ls "$REPO_DIR/patches/"*.patch 1>/dev/null 2>&1; then
        cd "$SGLANG_DIR"
        for patch in "$REPO_DIR/patches/"*.patch; do
            pname=$(basename "$patch")
            if [[ "$pname" == 001-* ]]; then
                echo "  Applying $pname (excluding tests)..."
                git apply --exclude='test/*' "$patch" || echo "  WARNING: $pname failed to apply"
            else
                echo "  Applying $pname..."
                git apply "$patch" || echo "  WARNING: $pname failed to apply"
            fi
        done
    else
        echo "  No patches to apply"
    fi
else
    echo "[1/3] Using existing SGLang source at $SGLANG_DIR"
    echo "  Updating to latest main..."
    cd "$SGLANG_DIR" && git pull --ff-only 2>/dev/null || echo "  (pull skipped — shallow clone)"
fi

# -------------------------------------------------------------------
# Step 2: Create virtual environment + install packages
# -------------------------------------------------------------------
if [ "$SKIP_ENV" = false ]; then
    echo ""
    echo "[2/3] Creating virtual environment at $VENV_DIR"

    if [ -d "$VENV_DIR" ]; then
        echo "  Removing existing venv..."
        rm -rf "$VENV_DIR"
    fi
    $PYTHON_CMD -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    echo "  Upgrading pip..."
    pip install --upgrade pip setuptools wheel 2>/dev/null

    echo "  Installing SGLang from source (MPS/MLX backend)..."
    # Use pyproject_other.toml which has the srt_mps extras (includes mlx, mlx-lm, torch)
    cd "$SGLANG_DIR/python"
    cp pyproject_other.toml pyproject.toml
    pip install -e ".[srt_mps]"

    echo "  Installing additional dependencies..."
    pip install openai requests matplotlib
else
    echo "[2/3] Skipping venv creation"
    activate_venv
fi

# -------------------------------------------------------------------
# Step 3: Verify installation
# -------------------------------------------------------------------
echo ""
echo "[3/3] Verifying installation..."

python3 -c "
import sys
print(f'Python {sys.version}')

import mlx.core as mx
print(f'MLX {mx.__version__} — backend: {mx.default_device()}')

import mlx_lm
print(f'mlx-lm installed')

import torch
print(f'PyTorch {torch.__version__}')
mps = torch.backends.mps.is_available()
print(f'MPS available: {mps}')

import sglang
print(f'SGLang {sglang.__version__}')

from sglang.srt.utils.tensor_bridge import use_mlx
print(f'MLX bridge available: {use_mlx()}')

import os
os.environ['SGLANG_USE_MLX'] = '1'
print(f'SGLANG_USE_MLX: {os.environ.get(\"SGLANG_USE_MLX\")}')

print()
print('All components verified!')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next: download a model and launch the server:"
echo ""
echo "  # Option 1: Use a pre-quantized MLX model from HuggingFace"
echo "  # (models download automatically on first use)"
echo "  ./scripts/launch.sh devstral"
echo ""
echo "  # Option 2: Convert a HuggingFace model to MLX 4-bit"
echo "  source .venv/bin/activate"
echo "  python -m mlx_lm.convert --hf-path <model-id> --mlx-path ~/AI/models/<name>-4bit -q --q-bits 4"
echo ""
