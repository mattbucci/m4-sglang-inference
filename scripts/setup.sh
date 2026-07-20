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

# Env-overridable so gate negative-tests and bisect arms can target scratch
# clones/local sources without editing the pin. Defaults are the live stack.
SGLANG_REPO="${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}"
SGLANG_BRANCH="main"
# Pin to a specific tag to ensure patches apply cleanly.
# Update when rebasing patches onto a newer upstream (upstream MLX covers
# hybrid radix cache, attention duck-typing, and gated-attn batched decode
# natively at this pin — see patches/README.md for the current stack).
SGLANG_COMMIT="${SGLANG_COMMIT:-v0.5.15.post1}"
PATCH_DIR="${PATCH_DIR:-$REPO_DIR/patches}"

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
    echo "[1/3] Cloning SGLang at pinned commit ${SGLANG_COMMIT}..."
    rm -rf "$SGLANG_DIR"
    mkdir -p "$(dirname "$SGLANG_DIR")"
    # Shallow single-tag clone — the full history is ~1-2 GB and unnecessary;
    # we only build one pinned tag. Leaves the tree checked out at the tag.
    git clone --depth 1 --branch "$SGLANG_COMMIT" "$SGLANG_REPO" "$SGLANG_DIR"
    cd "$SGLANG_DIR"

    # Apply per-feature patches in numeric order — see patches/README.md.
    # FATAL-abort on any failure: a silently skipped patch serves plausible
    # output with a whole modality missing (the patch-013 fabrication class).
    # Strict git apply only — no fuzzy fallback.
    cd "$SGLANG_DIR"
    shopt -s nullglob 2>/dev/null || true
    applied=0; failed_patches=""
    for p in "$PATCH_DIR"/0[01][0-9]-*.patch; do
        name=$(basename "$p")
        if git apply --check "$p" 2>/dev/null; then
            git apply "$p"
            echo "  ✓ $name"
            applied=$((applied + 1))
        else
            echo "  ✗ $name (failed git apply --check)"
            failed_patches="$failed_patches $name"
        fi
    done
    echo "  patches: $applied applied"
    if [ -n "$failed_patches" ]; then
        echo "  FATAL: patch(es) FAILED:$failed_patches"
        echo "  The tree at $SGLANG_DIR is now partially patched — fix the"
        echo "  chain (or rm the tree to re-clone) before serving anything."
        exit 1
    fi
else
    echo "[1/3] Using existing SGLang source at $SGLANG_DIR"
    echo "  Running patch gates against the existing tree..."
    SGLANG_DIR="$SGLANG_DIR" SGLANG_TAG="$SGLANG_COMMIT" PATCH_DIR="$PATCH_DIR" \
        bash "$REPO_DIR/scripts/test_patch_gates.sh" || {
        echo "  FATAL: existing tree fails the patch gates (see above)."
        exit 1
    }
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

    # mlx-vlm: VLM (image) model loading for the *ForConditionalGeneration
    # presets (Qwen3.5/3.6, Devstral/Mistral3, Gemma 4) that mlx_lm cannot
    # load. Installed with --no-deps: mlx-vlm 0.6.5 requires
    # transformers>=5.14.0 while SGLang v0.5.15.post1 hard-pins
    # transformers==5.12.1 — 5.12.1 works for our model set (verified via
    # mlx_vlm model-only load). Without this, launch.sh
    # {devstral,qwen35,qwen36,gemma4*} cannot load the model. See patch 008
    # (mlx-vlm-hybrid-integration).
    pip install "mlx-vlm==0.6.5" --no-deps

    # swebench: SWE-bench Lite dataset + harness for evals/swebench/run_rollouts.py.
    # Required to load `princeton-nlp/SWE-bench_Lite` for the agentic-coding probe.
    pip install swebench

    # librosa: required by sglang/srt/models/parakeet.py (ParakeetExtractor),
    # which SGLang's nano_nemotron_vl.py processor unconditionally
    # instantiates at startup for the NemotronH_Nano_Omni_Reasoning_V3
    # architecture even when audio inputs aren't used. Without it,
    # `launch.sh nemotron-omni` ImportErrors at processor init.
    pip install librosa

    # Pin torchcodec to 0.8 — pyproject's 0.11.1 ships dylibs that link
    # against libavutil.{56,57,58,60} (FFmpeg 4/5/6/8). On macOS with brew
    # FFmpeg 7 (libavutil.59) every server boot spams a stack trace per
    # mismatched dylib. 0.8 has core for libavutil.59 and works once
    # DYLD_LIBRARY_PATH points at /opt/homebrew/opt/ffmpeg/lib (handled
    # in scripts/common.sh::setup_mlx_env).
    pip install --force-reinstall --no-deps torchcodec==0.8
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

# Full patch-chain gate: pristine replay + byte identity + double-apply
# rejection, then the eager-import smoke over every patch-touched module.
# Every setup.sh run ends with these — a pass here means the tree serving
# tonight is exactly pin + patches and its boot chain imports.
echo ""
echo "Running patch gates + import smoke..."
SGLANG_DIR="$SGLANG_DIR" SGLANG_TAG="$SGLANG_COMMIT" PATCH_DIR="$PATCH_DIR" \
    bash "$REPO_DIR/scripts/test_patch_gates.sh" || exit 1
SGLANG_USE_MLX=1 PATCH_DIR="$PATCH_DIR" \
    python3 "$REPO_DIR/scripts/eval/import_smoke.py" || exit 1

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
