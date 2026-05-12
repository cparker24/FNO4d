#!/usr/bin/env bash
# install.sh — FNO4d environment setup
#
# Detects platform (Mac arm64 / Linux x86_64 / Linux aarch64) and environment
# mode (active conda env vs fresh virtualenv), then installs all required
# packages and local editable packages.
#
# Usage:
#   bash install.sh            # from the repo root
#
# In an active conda env the script installs directly into it (no venv created).
# Otherwise a virtualenv named "venv" is created in the repo root.

set -euo pipefail

# ── helpers ──────────────────────────────────────────────────────────────────
log()  { echo "[install.sh] $*"; }
warn() { echo "[install.sh] WARNING: $*" >&2; }
die()  { echo "[install.sh] ERROR: $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── detect platform ───────────────────────────────────────────────────────────
OS="$(uname -s)"    # Darwin | Linux
ARCH="$(uname -m)"  # arm64 | x86_64 | aarch64

log "Platform: OS=$OS  ARCH=$ARCH"

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    PLATFORM="mac"
elif [[ "$OS" == "Linux" ]]; then
    PLATFORM="linux"
else
    die "Unsupported platform: OS=$OS ARCH=$ARCH"
fi

# ── detect / set up environment ───────────────────────────────────────────────
USING_CONDA=false
VENV_CREATED=false

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    log "Active conda environment detected: $CONDA_PREFIX"
    log "Installing into current conda env — no virtualenv will be created."
    USING_CONDA=true
    PIP="pip"
else
    VENV_DIR="$REPO_ROOT/venv"
    if [[ -d "$VENV_DIR" ]]; then
        log "Existing virtualenv found at $VENV_DIR — reusing it."
    else
        log "Creating virtualenv at $VENV_DIR ..."
        if [[ "$PLATFORM" == "linux" ]]; then
            python3 -m venv "$VENV_DIR" --system-site-packages
        else
            python3 -m venv "$VENV_DIR"
        fi
        VENV_CREATED=true
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PIP="$VENV_DIR/bin/pip"
    log "Virtualenv activated: $VENV_DIR"
fi

$PIP install --upgrade pip --quiet

# ── Linux: torch detection and CUDA-aware install ────────────────────────────
install_torch_linux() {
    if python3 -c "import torch" 2>/dev/null; then
        log "torch already importable — skipping torch install."
        return
    fi

    warn "torch not found. Detecting CUDA version ..."

    CUDA_MAJOR=""
    CUDA_MINOR=""

    if command -v nvidia-smi &>/dev/null; then
        # nvidia-smi output: "CUDA Version: 12.4" (or similar)
        CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || true)
        if [[ -n "$CUDA_VER" ]]; then
            CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
            CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
            log "Detected CUDA $CUDA_MAJOR.$CUDA_MINOR"
        else
            warn "nvidia-smi found but could not parse CUDA version."
        fi
    else
        warn "nvidia-smi not found."
    fi

    # Map CUDA version to wheel tag
    WHEEL_TAG=""
    if [[ -z "$CUDA_MAJOR" ]]; then
        warn "No GPU detected — installing CPU-only torch. Training will be slow!"
        WHEEL_TAG="cpu"
    elif [[ "$CUDA_MAJOR" -eq 11 && "$CUDA_MINOR" -ge 8 ]]; then
        WHEEL_TAG="cu118"
    elif [[ "$CUDA_MAJOR" -eq 12 ]]; then
        if   [[ "$CUDA_MINOR" -le 3 ]]; then WHEEL_TAG="cu121"
        elif [[ "$CUDA_MINOR" -le 7 ]]; then WHEEL_TAG="cu124"
        else                                  WHEEL_TAG="cu128"
        fi
    elif [[ "$CUDA_MAJOR" -ge 13 ]]; then
        # Try the exact cu<MAJOR><MINOR_PADDED> tag first; fall back to cu128
        CANDIDATE="cu${CUDA_MAJOR}$(printf '%02d' "$CUDA_MINOR")"
        INDEX_URL="https://download.pytorch.org/whl/${CANDIDATE}"
        log "CUDA $CUDA_MAJOR.$CUDA_MINOR — trying wheel tag $CANDIDATE ..."
        # Probe: check if any torch wheel exists under this tag
        if pip index versions torch --index-url "$INDEX_URL" &>/dev/null 2>&1; then
            WHEEL_TAG="$CANDIDATE"
            log "Found wheels for $CANDIDATE."
        else
            warn "No PyTorch wheels found for $CANDIDATE — falling back to cu128."
            warn "Check https://pytorch.org/get-started/locally/ for the latest supported tag."
            WHEEL_TAG="cu128"
        fi
    else
        warn "Unrecognised CUDA version $CUDA_MAJOR.$CUDA_MINOR — falling back to cu128."
        WHEEL_TAG="cu128"
    fi

    if [[ "$WHEEL_TAG" == "cpu" ]]; then
        INDEX_URL="https://download.pytorch.org/whl/cpu"
    else
        INDEX_URL="https://download.pytorch.org/whl/${WHEEL_TAG}"
    fi

    log "Installing torch + torchvision from $INDEX_URL ..."
    $PIP install torch torchvision --index-url "$INDEX_URL"

    # torch-harmonics: Linux-only (needed for neuraloperator DISCO layers)
    log "Installing torch-harmonics ..."
    $PIP install torch-harmonics
}

# ── main install ──────────────────────────────────────────────────────────────
if [[ "$PLATFORM" == "mac" ]]; then
    log "Installing Mac Silicon requirements (torch + mlx + base) ..."
    $PIP install -r "$REPO_ROOT/requirements-mac.txt"

elif [[ "$PLATFORM" == "linux" ]]; then
    install_torch_linux
    log "Installing Linux base requirements ..."
    $PIP install -r "$REPO_ROOT/requirements-linux.txt"
fi

# ── local editable installs (both platforms) ──────────────────────────────────
log "Installing loc_libs (editable) ..."
$PIP install -e "$REPO_ROOT/loc_libs"

log "Installing neuraloperator-4d-2.0.0 (editable) ..."
$PIP install -e "$REPO_ROOT/neuraloperator-4d-2.0.0"

if [[ "$PLATFORM" == "mac" ]]; then
    log "Installing neuralop_mlx (editable, from repo root) ..."
    $PIP install -e "$REPO_ROOT"
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
log "Done."
if [[ "$USING_CONDA" == true ]]; then
    log "Active conda env: $CONDA_PREFIX"
    log "All packages installed into the current conda environment."
else
    log "Virtualenv: $REPO_ROOT/venv"
    log "To activate: source $REPO_ROOT/venv/bin/activate"
fi
echo ""
log "Verify with:"
if [[ "$PLATFORM" == "mac" ]]; then
    log "  python -c \"import neuralop_mlx; import neuralop; import mlx.core; import torch; print('OK')\""
else
    log "  python -c \"import neuralop; import torch; print(torch.cuda.is_available())\""
fi
log "  python -c \"from loc_libs.read_3d_root import read_3d_data; print('loc_libs OK')\""
