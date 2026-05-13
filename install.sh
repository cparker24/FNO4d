#!/usr/bin/env bash
# install.sh — FNO4d environment setup
#
# Detects platform (Mac arm64 / Linux x86_64 / Linux aarch64) and creates a
# virtualenv with --system-site-packages so that packages already present in
# the system Python, a conda env, or a Docker image are visible without
# reinstalling them.
#
# Usage:
#   bash install.sh            # always creates/reuses venv/ (default)
#   bash install.sh --conda    # install directly into the active non-base
#                              # conda env instead of creating a venv
#
# The --conda flag is useful when you want packages registered in the conda
# env itself (e.g. for conda-managed torch builds). It is refused if the
# active env is 'base'.

set -euo pipefail

# ── helpers ──────────────────────────────────────────────────────────────────
log()  { echo "[install.sh] $*"; }
warn() { echo "[install.sh] WARNING: $*" >&2; }
die()  { echo "[install.sh] ERROR: $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── parse flags ───────────────────────────────────────────────────────────────
USE_CONDA_FLAG=false
for arg in "$@"; do
    case "$arg" in
        --conda) USE_CONDA_FLAG=true ;;
        *) die "Unknown argument: $arg  (usage: bash install.sh [--conda])" ;;
    esac
done

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

# ── set up environment ────────────────────────────────────────────────────────
USING_CONDA=false

if [[ "$USE_CONDA_FLAG" == true ]]; then
    # --conda: install into active named conda env (no venv)
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        die "--conda specified but no conda environment is active. Run 'conda activate <name>' first."
    fi
    if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
        die "--conda specified but the active env is 'base'. Activate a named env first."
    fi
    log "Using conda env: $CONDA_PREFIX  (env: ${CONDA_DEFAULT_ENV:-unknown})"
    USING_CONDA=true
    PIP="python -m pip"
else
    # Default: always create/reuse a virtualenv with --system-site-packages.
    # This works whether conda is active or not — the venv inherits whatever
    # is installed in the parent Python (conda env, system, Docker image).
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        log "conda env '${CONDA_DEFAULT_ENV:-}' is active — venv will inherit its packages via --system-site-packages."
    fi
    VENV_DIR="$REPO_ROOT/venv"
    if [[ -d "$VENV_DIR" ]]; then
        log "Existing virtualenv found at $VENV_DIR — reusing it."
    else
        log "Creating virtualenv at $VENV_DIR (--system-site-packages) ..."
        python3 -m venv "$VENV_DIR" --system-site-packages
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PIP="python -m pip"
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

# ── pin s3fs/gcsfs to match any existing fsspec in the environment ────────────
# s3fs, gcsfs, and fsspec all share the same version number (fsspec org releases
# them in sync).  If fsspec is already installed (e.g. from conda or the system
# site-packages), install s3fs and gcsfs at that exact version so they don't
# trigger a conflicting fsspec upgrade.  This also pre-installs aiobotocore and
# locks botocore before boto3 arrives, eliminating the boto3 backtracking problem.
_FSSPEC_VER=$(python3 -c "import fsspec; print(fsspec.__version__)" 2>/dev/null || true)
if [[ -n "$_FSSPEC_VER" ]]; then
    log "Existing fsspec $_FSSPEC_VER detected — pre-installing s3fs==$_FSSPEC_VER gcsfs==$_FSSPEC_VER ..."
    $PIP install "s3fs==$_FSSPEC_VER" "gcsfs==$_FSSPEC_VER"
else
    log "No existing fsspec — pre-installing s3fs/gcsfs (latest) to lock botocore before boto3 ..."
    $PIP install s3fs gcsfs
fi

# ── main install ──────────────────────────────────────────────────────────────
if [[ "$PLATFORM" == "mac" ]]; then
    log "Installing Mac Silicon requirements (torch + mlx + base) ..."
    $PIP install -r "$REPO_ROOT/requirements-mac.txt"

elif [[ "$PLATFORM" == "linux" ]]; then
    install_torch_linux
    log "Installing Linux base requirements ..."
    $PIP install -r "$REPO_ROOT/requirements-linux.txt" -c "$_ENV_CONSTRAINTS"
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

# ── nbstripout git hook (skipped if git is not available) ────────────────────
$PIP install nbstripout --quiet
if command -v git &>/dev/null && git -C "$REPO_ROOT" rev-parse --git-dir &>/dev/null; then
    log "Installing nbstripout git filter ..."
    python3 -m nbstripout --install --attributes "$REPO_ROOT/.gitattributes"
    log "nbstripout installed — notebook outputs will be stripped on commit."
else
    warn "git not found or not a git repo — skipping nbstripout filter registration."
    log "nbstripout is installed; to enable later: python3 -m nbstripout --install"
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
log "Done."
if [[ "$USING_CONDA" == true ]]; then
    log "Installed into conda env: $CONDA_PREFIX  (${CONDA_DEFAULT_ENV:-})"
    log "No activation needed — already in the target env."
else
    log "Virtualenv: $REPO_ROOT/venv  (--system-site-packages)"
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
echo ""
log "Checking neuralop and loc_libs resolve to the local fork ..."
python3 -c "
import neuralop, loc_libs, os
repo = os.path.abspath('$REPO_ROOT')
no_path  = os.path.abspath(neuralop.__file__)
loc_path = os.path.abspath(loc_libs.__file__)
ok = True
if repo in no_path:
    print(f'[OK]   neuralop  -> {no_path}')
else:
    print(f'[WARN] neuralop  -> {no_path}')
    print(f'       Expected path inside {repo} -- local fork may be shadowed!')
    ok = False
if repo in loc_path:
    print(f'[OK]   loc_libs  -> {loc_path}')
else:
    print(f'[WARN] loc_libs  -> {loc_path}')
    print(f'       Expected path inside {repo} -- local copy may be shadowed!')
    ok = False
if not ok:
    print()
    print('To debug: run pip list and check which neuraloperator / loc-libs entry appears first.')
"
