#!/usr/bin/env bash
# install_full.sh — FNO4d fresh-environment installer
#
# Creates a brand-new, isolated environment (conda or virtualenv) and installs
# all dependencies from scratch.  No system-wide or parent-environment packages
# are inherited (unlike install.sh which uses --system-site-packages).
#
# Usage:
#   bash install_full.sh --venv  [--name NAME] [--s3] [--gcs]
#   bash install_full.sh --conda [--name NAME] [--s3] [--gcs]
#
# Options:
#   --venv          Create a plain Python virtualenv (default if neither given).
#   --conda         Create a conda environment.
#   --name NAME     Name (or path) for the environment.
#                   Virtualenv default : venv_fno4d  (inside repo root)
#                   Conda default      : fno4d
#   --s3            Also install S3 / S3-compatible storage (s3fs, boto3).
#   --gcs           Also install Google Cloud Storage (gcsfs, google-cloud-storage).
#   -h, --help      Show this help message and exit.
#
# Examples:
#   bash install_full.sh --venv --name .venv
#   bash install_full.sh --conda --name fno4d_cu124 --s3
#   bash install_full.sh --conda --name fno4d_gcs --gcs
#   bash install_full.sh --venv --s3 --gcs          # uses default name venv_fno4d

set -euo pipefail

# ── helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[install_full.sh] $*"; }
warn() { echo "[install_full.sh] WARNING: $*" >&2; }
die()  { echo "[install_full.sh] ERROR: $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: bash install_full.sh [--venv|--conda] [--name NAME] [--s3] [--gcs] [-h]

Environment type (choose one; defaults to --venv):
  --venv          Create a plain Python virtualenv (no system packages).
  --conda         Create a new conda environment.

Options:
  --name NAME     Environment name or path.
                    virtualenv default: venv_fno4d  (inside repo root)
                    conda default:      fno4d
  --torch VERSION Pin a specific PyTorch version (default: latest available).
                  Example: --torch 2.5.1
                  The matching torchvision is resolved automatically.
  --s3            Install S3 / S3-compatible backend (s3fs, boto3).
                  Enables s3:// URIs in read_3d_root.py / read_3d_hdf5.py.
  --gcs           Install Google Cloud Storage backend (gcsfs, google-cloud-storage).
                  Enables gs:// URIs in read_3d_root.py / read_3d_hdf5.py.
  -h, --help      Show this message and exit.

Flags may be combined, e.g.:
  bash install_full.sh --conda --name fno4d_s3 --s3 --gcs
  bash install_full.sh --venv --torch 2.5.1 --s3
EOF
}

# ── parse flags ───────────────────────────────────────────────────────────────
ENV_TYPE=""       # venv | conda
ENV_NAME=""
TORCH_VERSION=""  # empty = latest
INSTALL_S3=false
INSTALL_GCS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)        ENV_TYPE=venv ;;
        --conda)       ENV_TYPE=conda ;;
        --name)        [[ $# -ge 2 ]] || die "--name requires a value"; ENV_NAME="$2"; shift ;;
        --name=*)      ENV_NAME="${1#--name=}" ;;
        --torch)       [[ $# -ge 2 ]] || die "--torch requires a version, e.g. --torch 2.5.1"; TORCH_VERSION="$2"; shift ;;
        --torch=*)     TORCH_VERSION="${1#--torch=}" ;;
        --s3)          INSTALL_S3=true ;;
        --gcs)         INSTALL_GCS=true ;;
        -h|--help)     usage; exit 0 ;;
        *) die "Unknown argument: $1  (run 'bash install_full.sh --help' for usage)" ;;
    esac
    shift
done

# Default to venv if neither was specified
[[ -z "$ENV_TYPE" ]] && ENV_TYPE=venv

# Apply default names
if [[ -z "$ENV_NAME" ]]; then
    [[ "$ENV_TYPE" == venv  ]] && ENV_NAME="venv_fno4d"
    [[ "$ENV_TYPE" == conda ]] && ENV_NAME="fno4d"
fi

# ── detect platform ───────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
log "Platform: OS=$OS  ARCH=$ARCH"

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    PLATFORM="mac"
elif [[ "$OS" == "Linux" ]]; then
    PLATFORM="linux"
else
    die "Unsupported platform: OS=$OS ARCH=$ARCH"
fi

# ── create and activate the environment ───────────────────────────────────────
if [[ "$ENV_TYPE" == venv ]]; then
    # Resolve relative names relative to the repo root
    if [[ "$ENV_NAME" != /* ]]; then
        VENV_DIR="$REPO_ROOT/$ENV_NAME"
    else
        VENV_DIR="$ENV_NAME"
    fi

    if [[ -d "$VENV_DIR" ]]; then
        die "Directory '$VENV_DIR' already exists. Remove it or choose a different --name."
    fi

    log "Creating fresh virtualenv at $VENV_DIR ..."
    # Check that the venv module is available; offer to install it if not.
    if ! python3 -m venv --help &>/dev/null 2>&1; then
        warn "python3 'venv' module not found."
        echo -n "[install_full.sh] Install the 'virtualenv' package now? [y/N] "
        read -r _REPLY
        if [[ "$_REPLY" =~ ^[Yy]$ ]]; then
            if command -v pip3 &>/dev/null; then
                pip3 install --user virtualenv
            elif command -v pip &>/dev/null; then
                pip install --user virtualenv
            else
                die "pip not found. Install Python properly (python.org or your distro package manager) then re-run."
            fi
            # Use the just-installed 'virtualenv' command as a fallback
            if command -v virtualenv &>/dev/null; then
                virtualenv "$VENV_DIR"
            else
                # Try via python3 module path after --user install
                python3 -m virtualenv "$VENV_DIR" || \
                    die "Failed to create virtualenv. Try: python3 -m pip install --user virtualenv"
            fi
        else
            cat >&2 <<'VENV_HELP'
[install_full.sh] Aborting.

To fix manually, install a Python that includes the venv module:

  macOS:
    brew install python       # installs python3 with venv
    xcode-select --install    # may also be needed

  Ubuntu / Debian:
    sudo apt install python3-venv

  Fedora / RHEL:
    sudo dnf install python3-venv   # or python3.11-venv

  Or install virtualenv directly:
    pip3 install --user virtualenv
    virtualenv $VENV_DIR

Then re-run: bash install_full.sh --venv --name $ENV_NAME
VENV_HELP
            exit 1
        fi
    else
        python3 -m venv "$VENV_DIR"   # no --system-site-packages → fully isolated
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PIP="python -m pip"
    log "Virtualenv activated: $VENV_DIR"

elif [[ "$ENV_TYPE" == conda ]]; then
    if ! command -v conda &>/dev/null; then
        # Build arch-specific installer name for the current platform
        _MINI_OS=$( [[ "$OS" == Darwin ]] && echo MacOSX || echo Linux )
        _MINI_ARCH="$ARCH"   # x86_64 | aarch64 | arm64
        _MINI_FILE="Miniconda3-latest-${_MINI_OS}-${_MINI_ARCH}.sh"
        _MINI_URL="https://repo.anaconda.com/miniconda/${_MINI_FILE}"
        cat >&2 <<CONDA_HELP
[install_full.sh] ERROR: conda not found in PATH.

Install Miniconda (recommended — ~130 MB, minimal Python + conda):

  Linux x86_64 (most Lambda / HPC nodes):
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p \$HOME/miniconda3
    \$HOME/miniconda3/bin/conda init "\$(basename "\$SHELL")"
    source ~/.bashrc          # or ~/.zshrc — or open a new terminal

  Linux aarch64 (ARM64 servers / Ampere / Graviton):
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
    bash Miniconda3-latest-Linux-aarch64.sh -b -p \$HOME/miniconda3
    \$HOME/miniconda3/bin/conda init "\$(basename "\$SHELL")"
    source ~/.bashrc

  This system — auto-detected installer (${_MINI_OS} ${_MINI_ARCH}):
    wget ${_MINI_URL}
    bash ${_MINI_FILE} -b -p \$HOME/miniconda3
    \$HOME/miniconda3/bin/conda init "\$(basename "\$SHELL")"
    source ~/.bashrc

  macOS — Homebrew (easiest, updates via brew upgrade):
    brew install --cask miniconda
    conda init "\$(basename "\$SHELL")"
    # open a new terminal

  Full Anaconda (optional — includes scipy, numpy, pandas pre-installed):
    https://www.anaconda.com/download

After installation, open a new terminal so that 'conda' is in PATH, then:
    bash install_full.sh --conda --name ${ENV_NAME}
CONDA_HELP
        exit 1
    fi
    # Check the name doesn't already exist
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        die "Conda env '$ENV_NAME' already exists. Remove it with 'conda env remove -n $ENV_NAME' or choose a different --name."
    fi

    log "Creating fresh conda env '$ENV_NAME' (Python 3.11) ..."
    conda create -y -n "$ENV_NAME" python=3.11

    # Activate inside this shell using conda's shell hook
    # shellcheck disable=SC1090
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    PIP="python -m pip"
    log "Conda env activated: $ENV_NAME"
fi

$PIP install --upgrade pip --quiet

# ── Linux: detect CUDA and install torch ─────────────────────────────────────
install_torch_linux() {
    CUDA_MAJOR=""
    CUDA_MINOR=""

    if command -v nvidia-smi &>/dev/null; then
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
        CANDIDATE="cu${CUDA_MAJOR}${CUDA_MINOR}"
        INDEX_URL="https://download.pytorch.org/whl/${CANDIDATE}"
        log "CUDA $CUDA_MAJOR.$CUDA_MINOR — trying wheel tag $CANDIDATE ..."
        if $PIP index versions torch --index-url "$INDEX_URL" &>/dev/null 2>&1; then
            WHEEL_TAG="$CANDIDATE"
            log "Found wheels for $CANDIDATE."
        else
            warn "No PyTorch wheels found for $CANDIDATE — falling back to cu128."
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

    local _TORCH_SPEC="torch"
    local _USE_PYPI=false   # if true, install without --index-url (PyPI fallback)
    if [[ -n "$TORCH_VERSION" ]]; then
        # PyTorch 2.6+ wheels on the custom index carry a local version suffix
        # (+cpu, +cu124, …).  Pip's == operator does not match across local
        # identifiers, so "torch==2.8.0" won't find "torch==2.8.0+cpu".
        # Probe with --dry-run (fast, no download) to pick the right form.
        local _SPEC_TAGGED="torch==${TORCH_VERSION}+${WHEEL_TAG}"
        local _SPEC_PLAIN="torch==${TORCH_VERSION}"
        local _found=false
        log "Resolving torch==${TORCH_VERSION} on $INDEX_URL ..."
        if $PIP install --dry-run --quiet "${_SPEC_TAGGED}" \
                --index-url "$INDEX_URL" &>/dev/null; then
            _TORCH_SPEC="$_SPEC_TAGGED"; _found=true
            log "  → using ${_SPEC_TAGGED} (custom index)"
        elif $PIP install --dry-run --quiet "${_SPEC_PLAIN}" \
                --index-url "$INDEX_URL" &>/dev/null; then
            _TORCH_SPEC="$_SPEC_PLAIN"; _found=true
            log "  → using ${_SPEC_PLAIN} (custom index)"
        elif [[ "$WHEEL_TAG" == "cpu" ]]; then
            # CPU-only: the Linux custom index caps at 2.6.0+cpu for x86_64.
            # Newer versions exist on PyPI but bundle CUDA 13 runtime packages on Linux.
            log "  Not found on custom CPU index (Linux caps at 2.6.0) — checking PyPI ..."
            if $PIP install --dry-run --quiet "${_SPEC_PLAIN}" &>/dev/null; then
                _TORCH_SPEC="$_SPEC_PLAIN"; _found=true; _USE_PYPI=true
                warn "torch==${TORCH_VERSION} from PyPI bundles CUDA 13 runtime packages on Linux (~3 GB extra)."
                warn "CPU operations will still work; torch.cuda.is_available() returns False without a GPU."
                warn "For a lean CPU-only install, use --torch 2.6.0 (last CPU-only wheel for Linux)."
                log "  → using ${_SPEC_PLAIN} (PyPI; includes CUDA runtime packages on Linux)"
            fi
        fi

        if [[ "$_found" == false ]]; then
            echo >&2 ""
            warn "torch==${TORCH_VERSION} not found."
            warn "Available versions on custom CPU index (pass one to --torch):"
            $PIP index versions torch --index-url "$INDEX_URL" 2>&1 \
                | grep -i "Available versions" \
                | sed 's/+[a-zA-Z0-9]*//g' \
                | tr ',' '\n' \
                | sed 's/[[:space:]]//g; /^$/d; s/Availableversions://g' >&2 || true
            if [[ "$WHEEL_TAG" == "cpu" ]]; then
                warn "Note: Linux CPU wheels cap at 2.6.0 on the custom index."
                warn "      Versions 2.7.0+ on PyPI bundle CUDA 13 packages on Linux and"
                warn "      may require glibc >= 2.34 (Alma/RHEL 9, Ubuntu 22.04+) to install."
                warn "Also available on PyPI (>= 2.7.0 on Linux includes CUDA packages):"
                $PIP index versions torch 2>&1 \
                    | grep -i "Available versions" \
                    | sed 's/+[a-zA-Z0-9]*//g' \
                    | tr ',' '\n' \
                    | sed 's/[[:space:]]//g; /^$/d; s/Availableversions://g' >&2 || true
            fi
            die "Choose a valid version from the lists above and re-run with --torch <version>." 
        fi
    fi
    log "Installing ${_TORCH_SPEC} + torchvision ..."
    if [[ "$_USE_PYPI" == true ]]; then
        $PIP install "${_TORCH_SPEC}" torchvision
    else
        $PIP install "${_TORCH_SPEC}" torchvision --index-url "$INDEX_URL"
    fi

    log "Installing torch-harmonics ..."
    $PIP install torch-harmonics
}

# ── main install ──────────────────────────────────────────────────────────────
if [[ "$PLATFORM" == "mac" ]]; then
    if [[ -n "$TORCH_VERSION" ]]; then
        log "Installing Mac Silicon requirements (base + mlx) with pinned torch==${TORCH_VERSION} ..."
        $PIP install -r "$REPO_ROOT/requirements-base.txt" "mlx>=0.16"
        # PyPI mac wheels use plain version numbers (no +cpu suffix)
        if ! $PIP install --dry-run --quiet "torch==${TORCH_VERSION}" \
                &>/dev/null; then
            echo >&2 ""
            warn "torch==${TORCH_VERSION} not found on PyPI."
            warn "Available versions (pass one of these to --torch):"
            $PIP index versions torch 2>&1 \
                | grep -i "Available versions" \
                | sed 's/+[a-zA-Z0-9]*//g' \
                | tr ',' '\n' \
                | sed 's/[[:space:]]//g; /^$/d; s/Availableversions://g' >&2 || true
            die "Choose a valid version from the list above and re-run with --torch <version>."
        fi
        $PIP install "torch==${TORCH_VERSION}" torchvision
    else
        log "Installing Mac Silicon requirements (torch + mlx + base) ..."
        $PIP install -r "$REPO_ROOT/requirements-mac.txt"
    fi

elif [[ "$PLATFORM" == "linux" ]]; then
    install_torch_linux
    log "Installing Linux base requirements ..."
    $PIP install -r "$REPO_ROOT/requirements-linux.txt"
fi

# ── cloud storage (opt-in) ────────────────────────────────────────────────────
# In a fresh env there is no pre-existing fsspec conflict, so we pre-install
# the fsspec driver(s) first to lock aiobotocore+botocore before boto3 arrives,
# then record the resolved versions as a constraints file so the requirements
# install doesn't upgrade them.
if [[ "$INSTALL_S3" == true || "$INSTALL_GCS" == true ]]; then
    _CLOUD_CONSTRAINTS=$(mktemp /tmp/fno4d_cloud.XXXXXX)

    [[ "$INSTALL_S3"  == true ]] && { log "Pre-installing s3fs ...";  $PIP install s3fs; }
    [[ "$INSTALL_GCS" == true ]] && { log "Pre-installing gcsfs ..."; $PIP install gcsfs; }

    # Capture resolved versions to prevent re-resolution during full requirements install
    python3 -c "
try:
    import fsspec; print('fsspec=='+fsspec.__version__)
except ImportError: pass
try:
    import s3fs;   print('s3fs=='+s3fs.__version__)
except ImportError: pass
try:
    import gcsfs;  print('gcsfs=='+gcsfs.__version__)
except ImportError: pass
" > "$_CLOUD_CONSTRAINTS" 2>/dev/null || true

    log "Cloud constraints: $(tr '\n' ' ' < "$_CLOUD_CONSTRAINTS")"

    if [[ "$INSTALL_S3" == true ]]; then
        log "Installing S3 storage requirements (requirements-s3.txt) ..."
        $PIP install -r "$REPO_ROOT/requirements-s3.txt" -c "$_CLOUD_CONSTRAINTS"
    fi
    if [[ "$INSTALL_GCS" == true ]]; then
        log "Installing GCS storage requirements (requirements-gcs.txt) ..."
        $PIP install -r "$REPO_ROOT/requirements-gcs.txt" -c "$_CLOUD_CONSTRAINTS"
    fi

    rm -f "$_CLOUD_CONSTRAINTS"
else
    log "Cloud storage not requested. Re-run with --s3 and/or --gcs to add cloud backends."
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

# ── nbstripout git hook ───────────────────────────────────────────────────────
# .gitattributes is already tracked in the repo (filter=nbstripout on *.ipynb);
# we only need to register the filter driver in .git/config for this clone.
$PIP install nbstripout --quiet
if command -v git &>/dev/null && git -C "$REPO_ROOT" rev-parse --git-dir &>/dev/null; then
    log "Installing nbstripout git filter ..."
    nbstripout --install
    log "nbstripout installed — notebook outputs will be stripped on commit."
else
    warn "git not found or not a git repo — skipping nbstripout filter registration."
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
log "Done."
if [[ "$ENV_TYPE" == venv ]]; then
    log "Virtualenv created at: $VENV_DIR"
    log "To activate: source $VENV_DIR/bin/activate"
else
    log "Conda env created: $ENV_NAME"
    log "To activate: conda activate $ENV_NAME"
fi
echo ""
log "Verify with:"
if [[ "$PLATFORM" == "mac" ]]; then
    log "  python -c \"import neuralop_mlx; import neuralop; import mlx.core; import torch; print('OK')\""
else
    log "  python -c \"import neuralop; import torch; print('CUDA:', torch.cuda.is_available())\""
fi
log "  python -c \"from loc_libs.read_3d_root import read_3d_data; print('loc_libs OK')\""
