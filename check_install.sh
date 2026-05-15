#!/usr/bin/env bash
# check_install.sh — Pre-flight checks for install.sh
#
# Mirrors every probe that install.sh performs WITHOUT modifying the
# environment.  Differs from check_install_full.sh in three important ways:
#
#   1. The probe venv uses --system-site-packages, so packages already
#      present in the system / active conda env are visible.
#   2. torch is only probed if it is NOT already importable (install.sh
#      skips torch when it is already present).
#   3. An existing fsspec version is detected and shown as a constraint.
#
# Usage (same flags as install.sh):
#   bash check_install.sh [--s3] [--gcs] [--conda]

set -euo pipefail

# ── helpers ───────────────────────────────────────────────────────────────────
BOLD=$'\033[1m'; GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; NC=$'\033[0m'

hdr()  { echo; echo "${BOLD}── $* ──${NC}"; }
chk()  { echo "  $*"; }
pass() { echo "  ${GREEN}PASS${NC}  $*"; }
fail() { echo "  ${RED}FAIL${NC}  $*" >&2; FAILURES=$(( FAILURES + 1 )); }
warn() { echo "  ${YELLOW}WARN${NC}  $*"; }
die()  { echo "${RED}ERROR:${NC} $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILURES=0

# ── usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: bash check_install.sh [--s3] [--gcs] [--conda]

Probes every install step from install.sh without installing anything.
Use the same flags you plan to pass to install.sh.

  --s3     Check S3 requirements
  --gcs    Check GCS requirements
  --conda  Check as if installing into the active conda env (no venv)
EOF
}

# ── parse flags ───────────────────────────────────────────────────────────────
INSTALL_S3=false
INSTALL_GCS=false
USE_CONDA=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --s3)      INSTALL_S3=true ;;
        --gcs)     INSTALL_GCS=true ;;
        --conda)   USE_CONDA=true ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $1  (try --help)" ;;
    esac
    shift
done

# ── detect platform ───────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
if   [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    PLATFORM="mac"
elif [[ "$OS" == "Linux" ]]; then
    PLATFORM="linux"
else
    die "Unsupported platform: $OS $ARCH"
fi

# ── detect currently importable packages ─────────────────────────────────────
TORCH_IMPORTED=""
TORCH_CUDA=""
FSSPEC_VER=""

TORCH_IMPORTED=$(python3 -c "
import torch, sys
print(torch.__version__)
" 2>/dev/null || true)

if [[ -n "$TORCH_IMPORTED" ]]; then
    TORCH_CUDA=$(python3 -c "
import torch
print(torch.cuda.is_available())
" 2>/dev/null || true)
fi

FSSPEC_VER=$(python3 -c "import fsspec; print(fsspec.__version__)" 2>/dev/null || true)

# ── summary header ────────────────────────────────────────────────────────────
echo ""
echo "${BOLD}FNO4d install.sh pre-flight check (system-site-packages)${NC}"
echo "  Platform : $OS $ARCH  (→ ${PLATFORM} install path)"
echo "  Python   : $(python3 --version 2>&1)"
if [[ "$USE_CONDA" == true ]]; then
    echo "  Mode     : --conda (active env: ${CONDA_DEFAULT_ENV:-not set})"
else
    echo "  Mode     : venv (--system-site-packages)  →  $REPO_ROOT/venv"
fi
if [[ -n "$TORCH_IMPORTED" ]]; then
    echo "  torch    : ${GREEN}${TORCH_IMPORTED}${NC} already importable — install.sh will skip torch install"
    echo "  CUDA     : torch.cuda.is_available() = $TORCH_CUDA"
else
    echo "  torch    : ${YELLOW}not importable${NC} — install.sh will install torch"
fi
if [[ -n "$FSSPEC_VER" ]]; then
    echo "  fsspec   : ${FSSPEC_VER} already present — will be pinned as constraint"
else
    echo "  fsspec   : not present — will be installed freely"
fi
_OPTS=""
[[ "$INSTALL_S3"  == true ]] && _OPTS+=" --s3"
[[ "$INSTALL_GCS" == true ]] && _OPTS+=" --gcs"
[[ -n "$_OPTS" ]] && echo "  options  :${_OPTS}" || echo "  options  : (none)"

# ══════════════════════════════════════════════════════════════════════════════
hdr "Environment pre-check"
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$USE_CONDA" == true ]]; then
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        fail "No conda env is active — activate one first: conda activate <name>"
    elif [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
        fail "Active conda env is 'base' — install.sh refuses this. Activate a named env."
    else
        pass "Conda env active: ${CONDA_DEFAULT_ENV} ($CONDA_PREFIX)"
    fi
else
    VENV_DIR="$REPO_ROOT/venv"
    if [[ -d "$VENV_DIR" ]]; then
        warn "venv/ already exists — install.sh will reuse it (not a problem)"
        chk  "      To start fresh: rm -rf $VENV_DIR"
    else
        pass "No existing venv/ — install.sh will create it"
    fi
    if python3 -m venv --help &>/dev/null 2>&1; then
        pass "python3 venv module available"
    else
        fail "python3 venv module not found (install python3-venv / python3-devel)"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
hdr "Probe environment setup"
# ══════════════════════════════════════════════════════════════════════════════

PROBE_DIR=$(mktemp -d)
trap 'rm -rf "$PROBE_DIR"' EXIT

if [[ "$USE_CONDA" == true ]]; then
    # For --conda mode: probe directly with current python (already in conda env)
    PPIP="python -m pip"
    pass "Using active conda env as probe environment (no temp venv needed)"
else
    # Mirror install.sh: create a --system-site-packages venv so the probe sees
    # all packages already present in the parent environment.
    if ! python3 -m venv "$PROBE_DIR/venv" --system-site-packages &>/dev/null 2>&1; then
        die "Cannot create probe venv — fix the venv issue first."
    fi
    PPIP="$PROBE_DIR/venv/bin/python3 -m pip"
    $PPIP install --upgrade pip --quiet
    pass "Probe venv ready (--system-site-packages, cleaned up on exit)"
fi

# ── probe helpers ─────────────────────────────────────────────────────────────

# Print "Would install" (new) and count already-satisfied from dry-run output.
_show_delta() {
    local tmp="$1"
    local new_pkgs already_n
    new_pkgs=$(grep "^Would install" "$tmp" \
        | sed 's/^Would install //' \
        | tr ' ' '\n' \
        | sed 's/\(.*\)-\([0-9]\)/\1==\2/' \
        | sort)
    already_n=$(grep -c "^Requirement already satisfied" "$tmp" 2>/dev/null || true)

    if [[ -n "$new_pkgs" ]]; then
        chk   "        [+new packages]"
        echo "$new_pkgs" | sed 's/^/        /'
    else
        chk   "        (no new packages — all satisfied from system/conda)"
    fi
    if [[ "$already_n" -gt 0 ]]; then
        chk "        ($already_n packages already satisfied from system/conda)"
    fi
}

# probe_pkg LABEL PKG [PKG...] [-c CONSTRAINTS] [--index-url URL]
#   Does NOT use --ignore-installed so system packages count as satisfied.
probe_pkg() {
    local label="$1"; shift
    local tmp_out
    tmp_out=$(mktemp)
    if $PPIP install --dry-run "$@" >"$tmp_out" 2>&1; then
        pass "$label"
        _show_delta "$tmp_out"
        rm -f "$tmp_out"; return 0
    else
        fail "$label"
        grep -iE "error|could not find|no matching|not find|conflict" "$tmp_out" \
            | head -5 | sed "s/^/        /" >&2 || true
        rm -f "$tmp_out"; return 1
    fi
}

# probe_req LABEL REQFILE [extra pip args...]
probe_req() {
    local label="$1"; local reqfile="$2"; shift 2
    if [[ ! -f "$reqfile" ]]; then
        fail "$label — file not found: $reqfile"
        return 1
    fi
    probe_pkg "$label" -r "$reqfile" "$@"
}

# Build a constraints argument from the existing fsspec version (if any).
_FSSPEC_ARGS=()
if [[ -n "$FSSPEC_VER" ]]; then
    _FSSPEC_CONSTRAINT="$PROBE_DIR/fsspec_pin.txt"
    printf 'fsspec==%s\n' "$FSSPEC_VER" > "$_FSSPEC_CONSTRAINT"
    _FSSPEC_ARGS=(-c "$_FSSPEC_CONSTRAINT")
fi

# ══════════════════════════════════════════════════════════════════════════════
# Linux: CUDA detection (same logic as install.sh — including the printf bug
# that is still present in install.sh, shown here for accurate simulation).
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$PLATFORM" == "linux" ]]; then

    hdr "CUDA detection"

    CUDA_MAJOR=""
    CUDA_MINOR=""
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || true)
        if [[ -n "$CUDA_VER" ]]; then
            CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
            CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
            pass "nvidia-smi: CUDA $CUDA_MAJOR.$CUDA_MINOR detected"
        else
            warn "nvidia-smi present but CUDA version could not be parsed → CPU-only path"
        fi
    else
        warn "nvidia-smi not found → CPU-only path"
    fi

    WHEEL_TAG=""
    if [[ -z "$CUDA_MAJOR" ]]; then
        WHEEL_TAG="cpu"
        warn "No GPU — CPU-only torch"
    elif [[ "$CUDA_MAJOR" -eq 11 && "$CUDA_MINOR" -ge 8 ]]; then
        WHEEL_TAG="cu118"
    elif [[ "$CUDA_MAJOR" -eq 12 ]]; then
        if   [[ "$CUDA_MINOR" -le 3 ]]; then WHEEL_TAG="cu121"
        elif [[ "$CUDA_MINOR" -le 7 ]]; then WHEEL_TAG="cu124"
        else                                  WHEEL_TAG="cu128"
        fi
    elif [[ "$CUDA_MAJOR" -ge 13 ]]; then
        WHEEL_TAG="cu${CUDA_MAJOR}${CUDA_MINOR}"
        _CANDIDATE_URL="https://download.pytorch.org/whl/${WHEEL_TAG}"
        chk "Checking wheel index for $WHEEL_TAG ..."
        if $PPIP index versions torch --index-url "$_CANDIDATE_URL" &>/dev/null 2>&1; then
            pass "Wheel index confirmed: $WHEEL_TAG"
        else
            warn "No torch wheels for $WHEEL_TAG — install.sh would fall back to cu128"
            WHEEL_TAG="cu128"
        fi
    else
        warn "Unrecognised CUDA $CUDA_MAJOR.$CUDA_MINOR — install.sh falls back to cu128"
        WHEEL_TAG="cu128"
    fi

    if [[ "$WHEEL_TAG" == "cpu" ]]; then
        INDEX_URL="https://download.pytorch.org/whl/cpu"
    else
        INDEX_URL="https://download.pytorch.org/whl/${WHEEL_TAG}"
    fi
    chk "Wheel tag : $WHEEL_TAG"
    chk "Index URL : $INDEX_URL"
fi

# ══════════════════════════════════════════════════════════════════════════════
hdr "torch"
# ══════════════════════════════════════════════════════════════════════════════

if [[ -n "$TORCH_IMPORTED" ]]; then
    pass "torch ${TORCH_IMPORTED} already importable — install.sh will skip torch install"
    chk  "      torch.cuda.is_available() = $TORCH_CUDA"
    # Verify the importable torch is from the expected location
    TORCH_LOC=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null || true)
    chk "      location: $TORCH_LOC"
else
    if [[ "$PLATFORM" == "mac" ]]; then
        probe_pkg "torch + torchvision (PyPI, macOS)" torch torchvision \
            "${_FSSPEC_ARGS[@]+"${_FSSPEC_ARGS[@]}"}"
    elif [[ "$PLATFORM" == "linux" ]]; then
        probe_pkg "torch + torchvision (${WHEEL_TAG} index)" \
            torch torchvision --index-url "$INDEX_URL" \
            "${_FSSPEC_ARGS[@]+"${_FSSPEC_ARGS[@]}"}"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
hdr "torch-harmonics (Linux only)"
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$PLATFORM" == "linux" ]]; then
    probe_pkg "torch-harmonics" torch-harmonics \
        "${_FSSPEC_ARGS[@]+"${_FSSPEC_ARGS[@]}"}"
else
    chk "(torch-harmonics not installed on macOS)"
fi

# ══════════════════════════════════════════════════════════════════════════════
hdr "Platform requirements file"
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$PLATFORM" == "mac" ]]; then
    probe_req "requirements-mac.txt" "$REPO_ROOT/requirements-mac.txt" \
        "${_FSSPEC_ARGS[@]+"${_FSSPEC_ARGS[@]}"}"
elif [[ "$PLATFORM" == "linux" ]]; then
    probe_req "requirements-linux.txt" "$REPO_ROOT/requirements-linux.txt" \
        "${_FSSPEC_ARGS[@]+"${_FSSPEC_ARGS[@]}"}"
fi

# ══════════════════════════════════════════════════════════════════════════════
if [[ "$INSTALL_S3" == true ]]; then
    hdr "S3 requirements"
    if [[ -n "$FSSPEC_VER" ]]; then
        chk "Pinning s3fs to fsspec==${FSSPEC_VER} (existing system version)"
        probe_pkg "s3fs==${FSSPEC_VER}" "s3fs==${FSSPEC_VER}" \
            "${_FSSPEC_ARGS[@]+"${_FSSPEC_ARGS[@]}"}"
    else
        probe_pkg "s3fs" s3fs "${_FSSPEC_ARGS[@]+"${_FSSPEC_ARGS[@]}"}"
    fi
    probe_req "requirements-s3.txt" "$REPO_ROOT/requirements-s3.txt" \
        "${_FSSPEC_ARGS[@]+"${_FSSPEC_ARGS[@]}"}"
fi

if [[ "$INSTALL_GCS" == true ]]; then
    hdr "GCS requirements"
    if [[ -n "$FSSPEC_VER" ]]; then
        chk "Pinning gcsfs to fsspec==${FSSPEC_VER} (existing system version)"
        probe_pkg "gcsfs==${FSSPEC_VER}" "gcsfs==${FSSPEC_VER}" \
            "${_FSSPEC_ARGS[@]+"${_FSSPEC_ARGS[@]}"}"
    else
        probe_pkg "gcsfs" gcsfs "${_FSSPEC_ARGS[@]+"${_FSSPEC_ARGS[@]}"}"
    fi
    probe_req "requirements-gcs.txt" "$REPO_ROOT/requirements-gcs.txt" \
        "${_FSSPEC_ARGS[@]+"${_FSSPEC_ARGS[@]}"}"
fi

# ══════════════════════════════════════════════════════════════════════════════
hdr "Local editable packages"
# ══════════════════════════════════════════════════════════════════════════════

for _pkg in "loc_libs" "neuraloperator-4d-2.0.0"; do
    _dir="$REPO_ROOT/$_pkg"
    if [[ -f "$_dir/pyproject.toml" || -f "$_dir/setup.py" ]]; then
        pass "$_pkg  — source present"
    else
        fail "$_pkg  — missing directory or pyproject.toml/setup.py at $_dir"
    fi
done

if [[ "$PLATFORM" == "mac" ]]; then
    if [[ -f "$REPO_ROOT/pyproject.toml" ]]; then
        pass "neuralop_mlx  — pyproject.toml present at repo root"
    else
        fail "neuralop_mlx  — pyproject.toml not found at repo root"
    fi
fi

# ── shadow check: warn if system packages would shadow local ones ─────────────
hdr "Shadowing check (system vs. local editable)"
_shadow_ok=true
for _pkg_import in "neuralop:neuraloperator-4d-2.0.0" "loc_libs:loc_libs"; do
    _import="${_pkg_import%%:*}"
    _local="${_pkg_import##*:}"
    _loc=$(python3 -c "
import importlib.util, sys
spec = importlib.util.find_spec('$_import')
print(spec.origin if spec else 'NOT FOUND')
" 2>/dev/null || true)
    if [[ "$_loc" == "NOT FOUND" ]]; then
        chk "  $_import: not currently importable (will be installed as editable)"
    elif echo "$_loc" | grep -q "$REPO_ROOT"; then
        pass "$_import already points inside repo (editable install active)"
    else
        warn "$_import found OUTSIDE repo: $_loc"
        warn "      System copy may shadow the local editable install."
        warn "      Check 'pip show $_import' to see which takes precedence."
        _shadow_ok=false
    fi
done
[[ "$_shadow_ok" == true ]] && pass "No shadowing issues detected" || true

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════"
if [[ "$FAILURES" -eq 0 ]]; then
    _CMD="bash install.sh"
    [[ "$USE_CONDA"   == true ]] && _CMD+=" --conda"
    [[ "$INSTALL_S3"  == true ]] && _CMD+=" --s3"
    [[ "$INSTALL_GCS" == true ]] && _CMD+=" --gcs"
    echo "  ${GREEN}${BOLD}ALL CHECKS PASSED${NC}"
    echo "  Safe to run: $_CMD"
else
    echo "  ${RED}${BOLD}$FAILURES CHECK(S) FAILED${NC}"
    echo "  Fix the issues above, then re-run this script before installing."
fi
echo "══════════════════════════════════════════════════════════"
echo ""

exit "$FAILURES"
