#!/usr/bin/env bash
# check_install.sh — Pre-flight checks for install_full.sh
#
# Runs every package-resolution probe that install_full.sh would perform,
# WITHOUT creating the target environment or downloading/installing anything.
# Exits non-zero (exit code = number of failures) if any probe fails.
#
# Pass the same flags you intend to use with install_full.sh:
#   bash check_install.sh --venv  [--name NAME] [--torch VERSION] [--s3] [--gcs]
#   bash check_install.sh --conda [--name NAME] [--torch VERSION] [--s3] [--gcs]

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
Usage: bash check_install.sh [--venv|--conda] [--name NAME] [--torch VERSION] [--s3] [--gcs]

Probes every install step from install_full.sh without installing anything.
Use the same flags you plan to pass to install_full.sh.
EOF
}

# ── parse flags (mirrors install_full.sh) ─────────────────────────────────────
ENV_TYPE=""
ENV_NAME=""
TORCH_VERSION=""
INSTALL_S3=false
INSTALL_GCS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)    ENV_TYPE=venv ;;
        --conda)   ENV_TYPE=conda ;;
        --name)    [[ $# -ge 2 ]] || die "--name requires a value"; ENV_NAME="$2"; shift ;;
        --name=*)  ENV_NAME="${1#--name=}" ;;
        --torch)   [[ $# -ge 2 ]] || die "--torch requires a version"; TORCH_VERSION="$2"; shift ;;
        --torch=*) TORCH_VERSION="${1#--torch=}" ;;
        --s3)      INSTALL_S3=true ;;
        --gcs)     INSTALL_GCS=true ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $1  (try --help)" ;;
    esac
    shift
done

[[ -z "$ENV_TYPE" ]] && ENV_TYPE=venv
if [[ -z "$ENV_NAME" ]]; then
    [[ "$ENV_TYPE" == venv  ]] && ENV_NAME="venv_fno4d"
    [[ "$ENV_TYPE" == conda ]] && ENV_NAME="fno4d"
fi

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

# ── summary header ────────────────────────────────────────────────────────────
echo ""
echo "${BOLD}FNO4d install_full.sh pre-flight check${NC}"
echo "  Platform : $OS $ARCH  (→ ${PLATFORM} install path)"
echo "  Python   : $(python3 --version 2>&1)"
echo "  Env type : $ENV_TYPE  →  $ENV_NAME"
[[ -n "$TORCH_VERSION" ]] && echo "  torch    : pinned to $TORCH_VERSION" \
                           || echo "  torch    : latest available"
_OPTS=""
[[ "$INSTALL_S3"  == true ]] && _OPTS+=" --s3"
[[ "$INSTALL_GCS" == true ]] && _OPTS+=" --gcs"
[[ -n "$_OPTS" ]] && echo "  options  :${_OPTS}" || echo "  options  : (none)"

# ══════════════════════════════════════════════════════════════════════════════
hdr "Environment pre-check"
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$ENV_TYPE" == venv ]]; then
    VENV_TARGET="$REPO_ROOT/$ENV_NAME"
    [[ "$ENV_NAME" == /* ]] && VENV_TARGET="$ENV_NAME"
    if [[ -d "$VENV_TARGET" ]]; then
        fail "Target already exists: $VENV_TARGET"
        chk  "      install_full.sh will abort. Remove it first: rm -rf $VENV_TARGET"
    else
        pass "Target path clear: $VENV_TARGET"
    fi
    if python3 -m venv --help &>/dev/null 2>&1; then
        pass "python3 venv module available"
    else
        fail "python3 venv module not found (install python3-venv / python3-devel)"
    fi

elif [[ "$ENV_TYPE" == conda ]]; then
    if command -v conda &>/dev/null; then
        pass "conda found: $(conda --version 2>&1)"
        if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" 2>/dev/null; then
            fail "Conda env '$ENV_NAME' already exists"
            chk  "      Remove it: conda env remove -n $ENV_NAME"
        else
            pass "Conda env name '$ENV_NAME' is available"
        fi
    else
        fail "conda not found — see install_full.sh --help for install instructions"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
hdr "Probe environment setup"
# ══════════════════════════════════════════════════════════════════════════════

PROBE_DIR=$(mktemp -d)
trap 'rm -rf "$PROBE_DIR"' EXIT

if ! python3 -m venv "$PROBE_DIR/venv" &>/dev/null 2>&1; then
    die "Cannot create probe venv — python3 venv not available. Fix the venv issue first."
fi
PPIP="$PROBE_DIR/venv/bin/python3 -m pip"
$PPIP install --upgrade pip --quiet
pass "Temporary probe venv ready (cleaned up on exit)"

# ── probe helpers ─────────────────────────────────────────────────────────────

# Print "Would install" versions from a captured pip dry-run output file.
_show_versions() {
    grep "^Would install" "$1" \
        | sed 's/^Would install //' \
        | tr ' ' '\n' \
        | sed 's/\(.*\)-\([0-9]\)/\1==\2/' \
        | sort \
        | sed 's/^/        /'
}
# probe_pkg LABEL PKG [PKG...] [--index-url URL]
#   Runs pip install --dry-run, reports PASS/FAIL + resolved versions.
probe_pkg() {
    local label="$1"; shift
    local tmp_out
    tmp_out=$(mktemp)
    if $PPIP install --dry-run --ignore-installed "$@" >"$tmp_out" 2>&1; then
        pass "$label"
        # Parse the "Would install pkg-X.Y.Z ..." line into one version per line
        grep "^Would install" "$tmp_out" \
            | sed 's/^Would install //' \
            | tr ' ' '\n' \
            | sed 's/\(.*\)-\([0-9]\)/\1==\2/' \
            | sort \
            | sed 's/^/        /'
        rm -f "$tmp_out"; return 0
    else
        fail "$label"
        grep -iE "error|could not find|no matching|not find" "$tmp_out" \
            | head -3 | sed "s/^/        /" >&2 || true
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

# ══════════════════════════════════════════════════════════════════════════════
# Linux: CUDA detection + wheel tag
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
        warn "nvidia-smi not found → CPU-only path (training will be slow)"
    fi

    WHEEL_TAG=""
    if [[ -z "$CUDA_MAJOR" ]]; then
        WHEEL_TAG="cpu"
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
            warn "No torch wheels found for $WHEEL_TAG — install_full.sh would fall back to cu128"
            WHEEL_TAG="cu128"
        fi
    else
        warn "Unrecognised CUDA $CUDA_MAJOR.$CUDA_MINOR — install_full.sh would fall back to cu128"
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
hdr "torch + torchvision"
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$PLATFORM" == "mac" ]]; then
    if [[ -n "$TORCH_VERSION" ]]; then
        probe_pkg "torch==${TORCH_VERSION} + torchvision (PyPI, macOS)" \
            "torch==${TORCH_VERSION}" torchvision
    else
        probe_req "requirements-mac.txt" "$REPO_ROOT/requirements-mac.txt"
    fi

elif [[ "$PLATFORM" == "linux" ]]; then
    TORCH_RESOLVED=""   # will hold the spec that succeeded

    if [[ -n "$TORCH_VERSION" ]]; then
        _SPEC_TAGGED="torch==${TORCH_VERSION}+${WHEEL_TAG}"
        _SPEC_PLAIN="torch==${TORCH_VERSION}"

        # Probe 1: tagged (+cu130 / +cpu / …)
        chk "Probe 1: ${_SPEC_TAGGED}  on  $INDEX_URL"
        _tmp=$(mktemp)
        if $PPIP install --dry-run --ignore-installed \
                "${_SPEC_TAGGED}" torchvision \
                --index-url "$INDEX_URL" >"$_tmp" 2>&1; then
            pass "torch → ${_SPEC_TAGGED}  (custom index ${WHEEL_TAG})"
            _show_versions "$_tmp"
            TORCH_RESOLVED="${_SPEC_TAGGED}"
        else
            chk "  not found as tagged — trying plain"

            # Probe 2: plain (no local-version suffix)
            chk "Probe 2: ${_SPEC_PLAIN}  on  $INDEX_URL"
            if $PPIP install --dry-run --ignore-installed \
                    "${_SPEC_PLAIN}" torchvision \
                    --index-url "$INDEX_URL" >"$_tmp" 2>&1; then
                pass "torch → ${_SPEC_PLAIN}  (custom index ${WHEEL_TAG})"
                _show_versions "$_tmp"
                TORCH_RESOLVED="${_SPEC_PLAIN}"

            elif [[ "$WHEEL_TAG" == "cpu" ]]; then
                # Probe 3: PyPI fallback (CPU only — will bundle CUDA packages on Linux)
                chk "Probe 3: ${_SPEC_PLAIN}  on  PyPI (CPU fallback)"
                if $PPIP install --dry-run --ignore-installed \
                        "${_SPEC_PLAIN}" torchvision >"$_tmp" 2>&1; then
                    warn "torch → ${_SPEC_PLAIN} resolves from PyPI"
                    warn "      On Linux this bundles CUDA 13 runtime packages (~3 GB)."
                    warn "      Use --torch 2.6.0 for a CPU-only wheel without CUDA packages."
                    _show_versions "$_tmp"
                    TORCH_RESOLVED="${_SPEC_PLAIN} (PyPI)"
                else
                    fail "torch==${TORCH_VERSION} — not found on custom index or PyPI"
                    grep -iE "error|could not find|no matching" "$_tmp" \
                        | head -3 | sed "s/^/        /" >&2 || true
                    chk "      Available on custom index:"
                    $PPIP index versions torch --index-url "$INDEX_URL" 2>&1 \
                        | grep -i "Available versions" \
                        | sed 's/+[a-zA-Z0-9]*//g; s/Available versions: //' \
                        | tr ',' '\n' | sed 's/[[:space:]]//g; /^$/d' \
                        | head -10 | sed "s/^/        /" || true
                fi
            else
                fail "torch==${TORCH_VERSION} — not found on $INDEX_URL"
                grep -iE "error|could not find|no matching" "$_tmp" \
                    | head -3 | sed "s/^/        /" >&2 || true
                chk "      Available on this index:"
                $PPIP index versions torch --index-url "$INDEX_URL" 2>&1 \
                    | grep -i "Available versions" \
                    | sed 's/+[a-zA-Z0-9]*//g; s/Available versions: //' \
                    | tr ',' '\n' | sed 's/[[:space:]]//g; /^$/d' \
                    | head -10 | sed "s/^/        /" || true
            fi
        fi
        rm -f "$_tmp"
    else
        # No version pinned — verify the index is reachable and has torch
        probe_pkg "torch (latest) + torchvision  [${WHEEL_TAG} index]" \
            torch torchvision --index-url "$INDEX_URL"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
hdr "torch-harmonics"
# ══════════════════════════════════════════════════════════════════════════════
probe_pkg "torch-harmonics (PyPI)" torch-harmonics

# ══════════════════════════════════════════════════════════════════════════════
hdr "Platform requirements file"
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$PLATFORM" == "linux" ]]; then
    probe_req "requirements-linux.txt" "$REPO_ROOT/requirements-linux.txt"
elif [[ "$PLATFORM" == "mac" ]]; then
    # mac requirements already covered in torch step when no --torch pin;
    # probe base separately for pinned-torch case
    if [[ -n "$TORCH_VERSION" ]]; then
        probe_req "requirements-base.txt" "$REPO_ROOT/requirements-base.txt"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
if [[ "$INSTALL_S3" == true ]]; then
    hdr "S3 requirements"
    probe_pkg "s3fs"              s3fs
    probe_req "requirements-s3.txt" "$REPO_ROOT/requirements-s3.txt"
fi

if [[ "$INSTALL_GCS" == true ]]; then
    hdr "GCS requirements"
    probe_pkg "gcsfs"              gcsfs
    probe_req "requirements-gcs.txt" "$REPO_ROOT/requirements-gcs.txt"
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

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════"
if [[ "$FAILURES" -eq 0 ]]; then
    _CMD="bash install_full.sh"
    [[ "$ENV_TYPE" == conda ]] && _CMD+=" --conda" || _CMD+=" --venv"
    _CMD+=" --name $ENV_NAME"
    [[ -n "$TORCH_VERSION" ]]   && _CMD+=" --torch $TORCH_VERSION"
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
