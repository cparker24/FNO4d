#!/usr/bin/env bash
# install_kernel.sh — Register the active virtualenv or conda env as a Jupyter kernel.
#
# The kernel name is derived automatically from standard environment variables:
#
#   Conda    → $CONDA_DEFAULT_ENV
#   Virtualenv → basename of $VIRTUAL_ENV
#
# Override any derived value with the KERNEL_NAME environment variable:
#   KERNEL_NAME=my_name bash install_kernel.sh
#
# Usage:
#   bash install_kernel.sh                  # auto-detect name
#   KERNEL_NAME=fno4d bash install_kernel.sh  # explicit name
#   KERNEL_DISPLAY_NAME="FNO4d (MLX)" bash install_kernel.sh
#
# The kernel is registered for the current user (--user flag), so it is
# visible in JupyterLab / Jupyter Notebook regardless of how they were
# launched.

set -euo pipefail

# ── helpers ───────────────────────────────────────────────────────────────────
BOLD=$'\033[1m'
GREEN=$'\033[32m'
RED=$'\033[31m'
YELLOW=$'\033[33m'
NC=$'\033[0m'

log()  { echo "[install_kernel.sh] $*"; }
ok()   { echo "[install_kernel.sh] ${GREEN}OK${NC}  $*"; }
warn() { echo "[install_kernel.sh] ${YELLOW}WARN${NC} $*" >&2; }
die()  { echo "[install_kernel.sh] ${RED}ERROR${NC} $*" >&2; exit 1; }

# ── usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: bash install_kernel.sh [-h|--help]

Environment variables (all optional — values are auto-detected):
  KERNEL_NAME          Kernel identifier used internally by Jupyter.
                       Defaults to \$CONDA_DEFAULT_ENV or basename(\$VIRTUAL_ENV).
  KERNEL_DISPLAY_NAME  Human-readable name shown in the Jupyter UI.
                       Defaults to "<KERNEL_NAME> (Python X.Y)".

Examples:
  bash install_kernel.sh
  KERNEL_NAME=fno4d_cu124 bash install_kernel.sh
  KERNEL_DISPLAY_NAME="FNO4d CUDA 12.4" bash install_kernel.sh
EOF
}

for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $arg  (run 'bash install_kernel.sh --help' for usage)" ;;
    esac
done

# ── detect active environment type ────────────────────────────────────────────
ENV_TYPE=""
DETECTED_NAME=""

if [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-}" != "base" && -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    ENV_TYPE="conda"
    DETECTED_NAME="${CONDA_DEFAULT_ENV}"
    log "Detected conda env: ${CONDA_DEFAULT_ENV}  (prefix: ${CONDA_PREFIX})"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    ENV_TYPE="venv"
    DETECTED_NAME="$(basename "${VIRTUAL_ENV}")"
    log "Detected virtualenv: ${VIRTUAL_ENV}"
else
    die "No active conda environment or virtualenv found.
  Activate one first, e.g.:
    conda activate fno4d
    source venv/bin/activate"
fi

# ── resolve kernel name ───────────────────────────────────────────────────────
# Prefer the explicit KERNEL_NAME env var; fall back to the detected name.
KERNEL_NAME="${KERNEL_NAME:-${DETECTED_NAME}}"

# Sanitise: Jupyter kernel names must be alphanumeric + underscores/hyphens/dots,
# so replace spaces and special characters with underscores.
KERNEL_NAME_SAFE="$(echo "${KERNEL_NAME}" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_//;s/_$//')"
if [[ "${KERNEL_NAME_SAFE}" != "${KERNEL_NAME}" ]]; then
    warn "Kernel name '${KERNEL_NAME}' contained invalid characters — using '${KERNEL_NAME_SAFE}' instead."
    KERNEL_NAME="${KERNEL_NAME_SAFE}"
fi

# ── resolve Python version for the display name ───────────────────────────────
PYTHON_VER="$(python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')" || PYTHON_VER=""
PY_SUFFIX="${PYTHON_VER:+ (Python ${PYTHON_VER})}"

# Display name: explicit KERNEL_DISPLAY_NAME > auto-generated fallback
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-${KERNEL_NAME}${PY_SUFFIX}}"

log "Kernel name    : ${KERNEL_NAME}"
log "Display name   : ${KERNEL_DISPLAY_NAME}"
log "Python binary  : $(command -v python)"

# ── ensure ipykernel is installed ─────────────────────────────────────────────
if ! python -c "import ipykernel" 2>/dev/null; then
    log "ipykernel not found — installing ..."
    python -m pip install --quiet ipykernel
    ok "ipykernel installed."
else
    log "ipykernel already installed."
fi

# ── register the kernel ───────────────────────────────────────────────────────
log "Registering kernel ..."
python -m ipykernel install \
    --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"

# ── confirm ───────────────────────────────────────────────────────────────────
ok "Kernel '${KERNEL_NAME}' registered successfully."
echo
echo "  ${BOLD}Display name${NC}: ${KERNEL_DISPLAY_NAME}"
echo "  ${BOLD}Kernel name${NC} : ${KERNEL_NAME}"
echo "  ${BOLD}Env type${NC}    : ${ENV_TYPE}"
echo
echo "To list all installed kernels:"
echo "  jupyter kernelspec list"
echo
echo "To remove this kernel later:"
echo "  jupyter kernelspec remove ${KERNEL_NAME}"
