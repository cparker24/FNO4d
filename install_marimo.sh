#!/usr/bin/env bash
# install_marimo.sh — Optional: install marimo into the active environment.
#
# Marimo is the interactive notebook runner used by the eval scripts
# (eval_model_test_*.py).  It is intentionally excluded from the base
# requirements so that headless / batch setups are not forced to carry the
# dependency.  Run this script after install.sh when you want the interactive UI.
#
# Usage:
#   bash install_marimo.sh          # install marimo into the active env
#   bash install_marimo.sh --help

set -euo pipefail

log()  { echo "[install_marimo.sh] $*"; }
die()  { echo "[install_marimo.sh] ERROR: $*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: bash install_marimo.sh [-h|--help]

Installs marimo into the currently active virtualenv or conda environment.
Run install.sh first to set up the base FNO4d environment, then activate it
before calling this script:

  source venv/bin/activate          # virtualenv
  # or: conda activate <env>

  bash install_marimo.sh
EOF
}

for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        *) die "Unknown argument: $arg  (run 'bash install_marimo.sh --help' for usage)" ;;
    esac
done

# ── require an active environment ────────────────────────────────────────────
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    log "Active virtualenv : $VIRTUAL_ENV"
elif [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-}" != "base" ]]; then
    log "Active conda env  : ${CONDA_DEFAULT_ENV}  ($CONDA_PREFIX)"
else
    die "No active virtualenv or named conda environment found.
  Activate the FNO4d environment first, e.g.:
    source venv/bin/activate
    # or: conda activate <env>"
fi

# ── install marimo ────────────────────────────────────────────────────────────
if python -c "import marimo" 2>/dev/null; then
    MARIMO_VER=$(python -c "import marimo; print(marimo.__version__)")
    log "marimo $MARIMO_VER is already installed — nothing to do."
else
    log "Installing marimo ..."
    python -m pip install marimo
fi

# ── verify ────────────────────────────────────────────────────────────────────
MARIMO_VER=$(python -c "import marimo; print(marimo.__version__)")
log "marimo $MARIMO_VER installed at $(command -v marimo)."
echo
echo "Run an eval notebook:"
echo "  cd train/"
echo "  marimo run eval_model_test_torch.py"
echo "  marimo run eval_model_test_cont_torch.py"
echo
echo "Edit a notebook with live reactivity:"
echo "  marimo edit eval_model_test_cont_torch.py"
