#!/usr/bin/env bash
# install.sh — install Panel dependencies for eval_model_web/app.py
#
# Run from anywhere; the script locates itself automatically:
#   bash eval_model_web/install.sh
#   cd eval_model_web && bash install.sh
#
# NOTE: activate your conda/venv environment BEFORE running this script so
# that the correct Python is on PATH.  The script always installs into
# whichever Python 'python' resolves to and uses 'python -m pip' to guarantee
# pip and python are from the same environment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# 0. Show which Python will be used — catches wrong-env mistakes early
# ---------------------------------------------------------------------------
PYTHON=$(command -v python || command -v python3 || true)
if [[ -z "$PYTHON" ]]; then
    red "ERROR: no python found on PATH."
    exit 1
fi

PYTHON_VERSION=$("$PYTHON" --version 2>&1)
PYTHON_PATH=$("$PYTHON" -c "import sys; print(sys.executable)")

yellow "Python : $PYTHON_PATH"
yellow "Version: $PYTHON_VERSION"

# Warn when running outside a conda/venv environment
if [[ -z "${CONDA_DEFAULT_ENV:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
    yellow "WARNING: no conda or venv environment appears to be active."
    yellow "Activate your environment first, e.g.:"
    yellow "  conda activate <env-name>"
    yellow "Then re-run this script."
    read -r -p "Continue anyway? [y/N] " REPLY
    [[ "${REPLY,,}" == "y" ]] || exit 1
else
    green "Active env: ${CONDA_DEFAULT_ENV:-${VIRTUAL_ENV}}"
fi

# ---------------------------------------------------------------------------
# 1. Panel + Bokeh  (python -m pip ensures pip == this python)
# ---------------------------------------------------------------------------
green "Installing Panel stack…"
"$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements-panel.txt"

# ---------------------------------------------------------------------------
# 2. Verify Panel can be imported and report version
# ---------------------------------------------------------------------------
PANEL_VER=$("$PYTHON" -c "import panel; print(panel.__version__)" 2>/dev/null || true)
if [[ -z "$PANEL_VER" ]]; then
    red "ERROR: panel import failed after installation."
    exit 1
fi
green "panel $PANEL_VER — OK"

# ---------------------------------------------------------------------------
# 3. Check that the base FNO4d dependencies are present
# ---------------------------------------------------------------------------
yellow "Checking base FNO4d dependencies…"
MISSING=()
for pkg in numpy matplotlib yaml gcsfs; do
    "$PYTHON" -c "import $pkg" 2>/dev/null || MISSING+=("$pkg")
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    yellow "The following packages are missing — run the top-level install first:"
    for m in "${MISSING[@]}"; do
        echo "  • $m"
    done
    yellow "From the repo root:"
    echo "  $PYTHON -m pip install -r requirements-base.txt"
    echo "  $PYTHON -m pip install -r requirements-gcs.txt   # if using GCS"
else
    green "Base dependencies — OK"
fi

# ---------------------------------------------------------------------------
# 4. Done
# ---------------------------------------------------------------------------
green ""
green "Installation complete.  Start the app with:"
green "  cd $SCRIPT_DIR"
green "  panel serve app.py --show"
green ""
green "Or with a custom config:"
green "  EVAL_GCS_CONFIG=/path/to/eval_gcs_config.yaml panel serve app.py --show"
