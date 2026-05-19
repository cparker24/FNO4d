#!/usr/bin/env bash
# run.sh — activate a conda or venv environment and start the Panel app.
#
# Environment selection (pick one):
#
#   Conda  — set CONDA_ENV_NAME (default: fno_env_mlx)
#              CONDA_ENV_NAME=fno_env bash run.sh
#
#   venv   — set VENV_PATH to the venv root directory
#              VENV_PATH=~/my_venv bash run.sh
#              VENV_PATH=.venv bash run.sh
#
#   None   — if panel is already on PATH (env pre-activated), skip detection
#              bash run.sh --show
#
# Other options:
#   PORT=8080 bash run.sh            custom port (default 5006)
#   bash run.sh --show               open browser tab automatically
#   bash run.sh --port 8080 --show   panel serve pass-through args
#   EVAL_GCS_CONFIG=/path/to/config.yaml bash run.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Defaults — override via environment variables
# ---------------------------------------------------------------------------
CONDA_ENV_NAME="${CONDA_ENV_NAME:-fno_env_mlx}"
VENV_PATH="${VENV_PATH:-}"
PORT="${PORT:-5006}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# Environment activation
# ---------------------------------------------------------------------------

if [[ -n "$VENV_PATH" ]]; then
    # ── venv ────────────────────────────────────────────────────────────────
    VENV_PATH="${VENV_PATH/#\~/$HOME}"   # expand leading ~
    ACTIVATE="$VENV_PATH/bin/activate"

    if [[ ! -f "$ACTIVATE" ]]; then
        red "ERROR: venv activate script not found at $ACTIVATE"
        red "Check VENV_PATH=$VENV_PATH"
        exit 1
    fi

    # shellcheck disable=SC1090
    source "$ACTIVATE"
    green "venv activated: $VENV_PATH"

elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    # ── venv already active ─────────────────────────────────────────────────
    green "venv already active: $VIRTUAL_ENV"

elif [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV_NAME" ]]; then
    # ── conda already active ─────────────────────────────────────────────────
    green "conda env already active: $CONDA_DEFAULT_ENV"

else
    # ── try to activate conda ─────────────────────────────────────────────
    yellow "No env active — attempting conda activate '$CONDA_ENV_NAME' …"

    CONDA_BASE=$(conda info --base 2>/dev/null || true)

    if [[ -z "$CONDA_BASE" ]]; then
        # conda not on PATH — check common install locations
        for _candidate in \
            "$HOME/miniconda3" "$HOME/anaconda3" \
            "$HOME/opt/miniconda3" "$HOME/opt/anaconda3" \
            "/opt/homebrew/Caskroom/miniconda/base" \
            "/opt/miniconda3" "/opt/anaconda3"
        do
            if [[ -f "$_candidate/etc/profile.d/conda.sh" ]]; then
                CONDA_BASE="$_candidate"
                break
            fi
        done
    fi

    if [[ -z "$CONDA_BASE" ]]; then
        red "ERROR: conda not found and VENV_PATH not set."
        red ""
        red "Options:"
        red "  1. Activate your env manually, then re-run:"
        red "       conda activate $CONDA_ENV_NAME  &&  bash run.sh"
        red "       source /path/to/venv/bin/activate  &&  bash run.sh"
        red ""
        red "  2. Point run.sh at your venv:"
        red "       VENV_PATH=/path/to/venv bash run.sh"
        red ""
        red "  3. Point run.sh at a different conda env:"
        red "       CONDA_ENV_NAME=my_env bash run.sh"
        exit 1
    fi

    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"

    if ! conda activate "$CONDA_ENV_NAME" 2>/dev/null; then
        red "ERROR: conda env '$CONDA_ENV_NAME' not found."
        red ""
        red "Available envs:"
        conda env list 2>/dev/null | grep -v "^#" || true
        red ""
        red "Set the correct name with:  CONDA_ENV_NAME=<name> bash run.sh"
        red "Or use a venv:              VENV_PATH=/path/to/venv bash run.sh"
        exit 1
    fi

    green "conda env activated: $CONDA_DEFAULT_ENV"
fi

# ---------------------------------------------------------------------------
# Show which Python is being used
# ---------------------------------------------------------------------------
PYTHON=$(command -v python || command -v python3)
yellow "Python : $("$PYTHON" -c 'import sys; print(sys.executable)')"

# ---------------------------------------------------------------------------
# Verify panel is available
# ---------------------------------------------------------------------------
if ! "$PYTHON" -c "import panel" 2>/dev/null; then
    red "ERROR: panel not found in the active environment."
    red "Run:  bash $SCRIPT_DIR/install.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
green ""
green "Starting Panel app …  http://localhost:$PORT/app"
green "Press Ctrl-C to stop."
echo ""

exec panel serve "$SCRIPT_DIR/app.py" \
    --port "$PORT" \
    --allow-websocket-origin "*" \
    "$@"
