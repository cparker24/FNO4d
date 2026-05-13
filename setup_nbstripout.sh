#!/usr/bin/env bash
# setup_nbstripout.sh — Install nbstripout and register the git filter driver
#
# The repo's .gitattributes already declares "*.ipynb filter=nbstripout", but
# the filter *driver* (the command git actually calls) must be registered once
# per clone in .git/config.  This script handles both steps.
#
# Usage:
#   bash setup_nbstripout.sh            # install for this clone only
#   bash setup_nbstripout.sh --global   # install globally (~/.gitconfig)
#   bash setup_nbstripout.sh --status   # check current status and exit
#   bash setup_nbstripout.sh -h|--help  # show this help

set -euo pipefail

# ── helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[setup_nbstripout] $*"; }
warn() { echo "[setup_nbstripout] WARNING: $*" >&2; }
die()  { echo "[setup_nbstripout] ERROR: $*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── parse flags ───────────────────────────────────────────────────────────────
GLOBAL=false
STATUS_ONLY=false

usage() {
    cat <<EOF
Usage: bash setup_nbstripout.sh [OPTIONS]

Options:
  --global   Register the filter driver in ~/.gitconfig so it is active for
             every git repository on this machine (no need to re-run after
             future clones).
  --status   Print the current nbstripout status for this repo and exit.
  -h, --help Show this help message and exit.

Default (no flags): install nbstripout if missing and register the filter
driver in .git/config for this clone only.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --global)     GLOBAL=true ;;
        --status)     STATUS_ONLY=true ;;
        -h|--help)    usage; exit 0 ;;
        *) die "Unknown argument: '$arg'  (run with --help for usage)" ;;
    esac
done

# ── must be run inside the repo ───────────────────────────────────────────────
if ! git -C "$REPO_ROOT" rev-parse --git-dir &>/dev/null; then
    die "Not inside a git repository. Run this from inside your FNO4d clone."
fi

# ── status-only mode ─────────────────────────────────────────────────────────
if [[ "$STATUS_ONLY" == true ]]; then
    if command -v nbstripout &>/dev/null; then
        nbstripout --status || true
    else
        warn "nbstripout is not installed."
    fi
    exit 0
fi

# ── locate / install nbstripout ───────────────────────────────────────────────
#
# Resolution order:
#   1. Already on PATH — use it directly.
#   2. Importable via the current Python — run via 'python -m nbstripout'.
#   3. Not available — install with pip, then recheck.

find_nbstripout() {
    # Returns the command to invoke nbstripout, or empty string if not found.
    if command -v nbstripout &>/dev/null; then
        echo "nbstripout"
        return
    fi
    if python -m nbstripout --version &>/dev/null 2>&1; then
        echo "python -m nbstripout"
        return
    fi
    if python3 -m nbstripout --version &>/dev/null 2>&1; then
        echo "python3 -m nbstripout"
        return
    fi
    echo ""
}

NBSTRIPOUT_CMD="$(find_nbstripout)"

if [[ -z "$NBSTRIPOUT_CMD" ]]; then
    log "nbstripout not found — installing via pip ..."

    # Prefer the Python that owns the active env; fall back to python3.
    if command -v python &>/dev/null; then
        PIP_CMD="python -m pip"
    else
        PIP_CMD="python3 -m pip"
    fi

    $PIP_CMD install --quiet nbstripout
    log "nbstripout installed."

    # Recheck after install.
    NBSTRIPOUT_CMD="$(find_nbstripout)"
    if [[ -z "$NBSTRIPOUT_CMD" ]]; then
        die "nbstripout was installed but is still not callable. " \
            "Make sure the correct Python environment is active and retry."
    fi
fi

INSTALLED_VERSION=$($NBSTRIPOUT_CMD --version 2>&1 | head -1)
log "Using: $NBSTRIPOUT_CMD  ($INSTALLED_VERSION)"

# ── register the git filter driver ───────────────────────────────────────────
#
# .gitattributes already contains the "*.ipynb filter=nbstripout" lines (they
# are committed to the repo), so we do NOT pass --attributes here — that would
# append duplicate entries to the tracked file.

if [[ "$GLOBAL" == true ]]; then
    log "Registering filter driver in ~/.gitconfig (--global) ..."
    $NBSTRIPOUT_CMD --install --global
    log "Done. The filter is now active for all repositories on this machine."
else
    cd "$REPO_ROOT"
    log "Registering filter driver in .git/config for this clone ..."
    $NBSTRIPOUT_CMD --install
    log "Done. The filter is active for this clone of the repo."
fi

# ── verify ────────────────────────────────────────────────────────────────────
log ""
log "Status check:"
cd "$REPO_ROOT"
$NBSTRIPOUT_CMD --status

log ""
log "All set — notebook outputs will be stripped on 'git commit'."
log "To strip outputs from already-tracked notebooks run:"
log "  git add --renormalize '*.ipynb' && git commit -m 'strip notebook outputs'"
