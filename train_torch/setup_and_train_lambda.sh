#!/usr/bin/env bash
# setup_and_train_lambda.sh
#
# Clones FNO4d, installs its environment, runs the widened-eta /
# full-tauslice training job (config_lambda_32step_tau120.yaml), and relies
# on train_torch.py's own end-of-run step to upload the trained checkpoint
# and logs back to GCS. Meant for a fresh Lambda Cloud GPU instance (sized
# for a single A100 40GB -- see config_lambda_32step_tau120.yaml's header
# for the memory estimate this targets).
#
# Clones from the GitHub mirror (github.com/cparker24/FNO4d, cparker/FNO-3D
# branch) by default, not the original Codeberg repo -- the mirror already
# has this commit (verified: `git ls-remote github cparker/FNO-3D` matches
# local HEAD). Override FNO4D_REPO_URL below to clone from somewhere else.
#
# Before running this script:
#
#   1. Stage the GCS service-account key on the instance. It is NOT in git
#      and NOT fetched by this script -- copy it over yourself, e.g. from
#      your own machine:
#        scp /home/cparker/jetscape-docker/FNO4d/train_gcs/fnotest-wayne-gcs.json \
#            ubuntu@<lambda-instance-ip>:~/fnotest-wayne-gcs.json
#
#   2. On the instance:
#        bash setup_and_train_lambda.sh
#      Training is long (rough estimate: tens of minutes to a few hours for
#      25 epochs at tauslice=120 on one A100 -- re-check the actual
#      per-epoch time once the first couple of epochs have logged, same as
#      every other run in this investigation). Run this under tmux/screen/
#      nohup so it survives an SSH disconnect:
#        tmux new -s fno_train
#        bash setup_and_train_lambda.sh
#        # Ctrl-b d to detach; tmux attach -t fno_train to check back in.
#
# What it does:
#   - Clones FNO4d (github.com/cparker24/FNO4d, cparker/FNO-3D branch) to
#     ~/FNO4d (skips if already there)
#   - Runs install.sh --gcs (creates venv/, installs base + GCS deps, the
#     vendored neuraloperator fork, and detects/installs a CUDA-matching
#     torch build if one isn't already importable)
#   - Copies the staged GCS key into train_torch/
#   - Runs training with config_lambda_32step_tau120.yaml. Input ROOT files
#     are downloaded from gs://test_fno/training_data/widened_eta_n33/ to
#     ../data/ on first run (cached after that); the trained checkpoint and
#     logs are uploaded back to
#     gs://test_fno/runs/widened_eta_32step_tau120_lambda/{checkpoints,logs}
#     automatically by train_torch.py itself (gcs_output_prefix in the
#     config) -- no separate upload step needed here.
#
# Override any of these via environment variables before running, e.g.
#   FNO4D_BRANCH=some-other-branch GCS_KEY_SRC=~/other-key.json bash setup_and_train_lambda.sh

set -euo pipefail

log()  { echo "[setup_and_train_lambda] $*"; }
die()  { echo "[setup_and_train_lambda] ERROR: $*" >&2; exit 1; }

FNO4D_REPO_URL="${FNO4D_REPO_URL:-https://github.com/cparker24/FNO4d.git}"
FNO4D_BRANCH="${FNO4D_BRANCH:-cparker/FNO-3D}"
FNO4D_DIR="${FNO4D_DIR:-$HOME/FNO4d}"
GCS_KEY_SRC="${GCS_KEY_SRC:-$HOME/fnotest-wayne-gcs.json}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config_lambda_32step_tau120.yaml}"

# ── sanity checks before doing anything slow ────────────────────────────────
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found -- is this a GPU instance?"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

[ -f "$GCS_KEY_SRC" ] || die "GCS key not found at $GCS_KEY_SRC -- scp it here first (see this script's header), or set GCS_KEY_SRC."

# ── clone (or reuse) the repo ────────────────────────────────────────────────
if [ -d "$FNO4D_DIR/.git" ]; then
    log "Reusing existing clone at $FNO4D_DIR (fetching + checking out $FNO4D_BRANCH) ..."
    git -C "$FNO4D_DIR" fetch origin "$FNO4D_BRANCH"
    git -C "$FNO4D_DIR" checkout "$FNO4D_BRANCH"
    git -C "$FNO4D_DIR" reset --hard "origin/$FNO4D_BRANCH"
else
    log "Cloning $FNO4D_REPO_URL ($FNO4D_BRANCH) to $FNO4D_DIR ..."
    git clone --branch "$FNO4D_BRANCH" "$FNO4D_REPO_URL" "$FNO4D_DIR"
fi

cd "$FNO4D_DIR"

# ── environment ─────────────────────────────────────────────────────────────
log "Running install.sh --gcs (venv + base + GCS deps + vendored neuraloperator) ..."
bash install.sh --gcs

# shellcheck disable=SC1091
source venv/bin/activate

log "Sanity check: neuralop/torch import, CUDA visible ..."
python3 -c "import neuralop, torch; print('neuralop OK, torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"

# ── GCS key ──────────────────────────────────────────────────────────────────
cp "$GCS_KEY_SRC" "$FNO4D_DIR/train_torch/fnotest-wayne-gcs.json"
log "Copied GCS key to train_torch/fnotest-wayne-gcs.json"

# ── train ────────────────────────────────────────────────────────────────────
cd "$FNO4D_DIR/train_torch"
log "Starting training with $TRAIN_CONFIG ..."
log "(input ROOT files download from GCS on first run, then train; checkpoint" \
    "+ logs upload back to GCS automatically when training finishes)"
python3 -u train_torch.py --config "$TRAIN_CONFIG"

log "Done. Check the config's gcs_output_prefix in GCS for the uploaded checkpoint + logs:"
grep "gcs_output_prefix" "$TRAIN_CONFIG"
