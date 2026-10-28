#!/usr/bin/env bash
# launch_lambda_retrain.sh
#
# Run from YOUR OWN machine (not the Lambda instance) once a Lambda Cloud GPU
# instance is up. Stages the GCS service-account key onto the instance via
# scp, then starts setup_and_train_lambda.sh there inside a detached tmux
# session (so training survives you disconnecting), without needing to
# manually ssh in first.
#
# Usage:
#   ./launch_lambda_retrain.sh <instance-ip> [train-config] [ssh-user] [gcs-key-path]
#
#   instance-ip   required. The Lambda instance's public IP.
#   train-config  optional. Defaults to config_lambda_32step_tau120_maskedloss.yaml
#                 (the tail-masked-loss retrain -- see
#                 js-contrib contribs/FnoHydro/PARTICLIZATION_INVESTIGATION.md's
#                 "Root cause of the yield deficit" section).
#   ssh-user      optional. Defaults to ubuntu (Lambda's default image user).
#   gcs-key-path  optional. Defaults to train_gcs/fnotest-wayne-gcs.json next
#                 to this script.
#
# Example:
#   ./launch_lambda_retrain.sh 123.45.67.89
#   ./launch_lambda_retrain.sh 123.45.67.89 config_lambda_32step_tau120.yaml
#
# What it does, concretely:
#   1. scp the GCS key to ~/fnotest-wayne-gcs.json on the instance.
#   2. ssh in and curl this repo's setup_and_train_lambda.sh (from the
#      cparker/FNO-3D branch on the GitHub mirror -- make sure your latest
#      commits are pushed there first: `git push github cparker/FNO-3D`).
#   3. Start it inside a detached `tmux` session named fno_train, with
#      TRAIN_CONFIG set to the chosen config, logging to ~/lambda_train.log
#      in addition to tmux's own scrollback.
#
# After this returns, training is running on the instance whether or not you
# stay connected. Check on it with:
#   ssh <ssh-user>@<instance-ip> tmux attach -t fno_train      # live view
#   ssh <ssh-user>@<instance-ip> tail -f lambda_train.log      # just the log

set -euo pipefail

log()  { echo "[launch_lambda_retrain] $*"; }
die()  { echo "[launch_lambda_retrain] ERROR: $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INSTANCE_IP="${1:-}"
TRAIN_CONFIG="${2:-config_lambda_32step_tau120_maskedloss.yaml}"
SSH_USER="${3:-ubuntu}"
GCS_KEY_PATH="${4:-$SCRIPT_DIR/../train_gcs/fnotest-wayne-gcs.json}"

RAW_SETUP_SCRIPT_URL="https://raw.githubusercontent.com/cparker24/FNO4d/cparker/FNO-3D/train_torch/setup_and_train_lambda.sh"

[ -n "$INSTANCE_IP" ] || die "usage: $0 <instance-ip> [train-config] [ssh-user] [gcs-key-path]"
[ -f "$GCS_KEY_PATH" ] || die "GCS key not found at $GCS_KEY_PATH -- pass its path as the 4th argument."

SSH_TARGET="${SSH_USER}@${INSTANCE_IP}"

log "Waiting for SSH on $SSH_TARGET ..."
for _ in $(seq 1 30); do
    ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "$SSH_TARGET" true 2>/dev/null && break
    sleep 5
done
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "$SSH_TARGET" true \
    || die "could not SSH to $SSH_TARGET after 150s -- check the instance is up and your SSH key is registered with it."

log "Staging GCS key ($GCS_KEY_PATH) -> ${SSH_TARGET}:~/fnotest-wayne-gcs.json ..."
scp -o StrictHostKeyChecking=accept-new "$GCS_KEY_PATH" "${SSH_TARGET}:~/fnotest-wayne-gcs.json"

log "Fetching setup_and_train_lambda.sh and starting training in a detached tmux session (fno_train) ..."
log "  train config: $TRAIN_CONFIG"

# Built locally (unquoted heredoc) so $RAW_SETUP_SCRIPT_URL / $TRAIN_CONFIG are
# substituted here, then sent as one literal remote command -- nothing in it
# is meant to be evaluated by the remote shell.
REMOTE_CMD=$(cat <<EOF
set -e
curl -fsSL -o ~/setup_and_train_lambda.sh "$RAW_SETUP_SCRIPT_URL"
chmod +x ~/setup_and_train_lambda.sh
tmux new-session -d -s fno_train \
  "cd ~ && TRAIN_CONFIG=$TRAIN_CONFIG bash setup_and_train_lambda.sh 2>&1 | tee ~/lambda_train.log"
echo "Started. tmux session: fno_train"
EOF
)
ssh "$SSH_TARGET" "$REMOTE_CMD"

log "Done. Training is running in the background on $INSTANCE_IP."
log "Check on it with:"
log "  ssh $SSH_TARGET tmux attach -t fno_train      # live view (Ctrl-b d to detach)"
log "  ssh $SSH_TARGET tail -f lambda_train.log       # just the log"
