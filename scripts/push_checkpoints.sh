#!/bin/bash
# Runs alongside training and pushes every finished checkpoint whose step is a multiple of KEEP_PERIOD to the Hub,
# train_state included, so training can be resumed elsewhere if the host is lost (~50 GB per step).
# It only reads the checkpoint dir, so it can't interfere with saving:
#   - orbax writes into <step>.orbax-checkpoint-tmp-* and renames it to <step> only once the save is complete
#   - steps that are multiples of the config's keep_period are never deleted by orbax, so they can't vanish mid-upload
# A step only counts as pushed once check_pushed_checkpoint.py confirms every file is on the Hub with the right size;
# pushed steps are remembered in $HOME/.pushed_ckpts, so the loop can be restarted at any time.
#
# Usage: scripts/push_checkpoints.sh <checkpoint_dir> <hf_repo> [keep_period=5000] [last_step]
#   e.g. scripts/push_checkpoints.sh checkpoints/pi05_kinova_clutter_rtc_v2/clutter_rtc_v2_run01 \
#            FilippoGorini/pi05_kinova_clutter_rtc_v2 5000 55000
# KEEP_PERIOD must match the config's keep_period. With last_step set, the loop exits after pushing it.
set -u

CKPT_DIR=$1
REPO=$2
KEEP_PERIOD=${3:-5000}
LAST_STEP=${4:-}
INTERVAL=${INTERVAL:-300}
OPENPI_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DONE_DIR="$HOME/.pushed_ckpts/${REPO//\//_}"
mkdir -p "$DONE_DIR"

while true; do
    for step in $(ls "$CKPT_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n); do
        (( step > 0 && step % KEEP_PERIOD == 0 )) || continue
        [[ -e "$DONE_DIR/$step" ]] && continue
        echo "$(date '+%F %T') pushing step $step"
        # lowest CPU/IO priority so the data loader keeps its share; timeout (then kill) so a stuck connection gets retried
        if ! nice -n 19 ionice -c3 timeout -k 5m 6h hf upload "$REPO" "$CKPT_DIR/$step" "$step" \
            --commit-message "ckpt $step"; then
            echo "$(date '+%F %T') upload of step $step failed, retrying in ${INTERVAL}s"
        elif ! timeout -k 1m 10m uv run --project "$OPENPI_DIR" python "$OPENPI_DIR/scripts/check_pushed_checkpoint.py" \
            "$REPO" "$CKPT_DIR/$step" "$step"; then
            echo "$(date '+%F %T') step $step incomplete on the Hub, retrying in ${INTERVAL}s"
        else
            touch "$DONE_DIR/$step"
            echo "$(date '+%F %T') pushed and verified step $step"
        fi
    done
    if [[ -n "$LAST_STEP" && -e "$DONE_DIR/$LAST_STEP" ]]; then
        echo "last step $LAST_STEP pushed, exiting"
        exit 0
    fi
    sleep "$INTERVAL"
done
