#!/bin/bash
# Cancel a running ChartQA job on NOTS.
#
# Usage:
#   ./stop.sh train_frozen
#   ./stop.sh train_lora
#   ./stop.sh eval_zeroshot

source "$(dirname "$0")/config.sh"

JOB_NAME="${1:-train_frozen}"
JOB_FILE="$REMOTE_DIR/jobs/.${JOB_NAME}.jobid"

JOB_ID=$(ssh $REMOTE "cat $JOB_FILE 2>/dev/null")

if [ -z "$JOB_ID" ]; then
    echo "No job ID found for: $JOB_NAME. Nothing to cancel."
    exit 1
fi

ssh $REMOTE "scancel $JOB_ID && rm -f $JOB_FILE"
echo "Job $JOB_ID ($JOB_NAME) cancelled."
