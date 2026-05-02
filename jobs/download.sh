#!/bin/bash
# Download results, figures, and checkpoints from the cluster to local machine.
#
# Usage:
#   ./jobs/download.sh            — download results + figures only
#   ./jobs/download.sh --all      — also download checkpoints (large)
#
# Requires NETID to be set:
#   export NETID=your_netid

source "$(dirname "$0")/config.sh"

LOCAL_OUTPUT="./nots_output"
mkdir -p "$LOCAL_OUTPUT/results" "$LOCAL_OUTPUT/figures"

echo "Downloading results and figures from cluster..."

scp -r "$REMOTE:${WORK}/chartqa_project/results/" "$LOCAL_OUTPUT/"
scp -r "$REMOTE:${WORK}/chartqa_project/figures/"  "$LOCAL_OUTPUT/"

if [ "$1" = "--all" ]; then
    echo "Downloading checkpoints (this may take a while)..."
    mkdir -p "$LOCAL_OUTPUT/checkpoints"
    scp -r "$REMOTE:${WORK}/chartqa_project/checkpoints/" "$LOCAL_OUTPUT/"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "Downloaded to $LOCAL_OUTPUT/:"
    ls -lh "$LOCAL_OUTPUT/results/" 2>/dev/null
    ls -lh "$LOCAL_OUTPUT/figures/" 2>/dev/null
else
    echo "Download failed. Check if jobs have completed: ./jobs/status.sh <job_name>"
fi
