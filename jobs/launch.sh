#!/bin/bash
# Upload project code to the cluster and submit a job.
#
# Usage:
#   ./jobs/launch.sh train_frozen     — submit frozen CLIP training job
#   ./jobs/launch.sh train_lora       — submit LoRA training job
#   ./jobs/launch.sh evaluate         — submit evaluation job
#   ./jobs/launch.sh zero_shot        — submit zero-shot eval job
#   ./jobs/launch.sh gradcam          — submit GradCAM job
#   ./jobs/launch.sh train_frozen --fresh   — recreate venv before submitting
#
# Requires NETID to be set:
#   export NETID=your_netid

source "$(dirname "$0")/config.sh"

JOB_NAME="${1:-train_frozen}"
SLURM_FILE="$(dirname "$0")/${JOB_NAME}.slurm"

if [ ! -f "$SLURM_FILE" ]; then
    echo "Error: SLURM script not found: $SLURM_FILE"
    echo "Available jobs: train_frozen, train_lora, evaluate, zero_shot, gradcam"
    exit 1
fi

if [ "$2" = "--fresh" ]; then
    echo "Removing existing venv on cluster..."
    ssh $REMOTE "rm -rf ${SCRATCH}/venvs/chartqa"
fi

echo "Uploading project code to cluster..."
ssh $REMOTE "mkdir -p $REMOTE_DIR"

rsync -avz --exclude="*.pyc" \
           --exclude="__pycache__/" \
           --exclude="data/" \
           --exclude="checkpoints/" \
           --exclude="results/" \
           --exclude="figures/" \
           --exclude=".git/" \
           --exclude="jobs/" \
    "$(dirname "$0")/../" "$REMOTE:$REMOTE_DIR/"

echo "Uploading SLURM script..."
TMPFILE=$(mktemp)
sed "s|__WORK__|${WORK}|g; s|__NETID__|${NETID}|g" "$SLURM_FILE" > "$TMPFILE"
scp "$TMPFILE" "$REMOTE:$REMOTE_DIR/jobs/${JOB_NAME}.slurm"
rm "$TMPFILE"

echo "Submitting $JOB_NAME..."
JOB_ID=$(ssh $REMOTE "cd $REMOTE_DIR && sbatch --parsable jobs/${JOB_NAME}.slurm")

if [ -z "$JOB_ID" ]; then
    echo "Failed to submit job. Check SLURM output for errors."
    exit 1
fi

ssh $REMOTE "echo $JOB_ID > $REMOTE_DIR/jobs/.${JOB_NAME}.jobid"

echo ""
echo "Job submitted:  $JOB_ID ($JOB_NAME)"
echo "Run ./jobs/status.sh $JOB_NAME to check progress."
