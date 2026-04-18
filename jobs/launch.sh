#!/bin/bash
# Upload project code to NOTS and submit a training or eval job.
#
# Usage:
#   ./launch.sh train_frozen     — submit frozen CLIP training job
#   ./launch.sh train_lora       — submit LoRA training job
#   ./launch.sh eval_zeroshot    — submit zero-shot eval job
#   ./launch.sh train_frozen --fresh   — recreate venv before submitting

source "$(dirname "$0")/config.sh"

JOB_NAME="${1:-train_frozen}"
SLURM_FILE="$(dirname "$0")/${JOB_NAME}.slurm"

if [ ! -f "$SLURM_FILE" ]; then
    echo "Error: SLURM script not found: $SLURM_FILE"
    echo "Available jobs: train_frozen, train_lora, eval_zeroshot"
    exit 1
fi

if [ "$2" = "--fresh" ]; then
    echo "Removing existing venv on NOTS ..."
    ssh $REMOTE "rm -rf $WORK/venvs/chartqa"
fi

echo "Uploading project code to NOTS ..."
ssh $REMOTE "mkdir -p $REMOTE_DIR"

# Sync all Python source files (exclude data, checkpoints, results, figures)
rsync -avz --exclude="*.pyc" \
           --exclude="__pycache__/" \
           --exclude="data/" \
           --exclude="checkpoints/" \
           --exclude="results/" \
           --exclude="figures/" \
           --exclude=".git/" \
           --exclude="jobs/" \
    "$(dirname "$0")/../" "$REMOTE:$REMOTE_DIR/"

echo "Uploading SLURM script ..."
TMPFILE=$(mktemp)
sed "s|__WORK__|$WORK|g; s|__NETID__|$NETID|g" "$SLURM_FILE" > "$TMPFILE"
scp "$TMPFILE" "$REMOTE:$REMOTE_DIR/jobs/${JOB_NAME}.slurm"
rm "$TMPFILE"

echo "Submitting $JOB_NAME ..."
JOB_ID=$(ssh $REMOTE "cd $REMOTE_DIR && sbatch --parsable jobs/${JOB_NAME}.slurm")

if [ -z "$JOB_ID" ]; then
    echo "Failed to submit job. Check SLURM output for errors."
    exit 1
fi

# Save job ID for status/stop scripts
ssh $REMOTE "echo $JOB_ID > $REMOTE_DIR/jobs/.${JOB_NAME}.jobid"

echo ""
echo "Job submitted:  $JOB_ID ($JOB_NAME)"
echo "Run ./status.sh $JOB_NAME to check progress."
