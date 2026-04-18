#!/bin/bash
#SBATCH --job-name=chartqa-frozen
#SBATCH --partition=commons
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/__NETID__/chartqa_project/logs/train_frozen_%j.out
#SBATCH --error=/home/__NETID__/chartqa_project/logs/train_frozen_%j.err

echo "Job started on $(hostname) at $(date)"
echo "GPU info:"
nvidia-smi

module purge

# ---------------------------------------------------------------------------
# Virtual environment — stored in $WORK (too large for $HOME quota)
# ---------------------------------------------------------------------------
VENV_DIR=__WORK__/venvs/chartqa
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment ..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install --no-cache-dir transformers datasets accelerate Pillow
else
    source "$VENV_DIR/bin/activate"
fi

echo "Python:    $(which python)"
echo "PyTorch:   $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:      $(python -c 'import torch; print(torch.cuda.is_available())')"

# ---------------------------------------------------------------------------
# HuggingFace cache — keep in $WORK so model weights persist across jobs
# ---------------------------------------------------------------------------
export HF_HOME=__WORK__/hf_cache
export HF_DATASETS_CACHE=__WORK__/hf_cache/datasets

# ---------------------------------------------------------------------------
# Stage ChartQA data from $WORK to $SHARED_SCRATCH for fast job I/O
# ---------------------------------------------------------------------------
echo "Staging data to SHARED_SCRATCH ..."
SCRATCH_DATA=$SHARED_SCRATCH/ma200/chartqa_data
mkdir -p "$SCRATCH_DATA"

if [ ! -d "$SCRATCH_DATA/datasets" ]; then
    cp -r __WORK__/chartqa_data/ "$SCRATCH_DATA/"
    echo "Data staged."
else
    echo "Data already on scratch, skipping copy."
fi

SCRATCH_OUTPUT=$SHARED_SCRATCH/ma200/chartqa_output
mkdir -p "$SCRATCH_OUTPUT/checkpoints" "$SCRATCH_OUTPUT/results"

# ---------------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------------
PROJECT_DIR=/home/__NETID__/chartqa_project
cd "$PROJECT_DIR"

export CHARTQA_DATA_DIR=$SCRATCH_DATA
export CHARTQA_OUTPUT_DIR=$SCRATCH_OUTPUT

srun python train.py \
    --mode frozen \
    --epochs 20 \
    --lr 3e-4 \
    --batch-size 64 \
    --workers 4

# ---------------------------------------------------------------------------
# Copy results back to $WORK for persistence
# ---------------------------------------------------------------------------
echo "Copying results back to WORK ..."
mkdir -p __WORK__/chartqa_project/checkpoints
mkdir -p __WORK__/chartqa_project/results
cp -r "$SCRATCH_OUTPUT/checkpoints/" __WORK__/chartqa_project/
cp -r "$SCRATCH_OUTPUT/results/"     __WORK__/chartqa_project/

echo "Job finished at $(date)"
