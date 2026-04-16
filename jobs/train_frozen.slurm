#!/bin/bash
#SBATCH --job-name=chartqa-frozen
#SBATCH --partition=scavenge
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=__WORK__/chartqa_project/logs/train_frozen_%j.out
#SBATCH --error=__WORK__/chartqa_project/logs/train_frozen_%j.err

echo "Job started on $(hostname) at $(date)"
echo "GPU info:"
nvidia-smi

module purge

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------
VENV_DIR=__WORK__/venvs/chartqa
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment ..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install --no-cache-dir transformers datasets accelerate Pillow
else
    source "$VENV_DIR/bin/activate"
fi

echo "Python:    $(which python)"
echo "PyTorch:   $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:      $(python -c 'import torch; print(torch.cuda.is_available())')"

# ---------------------------------------------------------------------------
# HuggingFace cache — point to $WORK so models persist across jobs
# ---------------------------------------------------------------------------
export HF_HOME=__WORK__/hf_cache
export HF_DATASETS_CACHE=__WORK__/hf_cache/datasets
mkdir -p "$HF_HOME"

# ---------------------------------------------------------------------------
# Copy data from $WORK to $SHARED_SCRATCH for fast job I/O
# $WORK is not available on compute nodes, so data must be staged first.
# ---------------------------------------------------------------------------
echo "Staging data to SHARED_SCRATCH ..."
SCRATCH_DATA=$SHARED_SCRATCH/chartqa_data
mkdir -p "$SCRATCH_DATA"

# Only copy if not already present (avoids re-copying on resume)
if [ ! -d "$SCRATCH_DATA/datasets" ]; then
    cp -r __WORK__/chartqa_data/ "$SCRATCH_DATA/"
    echo "Data staged."
else
    echo "Data already on scratch, skipping copy."
fi

# ---------------------------------------------------------------------------
# Run training — frozen CLIP (MLP head only)
# ---------------------------------------------------------------------------
PROJECT_DIR=__WORK__/chartqa_project
cd "$PROJECT_DIR"

# Override DATA_DIR to point at the fast scratch copy
export WORK=$SCRATCH_DATA   # dataset.py uses os.environ.get("WORK", ...)

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/checkpoints"
mkdir -p "$PROJECT_DIR/results"

srun python train.py \
    --mode frozen \
    --epochs 20 \
    --lr 3e-4 \
    --batch-size 64 \
    --workers 4

# ---------------------------------------------------------------------------
# Copy results back to $WORK for persistence after job ends
# ---------------------------------------------------------------------------
echo "Copying results back to WORK ..."
cp -r "$PROJECT_DIR/checkpoints/" __WORK__/chartqa_project/
cp -r "$PROJECT_DIR/results/"     __WORK__/chartqa_project/

echo "Job finished at $(date)"
