#!/bin/bash
#SBATCH --job-name=chartqa-lora
#SBATCH --partition=scavenge
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=__WORK__/chartqa_project/logs/train_lora_%j.out
#SBATCH --error=__WORK__/chartqa_project/logs/train_lora_%j.err

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
# HuggingFace cache
# ---------------------------------------------------------------------------
export HF_HOME=__WORK__/hf_cache
export HF_DATASETS_CACHE=__WORK__/hf_cache/datasets
mkdir -p "$HF_HOME"

# ---------------------------------------------------------------------------
# Stage data to SHARED_SCRATCH
# ---------------------------------------------------------------------------
echo "Staging data to SHARED_SCRATCH ..."
SCRATCH_DATA=$SHARED_SCRATCH/chartqa_data
mkdir -p "$SCRATCH_DATA"

if [ ! -d "$SCRATCH_DATA/datasets" ]; then
    cp -r __WORK__/chartqa_data/ "$SCRATCH_DATA/"
    echo "Data staged."
else
    echo "Data already on scratch, skipping copy."
fi

# ---------------------------------------------------------------------------
# Run training — LoRA CLIP visual encoder + MLP head
# LoRA adds trainable parameters so this job needs slightly more time
# than frozen CLIP.
# ---------------------------------------------------------------------------
PROJECT_DIR=__WORK__/chartqa_project
cd "$PROJECT_DIR"

export WORK=$SCRATCH_DATA

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/checkpoints"
mkdir -p "$PROJECT_DIR/results"

srun python train.py \
    --mode lora \
    --epochs 20 \
    --lr 3e-4 \
    --batch-size 64 \
    --workers 4

# ---------------------------------------------------------------------------
# Copy results back to $WORK
# ---------------------------------------------------------------------------
echo "Copying results back to WORK ..."
cp -r "$PROJECT_DIR/checkpoints/" __WORK__/chartqa_project/
cp -r "$PROJECT_DIR/results/"     __WORK__/chartqa_project/

echo "Job finished at $(date)"
