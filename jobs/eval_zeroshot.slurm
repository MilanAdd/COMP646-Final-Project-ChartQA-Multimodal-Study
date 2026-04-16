#!/bin/bash
#SBATCH --job-name=chartqa-zeroshot
#SBATCH --partition=scavenge
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=__WORK__/chartqa_project/logs/eval_zeroshot_%j.out
#SBATCH --error=__WORK__/chartqa_project/logs/eval_zeroshot_%j.err

# Note: Qwen2.5-VL-7B needs ~14GB GPU memory in bfloat16.
# A single A100 (40GB) is more than sufficient.
# If only smaller GPUs are available, add: #SBATCH --gres=gpu:a100:1

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
    pip install --no-cache-dir transformers datasets accelerate Pillow qwen-vl-utils
else
    source "$VENV_DIR/bin/activate"
    # Ensure qwen-vl-utils is installed (may not be in earlier venv)
    pip install --no-cache-dir --quiet qwen-vl-utils
fi

echo "Python:       $(which python)"
echo "PyTorch:      $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:         $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "transformers: $(python -c 'import transformers; print(transformers.__version__)')"

# ---------------------------------------------------------------------------
# HuggingFace cache — Qwen2.5-VL-7B is ~15GB, must live in $WORK
# ---------------------------------------------------------------------------
export HF_HOME=__WORK__/hf_cache
export HF_DATASETS_CACHE=__WORK__/hf_cache/datasets
mkdir -p "$HF_HOME"

# ---------------------------------------------------------------------------
# Stage ChartQA data to SHARED_SCRATCH
# The model weights stay in $WORK (too large to copy to scratch)
# ---------------------------------------------------------------------------
echo "Staging dataset to SHARED_SCRATCH ..."
SCRATCH_DATA=$SHARED_SCRATCH/chartqa_data
mkdir -p "$SCRATCH_DATA"

if [ ! -d "$SCRATCH_DATA/datasets" ]; then
    cp -r __WORK__/chartqa_data/ "$SCRATCH_DATA/"
    echo "Dataset staged."
else
    echo "Dataset already on scratch."
fi

# ---------------------------------------------------------------------------
# Run zero-shot evaluation
# ---------------------------------------------------------------------------
PROJECT_DIR=__WORK__/chartqa_project
cd "$PROJECT_DIR"

export WORK=$SCRATCH_DATA

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/results"

srun python zero_shot.py \
    --split test \
    --batch-size 1 \
    --max-new-tokens 20

# ---------------------------------------------------------------------------
# Copy results back to $WORK
# ---------------------------------------------------------------------------
echo "Copying results back to WORK ..."
cp -r "$PROJECT_DIR/results/" __WORK__/chartqa_project/

echo "Job finished at $(date)"
