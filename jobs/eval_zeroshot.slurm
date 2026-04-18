#!/bin/bash
#SBATCH --job-name=chartqa-zeroshot
#SBATCH --partition=commons
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=/home/__NETID__/chartqa_project/logs/eval_zeroshot_%j.out
#SBATCH --error=/home/__NETID__/chartqa_project/logs/eval_zeroshot_%j.err

echo "Job started on $(hostname) at $(date)"
echo "GPU info:"
nvidia-smi

module purge

VENV_DIR=__WORK__/venvs/chartqa
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment ..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install --no-cache-dir transformers datasets accelerate Pillow qwen-vl-utils
else
    source "$VENV_DIR/bin/activate"
    pip install --no-cache-dir --quiet qwen-vl-utils
fi

echo "Python:       $(which python)"
echo "PyTorch:      $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:         $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "transformers: $(python -c 'import transformers; print(transformers.__version__)')"

export HF_HOME=__WORK__/hf_cache
export HF_DATASETS_CACHE=__WORK__/hf_cache/datasets

echo "Staging dataset to SHARED_SCRATCH ..."
SCRATCH_DATA=$SHARED_SCRATCH/chartqa_data
mkdir -p "$SCRATCH_DATA"

if [ ! -d "$SCRATCH_DATA/datasets" ]; then
    cp -r __WORK__/chartqa_data/ "$SCRATCH_DATA/"
    echo "Dataset staged."
else
    echo "Dataset already on scratch."
fi

SCRATCH_OUTPUT=$SHARED_SCRATCH/chartqa_output
mkdir -p "$SCRATCH_OUTPUT/results"

PROJECT_DIR=/home/__NETID__/chartqa_project
cd "$PROJECT_DIR"

export CHARTQA_DATA_DIR=$SCRATCH_DATA
export CHARTQA_OUTPUT_DIR=$SCRATCH_OUTPUT

srun python zero_shot.py \
    --split test \
    --batch-size 1 \
    --max-new-tokens 20

echo "Copying results back to WORK ..."
mkdir -p __WORK__/chartqa_project/results
cp -r "$SCRATCH_OUTPUT/results/" __WORK__/chartqa_project/

echo "Job finished at $(date)"
