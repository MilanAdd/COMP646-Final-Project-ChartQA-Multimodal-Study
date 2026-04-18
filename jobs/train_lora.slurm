#!/bin/bash
#SBATCH --job-name=chartqa-lora
#SBATCH --partition=commons
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/__NETID__/chartqa_project/logs/train_lora_%j.out
#SBATCH --error=/home/__NETID__/chartqa_project/logs/train_lora_%j.err

echo "Job started on $(hostname) at $(date)"
nvidia-smi

module purge

SHARED_SCRATCH=/scratch

PYTHON=__WORK__/venvs/chartqa/bin/python

echo "Python:    $($PYTHON --version)"
echo "PyTorch:   $($PYTHON -c 'import torch; print(torch.__version__)')"
echo "CUDA:      $($PYTHON -c 'import torch; print(torch.cuda.is_available())')"

export HF_HOME=__WORK__/hf_cache
export HF_DATASETS_CACHE=__WORK__/hf_cache/datasets

echo "Staging data to SHARED_SCRATCH ..."
SCRATCH_DATA=$SHARED_SCRATCH/ma200/chartqa_data
mkdir -p "$SCRATCH_DATA"

if [ ! -d "$SCRATCH_DATA/datasets" ]; then
    cp -r __WORK__/chartqa_data/ "$SCRATCH_DATA/"
    echo "Data staged."
else
    echo "Data already on scratch."
fi

SCRATCH_OUTPUT=$SHARED_SCRATCH/ma200/chartqa_output
mkdir -p "$SCRATCH_OUTPUT/checkpoints" "$SCRATCH_OUTPUT/results"

PROJECT_DIR=/home/__NETID__/chartqa_project
cd "$PROJECT_DIR"

export CHARTQA_DATA_DIR=$SCRATCH_DATA
export CHARTQA_OUTPUT_DIR=$SCRATCH_OUTPUT

srun $PYTHON train.py \
    --mode lora \
    --epochs 20 \
    --lr 3e-4 \
    --batch-size 64 \
    --workers 4

echo "Copying results back to WORK ..."
mkdir -p __WORK__/chartqa_project/checkpoints
mkdir -p __WORK__/chartqa_project/results
cp -r "$SCRATCH_OUTPUT/checkpoints/" __WORK__/chartqa_project/
cp -r "$SCRATCH_OUTPUT/results/"     __WORK__/chartqa_project/

echo "Job finished at $(date)"
