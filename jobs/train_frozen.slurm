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
nvidia-smi

module purge

# ---------------------------------------------------------------------------
# All paths hardcoded to /scratch — accessible on compute nodes
# ---------------------------------------------------------------------------
PYTHON=/scratch/ma200/venvs/chartqa/bin/python

echo "Python:    $($PYTHON --version)"
echo "PyTorch:   $($PYTHON -c 'import torch; print(torch.__version__)')"
echo "CUDA:      $($PYTHON -c 'import torch; print(torch.cuda.is_available())')"

export HF_HOME=/scratch/ma200/hf_cache
export HF_DATASETS_CACHE=/scratch/ma200/hf_cache/datasets

# Data and output already on scratch — no staging needed
export CHARTQA_DATA_DIR=/scratch/ma200/chartqa_data
export CHARTQA_OUTPUT_DIR=/scratch/ma200/chartqa_output

mkdir -p /scratch/ma200/chartqa_output/checkpoints
mkdir -p /scratch/ma200/chartqa_output/results

# ---------------------------------------------------------------------------
# Run training
# ---------------------------------------------------------------------------
cd /home/__NETID__/chartqa_project

srun $PYTHON train.py \
    --mode frozen \
    --epochs 20 \
    --lr 3e-4 \
    --batch-size 64 \
    --workers 4

# ---------------------------------------------------------------------------
# Copy results back to $WORK for persistence (run from login node after job)
# ---------------------------------------------------------------------------
echo "Copying results back to WORK ..."
mkdir -p /storage/hpc/work/comp646/ma200/chartqa_project/checkpoints
mkdir -p /storage/hpc/work/comp646/ma200/chartqa_project/results
cp -r /scratch/ma200/chartqa_output/checkpoints/ /storage/hpc/work/comp646/ma200/chartqa_project/
cp -r /scratch/ma200/chartqa_output/results/     /storage/hpc/work/comp646/ma200/chartqa_project/

echo "Job finished at $(date)"
