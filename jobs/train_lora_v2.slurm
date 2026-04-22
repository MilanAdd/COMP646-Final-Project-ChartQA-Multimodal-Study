#!/bin/bash
#SBATCH --job-name=chartqa-lora-v2
#SBATCH --partition=commons
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:lovelace:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/__NETID__/chartqa_project/logs/train_lora_v2_%j.out
#SBATCH --error=/home/__NETID__/chartqa_project/logs/train_lora_v2_%j.err

echo "Job started on $(hostname) at $(date)"
nvidia-smi

module purge

PYTHON=/scratch/ma200/venvs/chartqa/bin/python
$PYTHON -m pip install --quiet matplotlib

echo "Python:    $($PYTHON --version)"
echo "PyTorch:   $($PYTHON -c 'import torch; print(torch.__version__)')"
echo "CUDA:      $($PYTHON -c 'import torch; print(torch.cuda.is_available())')"

export HF_HOME=/scratch/ma200/hf_cache
export HF_DATASETS_CACHE=/scratch/ma200/hf_cache/datasets
export CHARTQA_DATA_DIR=/scratch/ma200/chartqa_data
export CHARTQA_OUTPUT_DIR=/scratch/ma200/chartqa_output_v2

mkdir -p /scratch/ma200/chartqa_output_v2/checkpoints
mkdir -p /scratch/ma200/chartqa_output_v2/results

cd /home/__NETID__/chartqa_project

# LoRA v2: r=32, alpha=64, q+k+v+out_proj, lr=1e-4
srun $PYTHON train.py \
    --mode lora \
    --epochs 20 \
    --lr 1e-4 \
    --batch-size 64 \
    --workers 4

echo "Copying results to WORK ..."
mkdir -p /storage/hpc/work/comp646/ma200/chartqa_project/checkpoints_v2
mkdir -p /storage/hpc/work/comp646/ma200/chartqa_project/results_v2
cp /scratch/ma200/chartqa_output_v2/checkpoints/best_lora.pt \
   /storage/hpc/work/comp646/ma200/chartqa_project/checkpoints_v2/best_lora_v2.pt 2>/dev/null || true
cp /scratch/ma200/chartqa_output_v2/results/history_lora.json \
   /storage/hpc/work/comp646/ma200/chartqa_project/results_v2/history_lora_v2.json 2>/dev/null || true

echo "Job finished at $(date)"
