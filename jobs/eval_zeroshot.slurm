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
nvidia-smi

module purge

PYTHON=/scratch/ma200/venvs/chartqa/bin/python

echo "Python:       $($PYTHON --version)"
echo "PyTorch:      $($PYTHON -c 'import torch; print(torch.__version__)')"
echo "CUDA:         $($PYTHON -c 'import torch; print(torch.cuda.is_available())')"
echo "transformers: $($PYTHON -c 'import transformers; print(transformers.__version__)')"

export HF_HOME=/scratch/ma200/hf_cache
export HF_DATASETS_CACHE=/scratch/ma200/hf_cache/datasets

export CHARTQA_DATA_DIR=/scratch/ma200/chartqa_data
export CHARTQA_OUTPUT_DIR=/scratch/ma200/chartqa_output

mkdir -p /scratch/ma200/chartqa_output/results

cd /home/__NETID__/chartqa_project

srun $PYTHON zero_shot.py \
    --split test \
    --batch-size 1 \
    --max-new-tokens 20

echo "Copying results back to WORK ..."
mkdir -p /storage/hpc/work/comp646/ma200/chartqa_project/results
cp -r /scratch/ma200/chartqa_output/results/ /storage/hpc/work/comp646/ma200/chartqa_project/

echo "Job finished at $(date)"
