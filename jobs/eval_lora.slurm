#!/bin/bash
#SBATCH --job-name=chartqa-eval-lora
#SBATCH --partition=commons
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/__NETID__/chartqa_project/logs/eval_lora_%j.out
#SBATCH --error=/home/__NETID__/chartqa_project/logs/eval_lora_%j.err

echo "Job started on $(hostname) at $(date)"
nvidia-smi

module purge

NETID=__NETID__
PYTHON=/scratch/${NETID}/venvs/chartqa/bin/python
SCRATCH=/scratch/${NETID}
WORK=/storage/hpc/work/comp646/${NETID}

echo "Python:    $($PYTHON --version)"
echo "PyTorch:   $($PYTHON -c 'import torch; print(torch.__version__)')"
echo "CUDA:      $($PYTHON -c 'import torch; print(torch.cuda.is_available())')"

export HF_HOME=${SCRATCH}/hf_cache
export HF_DATASETS_CACHE=${SCRATCH}/hf_cache/datasets
export CHARTQA_DATA_DIR=${SCRATCH}/chartqa_data
export CHARTQA_OUTPUT_DIR=${SCRATCH}/chartqa_output

mkdir -p ${SCRATCH}/chartqa_output/results

cd /home/${NETID}/chartqa_project

srun $PYTHON eval.py \
    --checkpoint ${SCRATCH}/chartqa_output/checkpoints/best_lora.pt \
    --mode lora \
    --split test

echo "Copying results back to WORK..."
mkdir -p ${WORK}/chartqa_project/results 2>/dev/null || true
cp ${SCRATCH}/chartqa_output/results/eval_lora_test.json \
   ${WORK}/chartqa_project/results/ 2>/dev/null || true

echo "Job finished at $(date)"
