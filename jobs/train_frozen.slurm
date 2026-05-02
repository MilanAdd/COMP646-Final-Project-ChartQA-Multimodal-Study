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

mkdir -p ${SCRATCH}/chartqa_output/checkpoints
mkdir -p ${SCRATCH}/chartqa_output/results

cd /home/${NETID}/chartqa_project

srun $PYTHON train.py \
    --mode frozen \
    --epochs 20 \
    --lr 3e-4 \
    --batch-size 64 \
    --workers 4

echo "Copying results back to WORK..."
mkdir -p ${WORK}/chartqa_project/checkpoints
mkdir -p ${WORK}/chartqa_project/results
cp -r ${SCRATCH}/chartqa_output/checkpoints/ ${WORK}/chartqa_project/
cp -r ${SCRATCH}/chartqa_output/results/     ${WORK}/chartqa_project/

echo "Job finished at $(date)"
