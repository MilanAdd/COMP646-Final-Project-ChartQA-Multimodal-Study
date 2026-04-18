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

PYTHON=__WORK__/venvs/chartqa/bin/python

echo "Python:       $($PYTHON --version)"
echo "PyTorch:      $($PYTHON -c 'import torch; print(torch.__version__)')"
echo "CUDA:         $($PYTHON -c 'import torch; print(torch.cuda.is_available())')"
echo "transformers: $($PYTHON -c 'import transformers; print(transformers.__version__)')"

export HF_HOME=__WORK__/hf_cache
export HF_DATASETS_CACHE=__WORK__/hf_cache/datasets

echo "Staging dataset to SHARED_SCRATCH ..."
SCRATCH_DATA=$SHARED_SCRATCH/ma200/chartqa_data
mkdir -p "$SCRATCH_DATA"

if [ ! -d "$SCRATCH_DATA/datasets" ]; then
    cp -r __WORK__/chartqa_data/ "$SCRATCH_DATA/"
    echo "Dataset staged."
else
    echo "Dataset already on scratch."
fi

SCRATCH_OUTPUT=$SHARED_SCRATCH/ma200/chartqa_output
mkdir -p "$SCRATCH_OUTPUT/results"

PROJECT_DIR=/home/__NETID__/chartqa_project
cd "$PROJECT_DIR"

export CHARTQA_DATA_DIR=$SCRATCH_DATA
export CHARTQA_OUTPUT_DIR=$SCRATCH_OUTPUT

srun $PYTHON zero_shot.py \
    --split test \
    --batch-size 1 \
    --max-new-tokens 20

echo "Copying results back to WORK ..."
mkdir -p __WORK__/chartqa_project/results
cp -r "$SCRATCH_OUTPUT/results/" __WORK__/chartqa_project/

echo "Job finished at $(date)"
