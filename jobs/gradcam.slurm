#!/bin/bash
#SBATCH --job-name=chartqa-gradcam
#SBATCH --partition=commons
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/__NETID__/chartqa_project/logs/gradcam_%j.out
#SBATCH --error=/home/__NETID__/chartqa_project/logs/gradcam_%j.err

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

mkdir -p ${SCRATCH}/chartqa_output/figures

echo "Installing matplotlib..."
$PYTHON -m pip install --quiet matplotlib

# ---------------------------------------------------------------------------
# GradCAM combined figure for LoRA CLIP (3 correct + 3 incorrect)
# ---------------------------------------------------------------------------
echo "Running GradCAM for LoRA CLIP..."
srun $PYTHON gradcam.py \
    --checkpoint ${SCRATCH}/chartqa_output/checkpoints/best_lora.pt \
    --mode lora \
    --eval-results ${SCRATCH}/chartqa_output/results/eval_lora_test.json \
    --n-correct 3 \
    --n-incorrect 3 \
    --alpha 0.5

# ---------------------------------------------------------------------------
# Copy figures back to WORK
# ---------------------------------------------------------------------------
echo "Copying figures to WORK..."
mkdir -p ${WORK}/chartqa_project/figures 2>/dev/null || true
cp ${SCRATCH}/chartqa_output/figures/gradcam_*.pdf \
   ${WORK}/chartqa_project/figures/ 2>/dev/null || true

echo "Job finished at $(date)"
