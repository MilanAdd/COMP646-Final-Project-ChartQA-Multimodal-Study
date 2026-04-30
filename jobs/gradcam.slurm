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

PYTHON=/scratch/ma200/venvs/chartqa/bin/python

echo "Python:    $($PYTHON --version)"
echo "PyTorch:   $($PYTHON -c 'import torch; print(torch.__version__)')"
echo "CUDA:      $($PYTHON -c 'import torch; print(torch.cuda.is_available())')"

export HF_HOME=/scratch/ma200/hf_cache
export HF_DATASETS_CACHE=/scratch/ma200/hf_cache/datasets
export CHARTQA_DATA_DIR=/scratch/ma200/chartqa_data
export CHARTQA_OUTPUT_DIR=/scratch/ma200/chartqa_output

mkdir -p /scratch/ma200/chartqa_output/figures

echo "Installing matplotlib..."
$PYTHON -m pip install --quiet matplotlib

# ---------------------------------------------------------------------------
# GradCAM for frozen CLIP
# ---------------------------------------------------------------------------
#echo "Running GradCAM for frozen CLIP..."
#srun $PYTHON gradcam.py \
#    --checkpoint /scratch/ma200/chartqa_output/checkpoints/best_frozen.pt \
#    --mode frozen \
#    --eval-results /scratch/ma200/chartqa_output/results/eval_frozen_test.json \
#    --n-correct 3 \
#    --n-incorrect 3 \
#    --alpha 0.5

# ---------------------------------------------------------------------------
# GradCAM for LoRA CLIP
# ---------------------------------------------------------------------------
#echo "Running GradCAM for LoRA CLIP..."
#srun $PYTHON gradcam.py \
#    --checkpoint /scratch/ma200/chartqa_output/checkpoints/best_lora.pt \
#    --mode lora \
#    --eval-results /scratch/ma200/chartqa_output/results/eval_lora_test.json \
#    --n-correct 3 \
#    --n-incorrect 3 \
#    --alpha 0.5

# ---------------------------------------------------------------------------
# GradCAM combined figure for LoRA CLIP (2 correct + 2 incorrect)
# ---------------------------------------------------------------------------
echo "Running GradCAM for LoRA CLIP..."
srun $PYTHON gradcam.py \
    --checkpoint /scratch/ma200/chartqa_output/checkpoints/best_lora.pt \
    --mode lora \
    --eval-results /scratch/ma200/chartqa_output/results/eval_lora_test.json \
    --n-correct 3 \
    --n-incorrect 3 \
    --alpha 0.5

# ---------------------------------------------------------------------------
# Copy figures back to WORK
# ---------------------------------------------------------------------------
echo "Copying figures to WORK..."
mkdir -p /storage/hpc/work/comp646/ma200/chartqa_project/figures 2>/dev/null || true
cp /scratch/ma200/chartqa_output/figures/gradcam_*.pdf \
   /storage/hpc/work/comp646/ma200/chartqa_project/figures/ 2>/dev/null || true

echo "Job finished at $(date)"
