## What this project does

We study whether photo-pretrained vision encoders can understand charts through the ChartQA benchmark. Three systems are evaluated:

- **Frozen CLIP + MLP** — CLIP ViT-B/32 visual encoder frozen, only MLP head trained
- **LoRA CLIP + MLP** — LoRA adapters (r=8, α=16) injected into CLIP's q_proj and v_proj attention layers
- **Zero-shot Qwen2.5-VL-7B-Instruct** — upper-bound baseline, no task-specific training

Beyond accuracy, GradCAM and Attention Rollout are applied to both trained models to understand *why* they succeed or fail.

**Reproducibility note:** All reported results were obtained on an HPC cluster using a single NVIDIA Tesla V100 or L40S GPU. Set `CHARTQA_DATA_DIR` and `CHARTQA_OUTPUT_DIR` to appropriate paths on your system before running any script (see Setup below).

---

## Pipeline

```
ChartQA dataset (HuggingFace: ahmed-masry/ChartQA)
    ↓
dataset.py — vocab construction (top-5000 answers + <UNK>)
    ↓
model.py — CLIP ViT-B/32 visual encoder + CLIP text encoder
         — l2-normalized embeddings concatenated → 2-layer MLP
         — optional LoRA adapters on q_proj, v_proj
    ↓
train.py — AdamW, lr=3e-4, cosine annealing, 20 epochs, batch=64
    ↓
evaluate.py — relaxed accuracy, breakdowns by question/chart/answer type
    ↓
gradcam.py — GradCAM + Attention Rollout on last ViT encoder block
zero_shot.py — Qwen2.5-VL-7B-Instruct greedy decoding
```

---

## Quick Start

**1. Clone and set up environment**
```bash
git clone https://github.com/MilanAdd/COMP646-Final-Project-ChartQA-Multimodal-Study
cd COMP646-Final-Project-ChartQA-Multimodal-Study

python -m venv venv
source venv/bin/activate
pip install torch torchvision transformers datasets peft pillow matplotlib
```

**2. Set environment variables**
```bash
export CHARTQA_DATA_DIR=<path where ChartQA data will be cached>
export CHARTQA_OUTPUT_DIR=<path for checkpoints, results, and figures>
export HF_HOME=<path for HuggingFace model cache>
```

**3. For NOTS users — set your NetID first**
```bash
export NETID=your_netid
```

The SLURM scripts use `__NETID__` as a placeholder which `launch.sh` replaces automatically. All scratch and work paths are derived from `$NETID` so no manual path editing is needed.

**4. Submit jobs**
```bash
./jobs/launch.sh train_frozen     # train frozen CLIP baseline
./jobs/launch.sh train_lora       # train LoRA CLIP
./jobs/launch.sh evaluate         # evaluate both on test set
./jobs/launch.sh zero_shot        # run Qwen2.5-VL zero-shot
./jobs/launch.sh gradcam          # generate GradCAM figures
```

If you are not using SLURM, the Python scripts can be run directly as long as the environment variables above are set.

---

## Repository Structure

```
.
├── config.py           # paths and hyperparameters
├── dataset.py          # ChartQA loading, vocab construction, chart type lookup
├── model.py            # ChartQAModel with LoRA support
├── train.py            # training loop
├── evaluate.py         # evaluation with full breakdowns
├── gradcam.py          # GradCAM + Attention Rollout
├── zero_shot.py        # Qwen2.5-VL inference
├── utils.py            # figure generation for report
└── jobs/
    ├── config.sh       # cluster config — set NETID before sourcing
    ├── launch.sh       # job submission helper
    ├── train.slurm
    ├── evaluate.slurm
    ├── gradcam.slurm
    └── zero_shot.slurm
```

---

## Models and Checkpoints

| Component | Details |
|-----------|---------|
| Visual encoder | CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) |
| Text encoder | CLIP (frozen in all variants) |
| MLP | 1024 → 512 (ReLU, dropout 0.3) → 5001 classes |
| LoRA rank | r=8, α=16, targets: q_proj, v_proj |
| Zero-shot model | Qwen2.5-VL-7B-Instruct |

Checkpoints are saved to `$CHARTQA_OUTPUT_DIR/checkpoints/` as `best_frozen.pt` and `best_lora.pt`, selected by validation accuracy.

---

## Generating Figures Locally

After running evaluations on your cluster, copy the eval JSON files
(`eval_frozen_test.json`, `eval_lora_test.json`, `eval_zeroshot_test.json`)
to a local `results/` directory, then run:

```bash
python utils.py
```

Figures are saved as PDFs to `config.FIGURES_DIR`.

---