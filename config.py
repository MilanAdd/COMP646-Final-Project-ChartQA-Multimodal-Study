"""
This is the main configuration for the ChartQA Multimodal study project.
It's where all the hyperparameters, paths, and model choices are.

Made to decouple the process of changing settings from other individual scripts
"""

import os


# Root directory of the project, which is one level above where this file is
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Directory where downloaded ChartQA data, cached HuggingFace models are stored.
# On NOTS cluster specifically, this is the $WORK storage space with a 2TB quota/group,
# also NFS-mounted so available on login nodes, don't need to deal with purging unlike $SHARED_SCRATCH
DATA_DIR = os.path.join(os.environ.get("WORK",PROJECT_ROOT),"chartqa_data")

# Where trained model checkpoints are saved
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT,"checkpoints")

# Where evaluation results (in JSON and/or CSV) are written
RESULTS_DIR = os.path.join(PROJECT_ROOT,"results")

# Where GradCAM figures and other graphs/plots are saved
FIGURES_DIR = os.path.join(PROJECT_ROOT,"figures")

# Make directories if they're not already present
for _dir in [DATA_DIR,CHECKPOINT_DIR,RESULTS_DIR,FIGURES_DIR]:
    os.makedirs(_dir,exist_ok=True)

# HF dataset ID for ChartQA
HF_DATASET_NAME = "/ahmed-masry/ChartQA"

# Amount of most frequent training answers to keep in vocabulary
# Answers outside vocab are labeled as UNK (unknown words outside vocab), not included in accuracy
# Starting with 100 for now as it should cover around two thirds of training answers
VOCAB_SIZE = 100

# Derived from ChartQA paper, the relaxed accuracy tolerance for numerical answers specifically
# Allows minor inaccuracy that could result from automatic data extraction
# Answer is seen as correct if it's within 5% of gold answer (ground truth)
RELAXED_TOLERANCE = 0.05

# The CLIP backbone that controls visual and text encoder
# Using the CLIP model with the ViT-B/32 architecture (32 being patch size) for now as it's
# faster, uses lower memory, and is good for iteration
CLIP_NAME = "openai/clip-vit-base-patch32"

# Dimensionality of CLIP's output embeddings for given backbone
# For the base-patch32 model, this dimensionality would be 512, but for some other variants, this
# could be 768 or 1024
CLIP_EMBED_DIM = 512

# Dimensionality of single hidden layer in CLIP model
MLP_HIDDEN_DIM = 512

# Dropout that's applied before the output layer
MLP_DROPOUT = 0.3

# Rank - size of trainable low-rank decomposition matrices A, B that are added to original model
LORA_R = 8

# Alpha - Scaling factor that controls infleunce of adapter weights 
# Using the heuristic 2 * rank in this case, but could also be same as rank
LORA_ALPHA = 16

# Dropout applied to matrices A, B
LORA_DROPOUT = 0.1

# Target modules - the specific modules (linear projection layers within attention blocks) we want to apply the adapter to
# In this case, we are only adapting the query and value projection matrices based on the original LoRA paper for reduced memory usage
LORA_TARGET_MODULES = ["q_proj","v_proj"]

# Training hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# Required input image size for CLIP
IMAGE_SIZE = 224
SEED = 42

# HF Model ID for multimodal VLM used as zero shot baseline
# Qwen2.5-VL (7 billion parameter model) is an instruction-tuned variant that takes in both image and text prompts
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# Prompt template used when querying QwenVL model
QWEN_PROMPT_TEMPLATE = ("Look at the chart carefully and answer the following question "
                        "with a short answer only (a number or a few words, no explanation).\n"
                        "Question: {question}\nAnswer:") # actual question string will replace {question} during runtime

# Max number of new tokens QwenVL is allowed to generate for each answer
QWEN_MAX_NEW_TOKENS = 20
