import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(os.environ.get("WORK",PROJECT_ROOT),"chartqa_data")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT,"checkpoints")

RESULTS_DIR = os.path.join(PROJECT_ROOT,"results")

FIGURES_DIR = os.path.join(PROJECT_ROOT,"figures")

for _dir in [DATA_DIR,CHECKPOINT_DIR,RESULTS_DIR,FIGURES_DIR]:
    os.makedirs(_dir,exist_ok=True)

HF_DATASET_NAME = "/ahmed-masry/ChartQA"

VOCAB_SIZE = 100

RELAXED_TOLERANCE = 0.05

CLIP_NAME = "openai/clip-vit-base-patch32"

CLIP_EMBED_DIM = 512

MLP_HIDDEN_DIM = 512

MLP_DROPOUT = 0.3

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj","v_proj"]

BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

IMAGE_SIZE = 224
SEED = 42

QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

QWEN_PROMPT_TEMPLATE = ("Look at the chart carefully and answer the following question "
                        "with a short answer only (a number or a few words, no explanation).\n"
                        "Question: {question}\nAnswer:")
QWEN_MAX_NEW_TOKENS = 20
