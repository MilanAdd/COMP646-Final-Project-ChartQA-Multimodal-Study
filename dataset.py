"""
This is the file that handles ChartQA dataset loading, answer vocabulary
construction, and PyTorch Dataset & DataLoader setup.

Here is the overall structrue of ChartQA:
    - image: PIL image of the chart
    - query: natural language question, which is a string
    - label: groundtruth answer, which is a string that's either a number or short text
    - type: how questions were generated, either "human" or "augmented" (machine-generated)
    - imgname: filename of the chart image (e.g. "10095.png")

Chart type labels come from the official annotation JSON files in the full ChartQA dataset (config.ANNOTATIONS_DIR).
Each annotation file contains a "type" field with values: "v_bar", "h_bar", "line", "pie". We merge both bar types into
"bar" for simplicity. Run build_chart_type_lookup() once to produce a cached JSON file, which is loaded automatically by
prepare_data().

The task can be seen as classification over the top-K most frequent training answer (determined by config.VOCAB_SIZE).
Samples whose groundtruth/gold answers lie outside the vocabulary are retained in the dataset, but their answer is mapped to
the UNK token and therefore won't be included in accuracy computation during evalution.
"""

import os
import re
import json
from collections import Counter

import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset

import config

"""
Special tokens
"""
UNK_TOKEN = "<UNK>" # answer not in top-K vocab
PAD_TOKEN = "<PAD>" # will be used in tasks further down the line, like answer generation


def build_answer_vocab(hf_train_split,vocab_size:int) -> dict:
    """
    Building an vocabulary from training split, where answer strings 
    are mapped to indices

    Arguments:
        hf_train_split: HF dataset train split
        vocab_size: number of most frequent answers to keep

    Returns:
        answer2idx: a dictionary where answer strings are maped to integer class indices
                    0th index reserved for UNK_TOKEN
    """
    counts = Counter()
    for sample in hf_train_split:
        answer = normalize_answer(sample["label"])
        counts[answer]+=1
    
    most_freq = [answer for answer,_ in counts.most_common(vocab_size)]

    # index 0 assigned to UNK, indices afterwards are assigned actual answers
    answer2idx = {UNK_TOKEN:0}
    for idx,ans in enumerate(most_freq,start=1):
        answer2idx[ans] = idx

    print(f"[Vocab] Built vocabulary of {len(answer2idx)} entries "
          f"({vocab_size} answers + UNK)")

    # Showing what proportion of training answers are correct
    total = sum(counts.values())
    covered = sum(cnt for ans,cnt in counts.items() if ans in answer2idx)
    print(f"[Vocab] Coverage on train set: {covered/total:.1%} of answers")
    
    return answer2idx

def save_vocab(answer2idx:dict,path:str) -> None:
    """
    Save vocabulary to JSON file
    """
    with open(path,"w") as f:
        json.dump(answer2idx,f,indent=2)
    print(f"[Vocab] Saved to {path}")
    
def load_vocab(path:str) -> dict:
    """
    Load previously saved vocabulary from JSON file
    """
    with open(path) as f:
        answer2idx = json.load(f)
    print(f"[Vocab] Loaded {len(answer2idx)} entries from {path}")
    return answer2idx

def normalize_answer(answer:str) -> str:
    """
    Minimal normalization applied to vocab reconstruction, eval-time answer lookup

    Rules to consider for normalization:
    - strip leading and trailing whitespace
    - lowercase
    - remove trailing punctuation, such as periods and commas
    - collapse any internal whitespace to just one space

    Keeps numeric formatting the same
    """
    answer = answer.strip().lower()
    answer = re.sub(r"[?!;,.]+$","",answer) # trailing punctuation
    answer = re.sub(r"\s+"," ",answer) # remove unnecessary whitespace
    return answer

def is_numeric(s:str) -> bool:
    """
    Helper function for correct_relaxed
    
    Returns True if string represents number (int/float)
    """
    try:
        float(s.replace(",","")) # handling any comma formatted numbers
        return True
    except ValueError:
        return False


def correct_relaxed(pred:str,gold:str,tol:float=config.RELAXED_TOLERANCE) -> bool:
    """
    ChartQA relaxed accuracy, which is a measure where numerical answers are correct if they are
    within a tol * abs(gold) of the gold value. Text answers need exact match however

    Arguments:
        pred: predicted answer string (already normalized)
        gold: gold answer string (already normalized)
        tol: fractional tolerance (this is by default 5% from both paper and config)

    Returns:
        True if prediction is seen as correct
    """
    if pred == gold:
        return True
    
    if is_numeric(pred) and is_numeric(gold):
        pred_val = float(pred.replace(",",""))
        gold_val = float(gold.replace(",",""))
        if gold_val == 0:
            return pred_val == 0
        return abs(pred_val - gold_val) / abs(gold_val) <= tol
    
    return False

CHART_TYPE_LOOKUP_PATH = os.path.join(config.DATA_DIR,"chart_type_lookup.json")

# Mapping raw annotation type strings to cleaner labels
_TYPE_MAP = {"v_bar":"bar","h_bar":"bar","line":"line","pie":"pie","scatter":"scatter"}

def build_chart_type_lookup(annotations_dir:str,save_path:str=CHART_TYPE_LOOKUP_PATH) -> dict:
    """
    Traverse through annotations directory of ChartQA dataset, build lookup dictionary that maps imgname to chart_type

    Each annotation JSON file is named after its chart image (e.g. "10095.json" -> "10095.png"), contains high level "type" key
    with vals "v_bar", "h_bar", "line" or "pie"

    Arguments:
        annotations_dir: path to folder that has annotation JSONs.
                         Should cover all splits, point it at merged
                         folder or call this function once per split
                         and merge.
        save_path:       where to cache resulting JSON lookup
    """
    lookup = {}
    missing_type = 0

    for split in ("train","val","test"):
        split_dir = os.path.join(annotations_dir,split)
        if not os.path.isdir(split_dir):
            # trying flat structure (meaning all JSONs are just in one folder)
            split_dir = annotations_dir

        json_files = [f for f in os.listdir(split_dir) if f.endswith(".json")]

        for fname in json_files:
            imgname = fname.replace(".json",".png")
            fpath = os.path.join(split_dir,fname)
            try:
                with open(fpath) as f:
                    ann = json.load(f)
                if isinstance(ann, list):
                    ann = ann[0] if ann else {}
                raw_type = ann.get("type", "unknown")
                chart_type = _TYPE_MAP.get(raw_type, "unknown")
            except (json.JSONDecodeError,OSError):
                chart_type = "unknown"
                missing_type += 1

            lookup[imgname] = chart_type

        print(f"[ChartType] {split}: {len(json_files)} annotation files read")

    print(f"[ChartType] Total entries: {len(lookup)} | "
          f"Unknown or missing type: {missing_type}")
    
    # Report distribution
    dist = Counter(lookup.values())
    for ct, cnt in sorted(dist.items()):
        print(f" {ct:10s}: {cnt:,}")

    with open(save_path,"w") as f:
        json.dump(lookup,f)
    print(f"[ChartType] Saved lookup to {save_path}")

    return lookup

def load_chart_type_lookup(path:str = CHART_TYPE_LOOKUP_PATH) -> dict:
    """Load a previously built chart type lookup from JSON file"""
    with open(path) as f:
        lookup = json.load(f)
    print(f"[ChartType] Loaded {len(lookup):,} entries from {path}")
    return lookup

def get_chart_type_lookup(annotations_dir:str = None, force_rebuild: bool = False) -> dict:
    """
    Load chart type lookup for cache or build from scratch

    Arguments:
        annotations_dir: path to annotation JSONs (needed if building)
        force_rebuild: if True, rebuild even if cache exists

    Results:
        lookup dict mapping imgname to chart_type, or empty dict if annotations arent'
        available (chart_type will fall back to "unknown")
    """
    if os.path.exists(CHART_TYPE_LOOKUP_PATH) and not force_rebuild:
        return load_chart_type_lookup()
    
    if annotations_dir is None:
        annotations_dir = config.ANNOTATIONS_DIR

    if not os.path.isdir(annotations_dir):
        print(f"[ChartType] Annotations dir not found: {annotations_dir}\n"
              f"            Chart type will be 'unknown' for all samples.\n"
              f"            Download the full ChartQA dataset and set "
              f"config.ANNOTATIONS_DIR to enable chart type breakdown")
        return {}
    
    return build_chart_type_lookup(annotations_dir)

# Image transforms applied to each chart image before passing it to CLIP model
CLIP_IMG_TRANSFORM = transforms.Compose([transforms.Resize((config.IMAGE_SIZE,config.IMAGE_SIZE)),transforms.ToTensor(),transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711],),])


class ChartQADataset(Dataset):
    """
    PyTorch Dataset wrapping HuggingFace ChartQA split

    Returns:
        image: chart image of type FloatTensor with dimension (3,H,W)
        question: raw question text
        answer: class index (0 = UNK)
        gold_answer: normalized gold answer string
        chart_type: from annotation JSON lookup (bar, line, pie, scatter, unknown)
        question_type: "human" or "augmented"
    """
    def __init__(self,hf_split,answer2idx:dict,chart_type_lookup:dict=None,transform=None):
        """
        Arguments:
            hf_split: HF dataset split (train/val/test)
            answer2idx: vocab dict built by build_answer_vocab()
            chart_type_lookup: dict from annotations JSONs mapping imgname to chart_type
            transform: torchvision transform, which defaults to CLIP_IMG_TRANSFORM
        """
        self.data = hf_split
        self.answer2idx = answer2idx
        self.chart_type_lookup = chart_type_lookup or {}
        self.transform = transform or CLIP_IMG_TRANSFORM

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self,idx:int) -> dict:
        sample = self.data[idx]

        # image
        image = sample["image"]
        if isinstance(image,bytes):
            import io
            image = Image.open(io.BytesIO(image)).convert("RGB")
        
        elif image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)

        # answer
        gold_answer = normalize_answer(sample["label"])
        answer_idx = self.answer2idx.get(gold_answer,0)

        # chart type from annotation JSON lookup
        imgname = sample.get("imgname","") or ""
        chart_type = self.chart_type_lookup.get(imgname,"unknown")

        return {"image":image,"question":sample["query"],"answer_idx":answer_idx,"gold_answer":gold_answer,"chart_type":chart_type,"question_type":sample.get("type","unknown")}


def collate_fn(batch:list) -> dict:
    """
    Collate that stacks tensors and keeps string fields as lists.
    Passed directly to DataLoader
    """
    return {"image":torch.stack([s["image"] for s in batch]),
            "question":[s["question"] for s in batch],
            "answer_idx":torch.tensor([s["answer_idx"] for s in batch]),
            "gold_answer":[s["gold_answer"] for s in batch],
            "chart_type":[s["chart_type"] for s in batch],
            "question_type":[s["question_type"] for s in batch]}


def get_dataloaders(answer2idx:dict,batch_size:int=config.BATCH_SIZE,num_workers:int=config.NUM_WORKERS):
    """
    Download/load from cache ChartQA and return train/val/test DataLoaders.

    Arguments:
        answer2idx : answer vocabulary
        batch_size : samples per batch
        num_workers : worker processes

    Returns:
        train_loader,val_loader,test_loader
    """
    print(f"[Data] Loading ChartQA from HuggingFace ({config.HF_DATASET_NAME})...")

    # cache_dir keeps downloaded data in DATA_DIR rather default HuggingFace cache
    hf_data = load_dataset(config.HF_DATASET_NAME,cache_dir=config.DATA_DIR)

    chart_type_lookup = get_chart_type_lookup()
    train_ds = ChartQADataset(hf_data["train"],answer2idx,chart_type_lookup)
    val_ds = ChartQADataset(hf_data["val"],answer2idx,chart_type_lookup)
    test_ds = ChartQADataset(hf_data["test"],answer2idx,chart_type_lookup)

    print(f"[Data] Splits - train: {len(train_ds):,} "
          f"val: {len(val_ds):,} test: {len(test_ds):,}")
    
    common_kwargs = dict(batch_size = batch_size, num_workers = num_workers, collate_fn = collate_fn, pin_memory = True)

    train_loader = DataLoader(train_ds, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader


VOCAB_PATH = os.path.join(config.DATA_DIR, "answer_vocab.json")

def prepare_data(force_rebuild_vocab: bool = False):
    """
    Entry point used by train and evalaute files

    It loads ChartQA from HuggingFace, builds/loads from cache answer vocab, and returns (answer2idx,train_loader,val_loader,test_loader)

    Arguments:
        force_rebuild_vocab: if True, always rebuild even if JSON is there
    """
    print(f"[Data] Loading ChartQA from HuggingFace ({config.HF_DATASET_NAME})...")
    hf_data = load_dataset(config.HF_DATASET_NAME,cache_dir=config.DATA_DIR)

    if os.path.exists(VOCAB_PATH) and not force_rebuild_vocab:
        answer2idx = load_vocab(VOCAB_PATH)
    else:
        answer2idx = build_answer_vocab(hf_data["train"],config.VOCAB_SIZE)
        save_vocab(answer2idx,VOCAB_PATH)

    train_loader, val_loader, test_loader = get_dataloaders(answer2idx)
    return answer2idx,train_loader,val_loader,test_loader


# For good measure/practice
if __name__ == "__main__":
    answer2idx,train_loader,val_loader,test_loader = prepare_data()

    batch = next(iter(train_loader))
    print("\n -----Sample batch-----")
    print(f"  image shape  :   {batch['image'].shape}")
    print(f"  answer_idx  :  {batch['answer_idx'][:8].tolist()}")
    print(f"  question[0] :  {batch['question'][0]}")
    print(f"  gold_answer[0]  :  {batch['gold_answer'][0]}")
    print(f"  chart_type[0]  : {batch['chart_type'][0]}")
    print(f"  question_type  :  {batch['question_type'][:4]}")

    # Reporting rate of UNKs on training set
    total = unk = 0
    for b in train_loader:
        total += len(b["answer_idx"])
        unk += (b["answer_idx"] == 0).sum().item()
    print(f"\n[Data] UNK rate on train: {unk}/{total} = {unk/total:.1%}")