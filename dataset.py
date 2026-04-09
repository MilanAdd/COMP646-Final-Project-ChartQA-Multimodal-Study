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

UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"

def build_answer_vocab(hf_train_split,vocab_size:int) -> dict:
    counts = Counter()
    for sample in hf_train_split:
        answer = normalize_answer(sample["label"])
        counts[answer]+=1
    
    most_freq = [answer for answer,_ in counts.most_common(vocab_size)]
    answer2idx = {UNK_TOKEN:0}
    for idx,ans in enumerate(most_freq,start=1):
        answer2idx[ans] = idx

    """
    total = sum(counts.values())
    covered = sum(cnt for ans,cnt in counts.items() if ans in answer2idx)
    """
    
    return answer2idx

def save_vocab(answer2idx:dict,path:str) -> None:
    with open(path,"w") as f:
        json.dump(answer2idx,f,indent=2)
    
def load_vocab(path:str) -> dict:
    with open(path) as f:
        answer2idx = json.load(f)
    return answer2idx

def normalize_answer(answer:str) -> str:
    answer = answer.strip().lower()
    answer = re.sub(r"[?!;,.]+$","",answer)
    answer = re.sub(r"\s+","",answer)
    return answer

def is_numeric(s:str) -> bool:
    try:
        float(s.replace(",",""))
        return True
    except ValueError:
        return False


def correct_relaxed(pred:str,gold:str,tol:float=config.RELAXED_TOLERANCE) -> bool:
    if pred == gold:
        return True
    
    if is_numeric(pred) and is_numeric(gold):
        pred_val = float(pred.replace(",",""))
        gold_val = float(gold.replace(",",""))
        if gold_val == 0:
            return pred_val == 0
        return abs(pred_val - gold_val) / abs(gold_val) <= tol
    
    return False




CLIP_IMG_TRANSFORM = transforms.Compose([transforms.Resize((config.IMAGE_SIZE,config.IMAGE_SIZE)),transforms.ToTensor(),transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711],),])


class ChartQADataset(Dataset):
    def __init__(self,hf_split,answer2idx:dict,transform=None):
        self.data = hf_split
        self.answer2idx = answer2idx
        self.transform = transform or CLIP_IMG_TRANSFORM

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self,idx:int) -> dict:
        sample = self.data[idx]

        image = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)

        gold_answer = normalize_answer(sample["label"])
        answer_idx = self.answer2idx.get(gold_answer,0)

        chart_type = _infer_chart_type(sample)

        return {"image":image,"question":sample["query"],"answer_idx":answer_idx,"gold_answer":gold_answer,"chart_type":chart_type,"question_type":sample.get("type","unknown")}
    


def _infer_chart_type(sample:dict) -> str:
    imgname = sample.get("imgname","") or ""
    imgname = imgname.lower()

    for chart_type in ("bar","line","pie","scatter"):
        if chart_type in imgname:
            return chart_type
        
    return "unknown"


def collate_fn(batch:list) -> dict:
    return {"image":torch.stack([s["image"] for s in batch]),
            "question":[s["question"] for s in batch],
            "answer_idx":torch.tensor([s["answer_idx"] for s in batch]),
            "gold_answer":[s["gold_answer"] for s in batch],
            "chart_type":[s["chart_type"] for s in batch],
            "question_type":[s["question_type"] for s in batch]}


def get_dataloaders(answer2idx:dict,batch_size:int=config.BATCH_SIZE,num_workers:int=config.NUM_WORKERS):
    print(f"[Data] Loading ChartQA from HuggingFace ({config.HF_DATASET_NAME})...")

    hf_data = load_dataset(config.HF_DATASET_NAME,cache_dir=config.DATA_DIR)
    train_ds = ChartQADataset(hf_data["train"],answer2idx)
    val_ds = ChartQADataset(hf_data["val"],answer2idx)
    test_ds = ChartQADataset(hf_data["test"],answer2idx)

    print(f"[Data] Splits - train: {len(train_ds):,} "
          f"val: {len(val_ds):,} test: {len(test_ds):,}")
    
    common_kwargs = dict(batch_size = batch_size, num_workers = num_workers, collate_fn = collate_fn, pin_memory = True)

    train_loader = DataLoader(train_ds, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader


VOCAB_PATH = os.path.join(config.DATA_DIR, "answer_vocab.json")

def prepare_data(force_rebuild_vocab: bool = False):
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

    total = unk = 0
    for b in train_loader:
        total += len(b["answer_idx"])
        unk += (b["answer_idx"] == 0).sum().item()
    print(f"\n[Data] UNK rate on train: {unk}/{total} = {unk/total:.1%}")