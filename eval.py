import os
import json
import argparse
from collections import defaultdict

import torch

import config
from dataset import prepare_data,correct_relaxed
from model import (ChartQAModel,get_tokenizer,tokenize_questions,load_checkpoint)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ChartQA Classifier")

    parser.add_argument("--checkpoint",type=str,required=True,help="Path to model checkpoint")
    parser.add_argument("--mode",type=str,choices=["frozen","lora"],required=True,help="Must match the mode used during training")
    parser.add_argument("--split",type=str,choices=["val","test"],default="test",help="Which split to evaluate on (default: test)")
    parser.add_argument("--batch_size",type=int,default=config.BATCH_SIZE)
    parser.add_argument("--workers",type=int,default=config.NUM_WORKERS)
    
    return parser.parse_args()


@torch.no_grad()
def run_eval(model,loader,tokenizer,device,idx2answer:dict):
    model.eval()
    results = []

    for batch in loader:
        pixel_vals = batch["image"].to(device)
        questions = batch["question"]
        gold_answers = batch["gold_answer"]
        chart_types = batch["chart_type"]
        question_types = batch["question_type"]

        tokens = tokenize_questions(questions,tokenizer,device)
        logits = model(pixel_vals,**tokens)
        preds = logits.argmax(dim=-1).cpu().tolist()

        for idx in range(len(questions)):
            pred_idx = preds[idx]
            pred_ans = idx2answer.get(pred_idx,"<UNK>")
            gold_ans = gold_answers[idx]

            if gold_ans == "<unk>":
                is_correct = False
                is_unk = True
            else:
                is_correct = correct_relaxed(pred_ans,gold_ans)
                is_unk = False

            results.append({"correct":is_correct,"is_unk":is_unk,"pred_answer":pred_ans,"gold_answer":gold_ans,"question":questions[idx],"question_type":question_types[idx],"chart_type":chart_types[idx],"answer_type":classify_answer_type(gold_ans)})

    
    return results

BINARY_ANSWERS = {"yes","no"}

def classify_answer_type(answer:str) -> str:
    a = answer.strip().lower()
    if a in BINARY_ANSWERS:
        return "binary"
    try:
        float(a.replace(",",""))
        return "numerical"
    except ValueError:
        return "textual"



def compute_acc(results:list,filter_fn=None) -> dict:
    subset = [r for r in results if not r["is_unk"]]
    if filter_fn:
        subset = [r for r in subset if filter_fn(r)]
    
    total = len(subset)
    correct = sum(r["correct"] for r in subset)
    acc = correct / total if total > 0 else 0.0

    return {"correct":correct,"total":total,"accuracy":acc}

def compute_breakdowns(results:list) -> dict:
    overall = compute_acc(results)

    ques_types = set(r["question_type"] for r in results)
    by_ques_type = {}
    for ques in sorted(ques_types):
        by_ques_type[ques] = compute_acc(results,filter_fn=lambda r, q=ques:r["question_type"]==q)

    ct_types = set(r["chart_type"] for r in results)
    by_chart_type = {}
    for ct in sorted(ct_types):
        by_chart_type[ct] = compute_acc(results,filter_fn=lambda r,c=ct:r["chart_type"]==c)
    
    by_cross = {}
    for ques in sorted(ques_types):
        for ct in sorted(ct_types):
            key = f"{ques}__{ct}"
            by_cross[key] = compute_acc(results,filter_fn=lambda r,q=ques,c=ct:(r["question_type"]==q and r["chart_type"]==c))

    atypes = set(r["answer_type"] for r in results)
    by_answer_type = {}
    for at in sorted(atypes):
        by_answer_type[at] = compute_acc(results,filter_fn=lambda r, a=at:r["answer_type"] == a)

    
    return {"overall":overall,"by_question_type":by_ques_type,"by_chart_type":by_chart_type,"by_cross":by_cross,"by_answer_type":by_answer_type}

def print_breakdowns(breakdowns:dict,mode:str) -> None:
    """
    Print formatted summary of evaluation results
    """

    print(f"\n{'='*60}")
    print(f" Evaluation Results - {mode.upper()}")
    print(f"{'='*60}")

    ovrll = breakdowns["overall"]
    print(f"\n Overall Accuracy: {ovrll['accuracy']:.4f}"
          f"({ovrll['correct']}/{ovrll['total']})")
    
    print("\n --- By Question Type ---")
    for ques,stats in breakdowns["by_question_type"].items():
        print(f"    {ques:15s}: {stats['accuracy']:.4f}"
              f"({stats['correct']}/{stats['total']})")

    print("\n --- By Chart Type ---")
    for ct,stats in breakdowns["by_chart_type"].items():
        print(f"    {ct:15s}: {stats['accuracy']:.4f} "
              f"({stats['correct']}/{stats['total']})")
    
    print("\n --- Cross Breakdown (Question Type x Chart Type) --- ")
    for key,stats in breakdowns["by_cross"].items():
        if stats["total"] > 0:
            print(f"    {key:30s}: {stats['accuracy']:.4f} "
                  f"({stats['correct']}/{stats['total']})")
    
    print(f"\n{'='*60}\n")

def get_qual_ex(results:list,n_correct:int=5,n_incorrect:int=5) -> dict:
    non_unk = [r for r in results if not r["is_unk"]]
    correct = [r for r in non_unk if r["correct"]]
    incorrect = [r for r in non_unk if not r["correct"]]

    def diverse_sample(pool,n):
        seen_chart_types = set()
        selected = []

        for r in pool:
            if r["chart_type"] not in seen_chart_types:
                selected.append(r)
                seen_chart_types.add(r["chart_type"])
            if len(selected) >= n:
                break
        
        for r in pool:
            if r not in selected:
                selected.append(r)
            if len(selected) >= n:
                break
        
        return selected[:n]

    return {"correct": diverse_sample(correct,n_correct),"incorrect":diverse_sample(incorrect,n_incorrect)}

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")
    print(f"[Eval] Mode: {args.mode}")
    print(f"[Eval] Split: {args.split}")
    print(f"[Eval] Checkpoint: {args.checkpoint}")

    answer2idx,train_loader,val_loader,test_loader = prepare_data()
    loader = test_loader if args.split == "test" else val_loader
    tokenizer = get_tokenizer()

    idx2answer = {v:k for k,v in answer2idx.items()}

    num_classes = len(answer2idx)
    use_lora = (args.mode == "lora")
    model = ChartQAModel(num_classes=num_classes,use_lora=use_lora).to(device)
    load_checkpoint(args.checkpoint,model)

    print(f"\n[Eval] Running inference on {args.split} set...")
    results = run_eval(model,loader,tokenizer,device,idx2answer)

    breakdowns = compute_breakdowns(results)
    print_breakdowns(breakdowns,args.mode)

    examples = get_qual_ex(results)
    print(f"[Eval] Qualitative examples: "
          f"{len(examples['correct'])} correct, "
          f"{len(examples['incorrect'])} incorrect")
    
    output = {"mode":args.mode,"split":args.split,"checkpoint":args.checkpoint,"breakdowns":breakdowns,"examples":examples}

    out_path = os.path.join(config.RESULTS_DIR,f"eval_{args.mode}_{args.split}.json")
    with open(out_path,"w") as f:
        json.dump(output,f,indent=2)
    
    print(f"[Eval] Results saved to {out_path}")


if __name__ == "__main__":
    main()