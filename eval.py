import os
import json
import argparse
from collections import defaultdict

import torch

import config
from dataset import prepare_data,correct_relaxed
from model import (ChartQAModel,get_tokenizer,tokenize_questions,load_checkpoint)

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

            if gold_ans == "<unk>" or pred_idx==0:
                is_correct = False
                is_unk = True
            else:
                is_correct = correct_relaxed(pred_ans,gold_ans)
                is_unk = False

            results.append({"correct":is_correct,"is_unk":is_unk,"pred_answer":pred_ans,"gold_answer":gold_ans,"question":questions[idx],"question_type":question_types[idx],"chart_type":chart_types[idx]})

    
    return results



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

    
    return {"overall":overall,"by_question_type":by_ques_type,"by_chart_type":by_chart_type,"by_cross":by_cross}