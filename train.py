import os
import json
import argparse
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from dataset import prepare_data
from model import (ChartQAModel, get_tokenizer, tokenize_questions, save_checkpoint, load_checkpoint)


def train_epoch(model,loader,optimizer,criterion,tokenizer,device,epoch,num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    non_unk_total = 0
    time_start = time.time()

    for batch_idx,batch in enumerate(loader):
        pixel_vals = batch["image"].to(device)
        answer_idx = batch["answer_idx"].to(device)
        questions = batch["question"]

        tokens = tokenize_questions(questions,tokenizer,device)

        logits = model(pixel_vals,**tokens)
        loss = criterion(logits,answer_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        non_unk_mask = (answer_idx!= 0)
        if non_unk_mask.sum() > 0:
            preds = logits.argmax(dim=-1)
            correct += (preds[non_unk_mask]==answer_idx[non_unk_mask]).sum().item()
            non_unk_total += non_unk_mask.sum().item()

        if (batch_idx + 1) % 50 == 0:
            time_past = time.time() - time_start
            print(f" Epoch [{epoch}/{num_epochs}] "
                  f"Step [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {loss.item():.4f}" 
                  f"({time_past:.1f}s past)")
            
    
    average_loss = total_loss/len(loader)
    acc = correct/non_unk_total if non_unk_total > 0 else 0.0
    return average_loss,acc


@torch.no_grad()
def eval(model,loader,criterion,tokenizer,device):
    model.eval()
    total_loss = 0.0
    correct = 0
    non_unk_total = 0

    for batch in loader:
        pixel_vals = batch["image"].to(device)
        answer_idx = batch["answer_idx"].to(device)
        questions = batch["question"]

        tokens = tokenize_questions(questions,tokenizer,device)
        logits = model(pixel_vals,**tokens)
        loss = criterion(logits,answer_idx)

        total_loss += loss.item()

        non_unk_mask = (answer_idx!=0)
        if non_unk_mask.sum() > 0:
            preds = logits.argmax(dim=-1)
            correct += (preds[non_unk_mask] == answer_idx[non_unk_mask]).sum().item()
            non_unk_total += non_unk_mask.sum().item()
    
    average_loss = total_loss / len(loader)
    acc = correct / non_unk_total if non_unk_total > 0 else 0.0
    return average_loss,acc