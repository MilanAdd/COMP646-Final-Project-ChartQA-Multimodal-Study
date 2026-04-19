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


# Parsing for command line/terminal arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train ChartQA Classifier")

    parser.add_argument("--mode",type=str,choices=["frozen","lora"],default="frozen",help="frozen: train only MLP head | lora: train LoRA adapters in addition to MLP")
    parser.add_argument("--epochs",type=int,default=config.NUM_EPOCHS)
    parser.add_argument("--lr",type=float,default=config.LEARNING_RATE)
    parser.add_argument("--batch-size",type=int,default=config.BATCH_SIZE)
    parser.add_argument("--workers",type=int,default=config.NUM_WORKERS)
    parser.add_argument("--resume",type=str,default=None,help="Path to checkpoint to resume training from")

    return parser.parse_args()


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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    print(f"[Train] Mode: {args.mode}")

    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    answer2idx,train_loader,val_loader,_ = prepare_data()
    num_classes = len(answer2idx)
    tokenizer = get_tokenizer()

    use_lora = (args.mode=="lora")
    model = ChartQAModel(num_classes=num_classes,use_lora=use_lora).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(trainable_params,lr=args.lr,weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optim,T_max=args.epochs,eta_min=1e-6)

    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    best_val_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        checkpnt = load_checkpoint(args.resume,model,optim)
        start_epoch = checkpnt["epoch"]+1
        best_val_acc = checkpnt["val_acc"]

    history = {"mode":args.mode,"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}

    checkpnt_path = os.path.join(config.CHECKPOINT_DIR,f"best_{args.mode}.pt")

    print(f"\n[Train] Starting training for {args.epochs} epochs\n")

    for epoch in range(start_epoch,args.epochs+1):
        start_time = time.time()

        train_loss,train_acc = train_epoch(model,train_loader,optim,criterion,tokenizer,device,epoch,args.epochs)

        val_loss,val_acc = eval(model,val_loader,criterion,tokenizer,device)

        scheduler.step()

        time_passed = time.time() - start_time

        print(f"\nEpoch {epoch}/{args.epochs} ({time_passed:.1f}s) - "
              f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model,optim,epoch,val_acc,checkpnt_path)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        history_path = os.path.join(config.RESULTS_DIR,f"history_{args.mode}.json")

        with open(history_path,"w") as f:
            json.dump(history,f,indent=2)

    
    print(f"\n[Train] Done training! The best validation accuracy is: {best_val_acc:.4f}")
    print(f"[Train] History saved to {history_path}")
    print(f"[Train] Best model checkpoint saved tp {checkpnt_path}")



if __name__ == "__main__":
    main()