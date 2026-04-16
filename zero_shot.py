import os
import json
import argparse
from collections import defaultdict

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import config
from dataset import (prepare_data,correct_relaxed,normalize_answer,get_chart_type_lookup)

def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot ChartQA evaluation with Qwen2.5-VL")
    parser.add_argument("--split",type=str,choices=["val","test"],default="test")
    parser.add_argument("--batch-size",type=int,default=4,help="Images per forward pass. Keep low to fit in GPU memory")
    parser.add_argument("--max-new-tokens",type=int,default=config.QWEN_MAX_NEW_TOKENS,help="Max tokens the model can generate per answer")
    parser.add_argument("--limit",type=int,default=None,help="Evaluate only first N samples")
    return parser.parse_args()

def build_msgs(image:Image.Image,question:str) -> list:
    prompt = config.QWEN_PROMPT_TEMPLATE.format(question=question)

    return [
        {
            "role":"user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

def extract_answer(raw_output:str) -> str:
    lines = [l.strip() for l in raw_output.strip().split("\n") if l.strip()]
    ans = lines[0] if lines else ""

    prefixes = ["answer:","the answer is","answer is"]
    ans_lower = ans.lower()
    for prefix in prefixes:
        if ans_lower.startswith(prefix):
            ans = ans[len(prefix):].strip()
            break
    
    return normalize_answer(ans)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n[ZeroShot] Loading model...")
    processor = AutoProcessor.from_pretrained(config.QWEN_MODEL_NAME,cache_dir=config.DATA_DIR)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.QWEN_MODEL_NAME,torch_dtype=torch.bfloat16,device_map="auto",cache_dir=config.DATA_DIR)
    model.eval()
    print("[ZeroShot] Model loaded")

    from datasets import load_dataset
    hf_data = load_dataset(config.HF_DATASET_NAME,cache_dir=config.DATA_DIR)
    split = hf_data[args.split]

    chart_type_lookup = get_chart_type_lookup()

    if args.limit:
        split = split.select(range(min(args.limit,len(split))))
        print(f"[ZeroShot] Limited to first {args.limit} samples")

    print(f"[ZeroShot] Evaluating {len(split):,} samples...\n")

    results = []

    for idx,sample in enumerate(split):
        img = sample["image"]
        question = sample["query"]
        gold_ans = normalize_answer(sample["label"])
        question_type = sample.get("type","unknown")
        imgname = sample.get("imgname","")
        chart_type = chart_type_lookup.get(imgname,"unknown")

        msgs = build_msgs(img,question)

        text = processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)

        inputs = processor(text=[text],images=[img],return_tensors="pt",padding=True).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)

        generated = output_ids[:,inputs["input_ids"].shape[1]:]
        raw_text = processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=True)[0]

        pred_ans = extract_answer(raw_text)
        is_correct = correct_relaxed(pred_ans,gold_ans)

        results.append({"correct":is_correct,"is_unk":False,"pred_answer":pred_ans,"gold_answer":gold_ans,"question":question,"question_type":question_type,"chart_type":chart_type,"raw_output":raw_text})

        if (idx+1) % 100 == 0:
            intermediate_progress = [r for r in results if not r["is_unk"]]
            running = sum(r["correct"] for r in intermediate_progress) / len(intermediate_progress)
            print(f"    [{idx+1}/{len(split)}] Running accuracy: {running:.4f}")
        
    from eval import compute_breakdowns,print_breakdowns,get_qual_ex

    breakdowns = compute_breakdowns(results)
    print_breakdowns(breakdowns,mode="zero-shot (Qwen2.5-VL)")

    examples = get_qual_ex(results)

    output = {"mode":"zeroshot","model":config.QWEN_MODEL_NAME,"split":args.split,"breakdowns":breakdowns,"examples":examples}
    out_path = os.path.join(config.RESULTS_DIR,f"eval_zeroshot_{args.split}.json")

    with open(out_path,"w") as f:
        json.dump(output,f,indent=2)

    print(f"\n[ZeroShot] Results saved to {out_path}")

if __name__== "__main__":
    main()
    

    