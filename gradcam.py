import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import config
from dataset import (prepare_data,get_chart_type_lookup,CLIP_IMG_TRANSFORM)
from model import (ChartQAModel,get_tokenizer,tokenize_questions,load_checkpoint)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate GradCAM figures")

    parser.add_argument("--checkpoint",type=str,required=True)
    parser.add_argument("--mode",type=str,choices=["frozen","lora"],required=True)
    parser.add_argument("--eval-results",type=str,required=True,help="Path to eval_<mode>_test.json produced by evaluate.py")
    parser.add_argument("--n-correct",type=int,default=5,help="Number of correct examples to visualize")
    parser.add_argument("--n-incorrect",type=int,default=5,help="Number of incorrect examples to visualize")
    parser.add_argument("--alpha",type=float,default=0.5,help="Opacity of heatmap overlay (0=just image,1=just heatmap)")

    return parser.parse_args()

class GradCAM:
    def __init__(self,model:ChartQAModel,target_layer=None):
        self.model = model
        if target_layer is None:
            target_layer = model.visual_encoder.encoder.layers[-1]

        self.target_layer = target_layer
        self._activations = None
        self._gradients = None

        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self,module,input,output):
        if isinstance(output,tuple):
            self._activations = output[0].detach()
        else:
            self._activations = output.detach()

    def _save_gradient(self,module,grad_input,grad_output):
        if isinstance(grad_output[0],tuple):
            self._gradients = grad_output[0][0].detach()
        else:
            self._gradients = grad_output[0].detach()

    def generate(self,pixel_values:torch.Tensor,input_ids:torch.Tensor,attention_mask:torch.Tensor,target_class:int=None) -> np.ndarray:
        self.model.eval()

        pixel_values= pixel_values.requires_grad_(True)

        logits = self.model(pixel_values,input_ids,attention_mask)
        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        self.model.zero_grad()
        score= logits[0,target_class]
        score.backward()

        activations = self._activations[0]
        gradients = self._gradients[0]

        activations = activations[1:]
        gradients = gradients[1:]

        weights = gradients.mean(dim=-1)

        cam = (weights.unsqueeze(-1) * activations).sum(dim=-1)

        cam = F.relu(cam)

        num_patches = cam.shape[0]
        grid_size = int(num_patches ** 0.5)
        cam_2d = cam.reshape(grid_size,grid_size).cpu().numpy()

        cam_tensor = torch.from_numpy(cam_2d).unsqueeze(0).unsqueeze(0)
        cam_upsample = F.interpolate(cam_tensor,size=(config.IMAGE_SIZE,config.IMAGE_SIZE),mode='bilinear',align_corners=False).squeeze().numpy()

        if cam_upsample.max() > cam_upsample.min():
            cam_upsample = (cam_upsample-cam_upsample.min()) / (cam_upsample.max() - cam_upsample.min())
        else:
            cam_upsample = np.zeros_like(cam_upsample)
        

        return cam_upsample


    def remove_hooks(self):
        self._forward_hook.remove()
        self._backward_hook.remove()

class AttentionRollout:
    def __init__(self,model:ChartQAModel,discard_ratio:float=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        self._attentions = []
        self._hooks = []

        for layer in model.visual_encoder.encoder.layers:
            hook = layer.self_attn.register_forward_hook(self._save_attention)
            self._hooks.append(hook)
        
    def _save_attention(self,module,input,output):
        if isinstance(output,tuple) and len(output) > 1 and output[1] is not None:
            self._attentions.append(output[1].detach())
    
    def generate(self,pixel_values:torch.Tensor) -> np.ndarray:
        self._attentions = []
        self.model.eval()

        with torch.no_grad():
            self.model.visual_encoder(pixel_values=pixel_values,output_attentions=True)
        if not self._attentions:
            grid = int(config.IMAGE_SIZE/32)
            return np.ones((config.IMAGE_SIZE,config.IMAGE_SIZE))/(grid ** 2)
    
        rollout = None
        for attention in self._attentions:
            attention_avg = attention.mean(dim=1)[0]
            flat = attention_avg.view(-1)
            threshold = flat.quantile(self.discard_ratio)
            attention_avg = torch.where(attention_avg < threshold, torch.zeros_like(attention_avg),attention_avg)

            seq_len = attention_avg.shape[0]
            identity = torch.eye(seq_len,device=attention_avg.device)
            attention_avg = 0.5 * attention_avg * identity

            attention_avg = attention_avg / attention_avg.sum(dim=-1,keepdim=True).clamp(min=1e-6)
             
            if rollout is None:
                rollout = attention_avg
            else:
                rollout = torch.matmul(attention_avg,rollout)
        
        cls_attention = rollout[0,1:].cpu().numpy()
        num_patches = cls_attention.shape[0]
        grid_size = int(num_patches**0.5)
        attention_2d = cls_attention.reshape(grid_size,grid_size)
        attention_tensor = torch.from_numpy(attention_2d).unsqueeze(0).unsqueeze(0)
        attention_upsamp = F.interpolate(attention_tensor,size=(config.IMAGE_SIZE,config.IMAGE_SIZE),mode='bilinear',align_corners=False).squeeze().numpy()

        if attention_upsamp.max() > attention_upsamp.min():
            attention_upsamp = (attention_upsamp-attention_upsamp.min()) / (attention_upsamp.max()-attention_upsamp.min())
        else:
            attention_upsamp = np.zeros_like(attention_upsamp)
        return attention_upsamp

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()


def overlay_heatmap(image:Image.Image,heatmap:np.ndarray,alpha:float=0.5,colormap:str="jet")-> np.ndarray:
    img_resized = image.resize((config.IMAGE_SIZE,config.IMAGE_SIZE),Image.LANCZOS)
    img_array = np.array(img_resized).astype(np.float32)/255.0

    cmap = plt.colormaps[colormap]
    heat_rgb = cmap(heatmap)[:,:,:3]

    blended = (1-alpha)*img_array + alpha* heat_rgb
    blended = np.clip(blended,0,1)
    return (blended*255).astype(np.uint8)

def make_combined_figure(examples:list,model:ChartQAModel,gradcam:GradCAM,rollout:AttentionRollout,tokenizer,device:torch.device,idx2answer:dict,alpha:float,title:str,save_path:str) -> None:
    n = len(examples)
    if n ==  0:
        print(f"[GradCAM] No examples to plot for: {title}")
        return
    
    fig,axes = plt.subplots(n,3,figsize=(14,4*n))
    if n==1:
        axes = [list(axes)]

    fig.suptitle(title,fontsize=13,fontweight="bold",y=1.01)

    for row_idx,example in enumerate(examples):
        question = example["question"]
        gold_answer = example["gold_answer"]
        pred_answer = example["pred_answer"]
        chart_type = example["chart_type"]
        question_type = example["question_type"]

        imgname = example.get("imgname","")

        pil_img = _load_img_for_example(example)
        if pil_img is None:
            print(f"[GradCAM] Could not load image for: {question[:50]}")
            continue
            
        pixel_values = CLIP_IMG_TRANSFORM(pil_img.convert("RGB")).unsqueeze(0).to(device)

        tokens = tokenize_questions([question],tokenizer,device)
        with torch.enable_grad():
            heatmap = gradcam.generate(pixel_values,tokens["input_ids"],tokens["attention_mask"])

        rollout_map = rollout.generate(pixel_values)

        
        gc_overlay = overlay_heatmap(pil_img,heatmap,alpha=alpha)
        rollout_overlay = overlay_heatmap(pil_img,rollout_map,alpha=alpha,colormap="viridis")

        ax_img = axes[row_idx][0]
        ax_grad = axes[row_idx][1]
        ax_roll = axes[row_idx][2]

        ax_img.imshow(pil_img.resize((config.IMAGE_SIZE,config.IMAGE_SIZE),Image.LANCZOS))
        ax_img.axis("off")
        ax_img.set_title("Original",fontsize=9,pad=4)

        ax_grad.imshow(gc_overlay)
        ax_grad.axis("off")
        ax_grad.set_title("GradCAM",fontsize=9,pad=4)

        ax_roll.imshow(rollout_overlay)
        ax_roll.axis("off")
        ax_roll.set_title("Attention Rollout",fontsize=9,pad=4)

        status = "Correct" if example["correct"] else "Incorrect"
        caption = (f"{status} | Chart: {chart_type} | Question Type: {question_type}\n"
                   f"Question: {question[:80]}{'...' if len(question) > 80 else ''}\n"
                   f"Gold: {gold_answer}    Pred: {pred_answer}")
        fig.text(0.5,1-(row_idx+1)/n+0.01/n,caption,ha="center",va="top",fontsize=7.5,transform=fig.transFigure,wrap=True)
    
    plt.tight_layout()
    plt.savefig(save_path,format="pdf",dpi=300,bbox_inches="tight")
    plt.close()
    print(f"[GradCAM] Saved figure to {save_path}")



def _load_img_for_example(example:dict):
    try:
        from datasets import load_dataset
        hf_data = load_dataset(config.HF_DATASET_NAME,cache_dir=config.DATA_DIR)
        question = example["question"]
        for split in ("test","val","train"):
            for sample in hf_data[split]:
                if sample["query"] == question:
                    img = sample["image"]
                    if isinstance(img, bytes):
                        import io
                        from PIL import Image
                        return Image.open(io.BytesIO(img)).convert("RGB")
                    return img
    except Exception as e:
        print(f"[GradCAM] Image load error: {e}")
    return None

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GradCAM] Device: {device}")

    with open(args.eval_results) as f:
        eval_data = json.load(f)

    correct_ex = eval_data["examples"]["correct"][:args.n_correct]
    incorrect_ex = eval_data["examples"]["incorrect"][:args.n_incorrect]

    answer2idx,_,_,_ = prepare_data()
    idx2answer = {v:k for k,v in answer2idx.items()}
    tokenizer = get_tokenizer()

    num_classes = len(answer2idx)
    use_lora = (args.mode == "lora")
    model = ChartQAModel(num_classes=num_classes,use_lora=use_lora).to(device)
    load_checkpoint(args.checkpoint,model)

    model.visual_encoder.config._attn_implementation = "eager"

    gradcam = GradCAM(model)
    rollout = AttentionRollout(model)

    common_kwargs = dict(model=model,gradcam=gradcam,rollout=rollout,tokenizer=tokenizer,device=device,idx2answer=idx2answer,alpha=args.alpha)

    make_combined_figure(**common_kwargs,examples=correct_ex,title=f"GradCAM vs Attention Rollout - (Correct examples) ({args.mode})",save_path=os.path.join(config.FIGURES_DIR,f"gradcam_{args.mode}_correct.pdf"))

    make_combined_figure(**common_kwargs,examples=incorrect_ex,title=f"GradCAM vs Attention Rollout - (Incorrect predictions) ({args.mode})",save_path=os.path.join(config.FIGURES_DIR,f"gradcam_{args.mode}_incorrect.pdf"))
    gradcam.remove_hooks()
    rollout.remove_hooks()
    print("[GradCAM] Done")

if __name__ == "__main__":
    main()