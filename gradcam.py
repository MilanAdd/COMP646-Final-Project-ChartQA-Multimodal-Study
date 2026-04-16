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
        self._activations = output.detach()

    def _save_gradient(self,module,grad_input,grad_output):
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


def overlay_heatmap(image:Image.Image,heatmap:np.darray,alpha:float=0.5,colormap:str="jet")-> np.ndarray:
    img_resized = image.resize((config.IMAGE_SIZE,config.IMAGE_SIZE),Image.LANCZOS)
    img_array = np.array(img_resized).astype(np.float32)/255.0

    cmap = cm.get_cmap(colormap)
    heat_rgb = cmap(heatmap)[:,:,:3]

    blended = (1-alpha)*img_array + alpha* heatmap
    blended = np.clip(blended,0,1)
    return (blended*255).astype(np.uint8)

def make_gradcam_figure(examples:list,model:ChartQAModel,gradcam:GradCAM,tokenizer,device:torch.device,idx2answer:dict,alpha:float,title:str,save_path:str) -> None:
    n = len(examples)
    if n ==  0:
        print(f"[GradCAM] No examples to plot for: {title}")
    
    fig,axes = plt.subplots(n,2,figsize=(10,4*n))
    if n==1:
        axes = [axes]

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
        
        overlay = overlay_heatmap(pil_img,heatmap,alpha=alpha)
        ax_img = axes[row_idx][0]
        ax_heat = axes[row_idx][1]

        ax_img.imshow(pil_img.resize(config.IMAGE_SIZE,config.IMAGE_SIZE),Image.LANCZOS)
        ax_img.axis("off")

        ax_heat.imshow(overlay)
        ax_heat.axis("off")

        status = "Correct" if example["correct"] else "Incorrect"
        caption = (f"{status} | Chart: {chart_type} | Question Type: {question_type}\n"
                   f"Question: {question[:80]}{'...' if len(question) > 80 else ''}\n"
                   f"Gold: {gold_answer}    Pred: {pred_answer}")
        ax_img.set_title("Original",fontsize=9,pad=4)
        ax_heat.set_title("GradCAM overlay",fontsize=9,pad=4)
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
                    return sample["image"]
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

    gradcam = GradCAM(model)

    common_kwargs = dict(model=model,gradcam=gradcam,tokenizer=tokenizer,device=device,idx2answer=idx2answer,alpha=args.alpha)

    make_gradcam_figure(examples=correct_ex,title=f"GradCAM (Correct examples) ({args.mode})",save_path=os.path.join(config.FIGURES_DIR,f"gradcam_{args.mode}_correct.pdf"))

    make_gradcam_figure(examples=incorrect_ex,title=f"GradCAM (Incorrect predictions) ({args.mode})",save_path=os.path.join(config.FIGURES_DIR,f"gradcam_{args.mode}_incorrect.pdf"))
    gradcam.remove_hooks()
    print("[GradCAM] Done")

if __name__ == "__main__":
    main()