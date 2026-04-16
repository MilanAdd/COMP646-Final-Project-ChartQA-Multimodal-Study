"""
This file houses the different model definitions for the ChartQA project.

Three variants are supported and they all share the same ChartQAModel class:
    1. Frozen CLIP - both visual and text encoders are frozen. Only the MLP fusion head is trained.
                     It tests how much chart understanding is already in CLIP's pretrained representations
    2. LoRA CLIP   - the visual encoder gets LoRA adapters (q_proj and v_proj). The text encoder stays frozen, but
                     the MLP head is trained. It tests whether visual domain adaptation helps.
    3. Zero-shot   - handled separately; requires no training
"""

import os

import torch 
import torch.nn as nn
from transformers import CLIPModel,CLIPTokenizerFast

import config

class LoRALinear(nn.Module):
    """
    Wraps an existing linear layer with a LoRA adapter

    What the forward pass is compute this formula:
        y = W_frozen @ x + (alpha/r) * B @ A @ x

        W_frozen contain the original weights of the model and are frozen. We only train
        a smaller subset of parameters, which are the lower-rank matrices A and B that adjusts 
        for the original weights so that the outputs match the desired application. 
        We do random initialization of a Gaussian for matrix A and zeros for B, so training starts with
        pretrained weights.

    Arguments:
        linear: original layer to wrap with adapter for finetuning
        r: rank of the lower-rank matrices A and B (# of cols in A, # of rows in B)
        alpha: scaling factor (usually set to r or 2 * r, latter in this case)
        dropout: dropout prob applied to input before A
    """
    def __init__(self,linear:nn.Linear,r:int,alpha:float,dropout:float):
        super().__init__()

        self.linear = linear # original frozen weights
        self.r = r
        self.scaling = alpha / r

        in_feats = linear.in_features
        out_feats = linear.out_features

        # Lower rank matrices A and B
        self.lora_A = nn.Linear(in_feats,r,bias=False)
        self.lora_B = nn.Linear(r,out_feats,bias=False)

        self.dropout = nn.Dropout(p=dropout)

        # Initializing A to be distributed as a Gaussian with 0 mean, standard deviation of 0.02
        nn.init.normal_(self.lora_A.weight,std=0.02)
        # Initialize B to be just zeros
        nn.init.zeros_(self.lora_B.weight)

        # Freeze the original weights
        for param in self.linear.parameters():
            param.requires_grad = False

    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base + lora


def lora_to_visual_enc(clip_model:CLIPModel,r:int = config.LORA_R, alpha:int = config.LORA_ALPHA, dropout: float = config.LORA_DROPOUT, target_modules: list = config.LORA_TARGET_MODULES) -> CLIPModel:
    """
    Traverse through CLIP visual encoder, replace each linear layer whose name ends with
    with one of the target_modules strings with LoRALinear wrapper

    Only LoRA adapater params (lora_A and lora_B) will have gradients calculated/updated after call to this function.
    Everything else stays frozen

    Arguments:
        clip_model: full CLIPModel (visual and text)
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: dropout on LoRA input
        target_modules: list of Linear layer name suffixes to adapt

    Returns:
        clip_model with LoRA applied in-place to visual encoder
    """
    visual_encoder = clip_model.vision_model

    for name,module in visual_encoder.named_modules():
        # Checking if module's short name matches target
        short_name = name.split(".")[-1]
        if short_name in target_modules and isinstance(module,nn.Linear):
            # Navigate to parent, replace child
            parts = name.split(".")
            parent = visual_encoder
            for part in parts[:-1]:
                parent = getattr(parent,part)
            lora_layer = LoRALinear(module,r=r,alpha=alpha,dropout=dropout)
            setattr(parent,parts[-1],lora_layer)

    n_trainable = sum(p.numel() for p in visual_encoder.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in visual_encoder.parameters())
    print(f"[LoRA] Applied to visual encoder - "
          f"Trainable: {n_trainable:,} / {n_total:,} params "
          f"({100 * n_trainable / n_total:.2f}%)")

    return clip_model

class FusionMLP(nn.Module):
    """
    A two layer multilayer perceptron that maps concatenated image and text featues to class logits

    Architecture:
        [image_feat and text_feat] -> Linear -> ReLU -> Dropout -> Linear -> logits

    Arguments:
        input_dim: dimension of concatenated feature vector
        hidden_dim: size of hidden layer
        num_classes: number of output classes (vocab_size + 1 for UNK)
        dropout: dropout prob before output layer
    """
    def __init__(self,input_dim:int,hidden_dim:int,num_classes:int,dropout:float):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_dim,num_classes))

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class ChartQAModel(nn.Module):
    """
    Full ChartQA model: CLIP encoders and MLP fusion head

    Supports two training modes controlled by 'use_lora':
        - use_lora=False: frozen CLIP (only MLP head is trained)
        - use_lora=True: LoRA adapters on visual encoder + MLP head trained

    Text encoder is always frozen no matter what

    Arguments:
        num_classes: number of answer classes (vocab_size + 1)
        use_lora: whether to apply LoRA to the visual encoder
        clip_model_name: HF model ID for CLIP backbone
    """
    def __init__(self,num_classes:int,use_lora:bool=False,clip_model_name:str = config.CLIP_NAME):
        super().__init__()
        self.use_lora = use_lora

        # Load CLIP
        print(f"[Model] Loading CLIP: {clip_model_name}")
        clip = CLIPModel.from_pretrained(clip_model_name)

        # Freeze everything first
        for param in clip.parameters():
            param.requires_grad = False

        # Optionally apply LoRA to visual encoder
        if use_lora:
            clip = lora_to_visual_enc(clip)

        # Attach encoder (could potentially be LoRA-adapted)
        self.visual_encoder = clip.vision_model
        self.visual_proj = clip.visual_projection # projects to embed_dim
        self.text_encoder = clip.text_model
        self.text_proj = clip.text_projection # projects to embed_dim

        # MLP fusion head (always trained)
        input_dim = config.CLIP_EMBED_DIM * 2  # concatenation of image, text features
        self.fusion = FusionMLP(input_dim=input_dim,hidden_dim = config.MLP_HIDDEN_DIM,num_classes=num_classes, dropout = config.MLP_DROPOUT)

        self._report_params()
        
    def encode_img(self,pixel_values:torch.Tensor) -> torch.Tensor:
        """
        Extract and normalize visual features

        Arguments:
            pixel_values: FloatTensor [B,3,H,W]

        Returns:
            FloatTensor [B,embed_dim]
        """
        outputs = self.visual_encoder(pixel_values=pixel_values)
        # Pooler output is CLS token representation [B,hidden_dim]
        pooled = outputs.pooler_outuput
        feat = self.visual_proj(pooled) # [B,embed_dim]
        return nn.functional.normalize(feat,dim=-1)
    
    def encode_text(self,input_ids:torch.Tensor,attention_mask:torch.Tensor) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=input_ids,attention_mask=attention_mask)
        pooled = outputs.pooler_output
        feat = self.text_proj(pooled)
        return nn.functional.normalize(feat,dim=-1)
    
    def forward(self,pixel_values:torch.Tensor,input_ids:torch.Tensor,attention_mask:torch.Tensor) -> torch.Tensor:
        visual_feats = self.encode_img(pixel_values)
        text_feats = self.encode_text(input_ids,attention_mask)
        combined = torch.cat([visual_feats,text_feats],dim=-1)
        logits = self.fusion(combined)
        return logits
    
    def _report_params(self) -> None:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        mode = "LoRA" if self.use_lora else "Frozen CLIP"
        print(f"[Model] Mode: {mode} | "
              f"Trainable: {trainable:,} / {total:,} params "
              f"({100 * trainable / total:.2f}%)")
    

def get_tokenizer(clip_model_name:str = config.CLIP_NAME):
    return CLIPTokenizerFast.from_pretrained(clip_model_name)

def tokenize_questions(questions:list,tokenizer,device:torch.device) -> dict:
    encoded = tokenizer(questions,padding=True,truncation=True,max_length=77,return_tensors="pt")
    return {k:v.to(device) for k,v in encoded.items()}

def save_checkpoint(model:ChartQAModel, optimizer,epoch:int, val_acc:float,path:str) -> None:
    torch.save({"epoch":epoch,"val_acc":val_acc,"use_lora":model.use_lora,"model":model.state_dict(),"optimizer":optimizer.state_dict()},path)
    print(f"[Checkpoint] Saved epoch {epoch} (val_acc={val_acc:.4f}) -> {path}")

def load_checkpoint(path:str, model:ChartQAModel,optimizer=None) -> dict:
    checkpoint = torch.load(path,map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"[Checkpoint] Loaded from {path} "
          f"(epoch = {checkpoint['epoch']},val_acc={checkpoint['val_acc']:.4f})")
    return checkpoint


# For good measure/practice
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Sanity] Device: {device}")

    num_classes = config.VOCAB_SIZE + 1

    print("\n ===== Frozen CLIP =====")
    frozen_model = ChartQAModel(num_classes=num_classes,use_lora=False).to(device)

    print("\n ===== LoRA CLIP =====")
    lora_model = ChartQAModel(num_classes=num_classes,use_lora=True).to(device)

    tokenizer = get_tokenizer()
    questions = ["What is the value of the tallest bar?", "Which year had the highest sales?"]
    tokens = tokenize_questions(questions,tokenizer,device)
    pixel_vals = torch.randn(2,3,config.IMAGE_SIZE,config.IMAGE_SIZE).to(device)

    with torch.no_grad():
        logits = frozen_model(pixel_vals,**tokens)
    
    print(f"\n[Sanity] Output logits shape: {logits.shape}")
    print("[Sanity] Forward pass OK")