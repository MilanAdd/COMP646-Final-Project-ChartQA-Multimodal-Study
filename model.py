import torch 
import torch.nn as nn
from transformers import CLIPModel,CLIPTokenizerFast

import config

class LoRALinear(nn.Module):
    def __init__(self,linear:nn.Linear,r:int,alpha:float,dropout:float):
        super().__init__()

        self.linear = linear
        self.r = r
        self.scaling = alpha / r

        in_feats = linear.in_features
        out_feats = linear.out_features

        self.lora_A = nn.Linear(in_feats,r,bias=False)
        self.lora_B = nn.Linear(r,out_feats,bias=False)

        self.dropout = nn.Dropout(p=dropout)

        nn.init.normal_(self.lora_A.weight,std=0.02)
        nn.init.zeros_(self.lora_B.weight)

        for param in self.linear.parameters():
            param.requires_grad = False

    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base + lora


def lora_to_visual_enc(clip_model:CLIPModel,r:int = config.LORA_R, alpha:int = config.LORA_ALPHA, dropout: float = config.LORA_DROPOUT, target_modules: list = config.LORA_TARGET_MODULES) -> CLIPModel:
    visual_encoder = clip_model.vision_model

    for name,module in visual_encoder.named_modules():
        short_name = name.split(".")[-1]
        if short_name in target_modules and isinstance(module,nn.Linear):
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
    def __init__(self,input_dim:int,hidden_dim:int,num_classes:int,dropout:float):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden_dim,num_classes))

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

class ChartQAModel(nn.Module):
    def __init__(self,num_classes:int,use_lora:bool=False,clip_model_name:str = config.CLIP_NAME):
        super().__init__()
        self.use_lora = use_lora

        print(f"[Model] Loading CLIP: {clip_model_name}")
        clip = CLIPModel.from_pretrained(clip_model_name)

        for param in clip.parameters():
            param.requires_grad = False

        if use_lora:
            clip = lora_to_visual_enc(clip)

        self.visual_encoder = clip.vision_model
        self.visual_proj = clip.visual_projection
        self.text_encoder = clip.text_model
        self.text_proj = clip.text_projection

        input_dim = config.CLIP_EMBED_DIM * 2
        self.fusion = FusionMLP(input_dim=input_dim,hidden_dim = config.MLP_HIDDEN_DIM,num_classes=num_classes, dropout = config.MLP_DROPOUT)
        
    def encode_img(self,pixel_values:torch.Tensor) -> torch.Tensor:
        outputs = self.visual_encoder(pixel_values=pixel_values)
        pooled = outputs.pooler_outuput
        feat = self.visual_proj(pooled)
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