from transformers import CLIPProcessor, CLIPModel
from typing import List
import torch
from torch import nn
from PIL import Image
import numpy as np
from data_processing.text_processing import sliding_chunks
from tqdm import tqdm

class CLIPHandler:
    CLIPPROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    TOKENIZER = CLIPPROCESSOR.tokenizer
    
    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(_DEVICE)
    CLIP.eval()
    
    @staticmethod
    def text2ids(text:str)->List[int]:
        return CLIPHandler.TOKENIZER(
            text,
            add_special_tokens=True,
            return_attention_mask=False,
            return_tensors=None
        )["input_ids"]
    
    @staticmethod
    @torch.no_grad()
    def process_text(text:str, max_len:int = 77, device = 'cpu')->torch.Tensor:
        token_ids = CLIPHandler.text2ids(text)
        if len(token_ids) > max_len:
            clip_input = sliding_chunks(token_ids, max_len = max_len, stride=max_len // 2)
        else:
            clip_input = {"input_ids":torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)}
            
        clip_input = {k:v.to(CLIPHandler._DEVICE) for k,v in clip_input.items()}
        text_embeds = CLIPHandler.CLIP.get_text_features(**clip_input)
        return text_embeds.mean(dim = 0, keepdim = False).to(device)
    
    @staticmethod
    @torch.no_grad()
    def process_image(image_path:str, device = 'cpu')->torch.Tensor:
        try:
            image = Image.open(image_path)
        except:
            tqdm.write(f"{image_path} miss!")
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        
        inputs = CLIPHandler.CLIPPROCESSOR(images=image, return_tensors="pt")
        inputs = {k: v.to(CLIPHandler._DEVICE) for k, v in inputs.items()}
        
        image_embeds = CLIPHandler.CLIP.get_image_features(**inputs)
        
        return image_embeds.squeeze(0).to(device)