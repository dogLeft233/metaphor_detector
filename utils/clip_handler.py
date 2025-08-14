from transformers import CLIPProcessor, CLIPModel
from typing import List
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

def sliding_chunks(token_ids: List[int], max_len: int = 77, stride: int = 38):
    """
    将输入的 token ID 序列切分为多个重叠片段，使用 pad_sequence 填充，并构造 attention mask。

    Args:
        token_ids (List[int]): 原始 token ID 列表（不含 special tokens）
        max_len (int): 每个块的最大 token 数（包含特殊 tokens）
        stride (int): 滑动步长（下一个片段起点距离上一个片段起点的距离）

    Returns:
        Dict[str, torch.Tensor]: 包含 'input_ids' 和 'attention_mask'，
            shape 分别为 (num_chunks, max_len)
    """

    chunks = []
    length = len(token_ids)
    for start in range(0, length, stride):
        end = min(start + max_len, length)
        sub_ids = token_ids[start:end]
        chunks.append(torch.tensor(sub_ids, dtype=torch.long))
        if end == length:
            break

    padded = pad_sequence(chunks, batch_first=True, padding_value=CLIPHandler.TOKENIZER.pad_token_id)

    attention_mask = (padded != CLIPHandler.TOKENIZER.pad_token_id).long()

    return {"input_ids": padded, "attention_mask": attention_mask}

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