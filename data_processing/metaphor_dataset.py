from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict
from transformers import GPT2Tokenizer
import torch
    
class EmbeddedDataset(Dataset):
    def __init__(self, file_path):
        # file_path = Path(file_path)
        # assert file_path.suffix == ".tsv"
        
        data = pd.read_csv(file_path, encoding="utf-8")
        data["text_embeds"] = data["text_embeds"].apply(lambda x: eval(x))
        data["image_embeds"] = data["image_embeds"].apply(lambda x: eval(x))
        data["shifted_image_embeds"] = data["shifted_image_embeds"].apply(lambda x: eval(x))
        data["shifted_text_embeds"] = data["shifted_text_embeds"].apply(lambda x: eval(x))
        # data["Metaphor?"] = data["Metaphor?"].apply(lambda x: 1 if int(x) == 1 else 0)
        
        self.data = data
        
        print(f"成功加载{len(self)}条数据")
        
    def __len__(self)->int:
        return len(self.data)
    
    def __getitem__(self, index:int)->Dict:
        return dict(self.data.iloc[index].items())
    
class EmbeddedCollator:
    def __init__(self, device):
        self.device = device
    def collate(self, batch):
        text_embeds = torch.stack([torch.tensor(item["text_embeds"], dtype=torch.float, device=self.device) for item in batch], dim=0)
        image_embeds = torch.stack([torch.tensor(item["image_embeds"], dtype=torch.float, device=self.device) for item in batch], dim=0)
        shifted_image_embeds = torch.stack([torch.tensor(item["shifted_image_embeds"], dtype=torch.float, device=self.device) for item in batch], dim=0)
        shifted_text_embeds = torch.stack([torch.tensor(item["shifted_text_embeds"], dtype=torch.float, device=self.device) for item in batch], dim=0)
        labels = torch.tensor([item["metaphor occurrence"] for item in batch], dtype=torch.long ,device=self.device)
        
        return{
            "text_embeds":text_embeds,
            "image_embeds":image_embeds,
            "shifted_image_embeds":shifted_image_embeds,
            "shifted_text_embeds":shifted_text_embeds,
            "labels":labels
        }