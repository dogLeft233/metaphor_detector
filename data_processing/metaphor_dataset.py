from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict
from transformers import GPT2Tokenizer
import re
import torch
    
class MIPEmbeddingDataset(Dataset):
    def __init__(self, file_path):
        # file_path = Path(file_path)
        # assert file_path.suffix == ".tsv"
        
        data = pd.read_csv(file_path, encoding="utf-8")
        data["text_embeds"] = data["text_embeds"].apply(lambda x: eval(x))
        data["image_embeds"] = data["image_embeds"].apply(lambda x: eval(x))
        data["solo_image_embeds"] = data["solo_image_embeds"].apply(lambda x: eval(x))
        data["solo_text_embeds"] = data["solo_text_embeds"].apply(lambda x: eval(x))
        
        data["sentiment category"] = data["sentiment category"].apply(lambda x: self._extract_integer(x) - 1)
        data["intention detection"] = data["intention detection"].apply(lambda x: self._extract_integer(x) - 1)
        data["offensiveness detection"] = data["offensiveness detection"].apply(lambda x: self._extract_integer(x))
        
        self.data = data
        
        print(f"成功加载{len(self)}条数据")
        
    def _extract_integer(self, s: str) -> int:
        """
        提取形如 '123(描述)' 字符串中的整数部分
        """
        match = re.match(r"^(\d+)", s.strip())
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"无法从字符串中提取整数: {s}")
        
    def __len__(self)->int:
        return len(self.data)
    
    def __getitem__(self, index:int)->Dict:
        return dict(self.data.iloc[index].items())
    
class MIPEmbeddingCollator:
    def __init__(self, device):
        self.device = device
    def collate(self, batch):
        text_embeds = torch.stack([torch.tensor(item["text_embeds"], dtype=torch.float, device=self.device) for item in batch], dim=0)
        image_embeds = torch.stack([torch.tensor(item["image_embeds"], dtype=torch.float, device=self.device) for item in batch], dim=0)
        solo_image_embeds = torch.stack([torch.tensor(item["solo_image_embeds"], dtype=torch.float, device=self.device) for item in batch], dim=0)
        solo_text_embeds = torch.stack([torch.tensor(item["solo_text_embeds"], dtype=torch.float, device=self.device) for item in batch], dim=0)
        labels = torch.tensor([item["metaphor occurrence"] for item in batch], dtype=torch.long ,device=self.device)
        intention_detection = torch.tensor([item["intention detection"] for item in batch], dtype=torch.long, device=self.device)
        sentiment_category = torch.tensor([item["sentiment category"] for item in batch], dtype=torch.long, device=self.device)
        offensiveness_detection = torch.tensor([item["offensiveness detection"] for item in batch], dtype=torch.long, device=self.device)
        
        return{
            "text_embeds":text_embeds,
            "image_embeds":image_embeds,
            "solo_image_embeds":solo_image_embeds,
            "solo_text_embeds":solo_text_embeds,
            "labels":labels,
            "intention_detection":intention_detection,
            "sentiment_category":sentiment_category,
            "offensiveness_detection":offensiveness_detection
            
        }