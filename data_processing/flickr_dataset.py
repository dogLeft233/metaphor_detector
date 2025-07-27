import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import trange
from typing import Dict, List, Tuple
import torchvision.transforms as T
import torch
import numpy as np
import string
import re
from tqdm.auto import tqdm

class FlickrDataset(Dataset):
    
    TRANSFORM = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    def __init__(self, csv_path, images_dir, use_transform:bool=False):
        super().__init__()
        self.data = pd.read_csv(csv_path)
        self.images_dir = images_dir
        # 替换caption中的标点
        punct_re = f'[{re.escape(string.punctuation)}]'
        self.data['caption'] = self.data['caption'].apply(lambda s: re.sub(punct_re, '', s))
        # 分组：每个image对应多条caption
        self.groups = self.data.groupby('image')['caption'].apply(list).reset_index()
        # 预处理并存储所有图像
        self.images = []
        for i in trange(len(self.groups), ncols=80, desc="图像处理中...", leave=False):
            row = self.groups.iloc[i]
            image_path = os.path.join(self.images_dir, row['image'])
            image = Image.open(image_path).convert('RGB')
            if use_transform:
                image = FlickrDataset.TRANSFORM(image)
            self.images.append(image)
            
        print(f"成功读取{len(self.groups)}条数据")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        image = self.images[idx]
        captions = self.groups.iloc[idx]['caption']  # list of 5 captions
        return {
            'image': image,
            'captions': captions
        }

class FlickrCollator:
    def __init__(self, processor, w2v,device = "cpu"):
        self.processor = processor
        self.device = device
        self.w2v = w2v
        
    def _word2vec(self, word:str)->torch.Tensor:
        if word in self.w2v:
            return torch.tensor(np.array(self.w2v[word], dtype=np.float32), dtype=torch.float, device=self.device)
        return torch.zeros(300, dtype=torch.float, device=self.device)
    
    def _get_sentence_embed(self, sentence:str)->torch.Tensor:
        word_list = sentence.split(' ')
        embeds = torch.stack([self._word2vec(word) for word in word_list], dim=0)
        return embeds.mean(dim=0, keepdim=False)
        
    def collate(self, batch):
        texts = [text for item in batch for text in item["captions"]]
        images = [item["image"] for item in batch]
        
        inputs = self.processor(text = texts, images = images, return_tensors = "pt", padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        
        sentence_embeds = torch.stack([self._get_sentence_embed(text) for text in texts], dim=0)
        
        inputs["sentence_embeds"] = sentence_embeds
        return inputs
    
class EmbeddingsDataset(Dataset):
    def __init__(self, csv_path)->None:
        super().__init__()
        self.data = pd.read_csv(csv_path)
        tqdm.pandas(desc="处理数据中...", ncols=80, leave=False)
        
        self.data["image_embeds"] = self.data["image_embeds"].progress_apply(eval)
        self.data["text_embeds"] = self.data["text_embeds"].progress_apply(eval)
        self.data["labels"] = self.data["labels"].progress_apply(eval)
        
        print(f"成功读取{len(self.data)}条数据")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx:int)->Dict[str,torch.Tensor]:
        return self.data.iloc[idx]
    
class EmbeddingsCollator:
    def __init__(self, device = "cpu"):
        self.device = device
        
    def collate(self, batch):
        image_embeds = torch.stack([torch.tensor(item["image_embeds"], dtype=torch.float, device=self.device) for item in batch], dim=0)
        text_embeds = torch.stack([torch.tensor(item["text_embeds"], dtype=torch.float, device=self.device) for item in batch], dim=0)
        labels = torch.stack([torch.tensor(item["labels"], dtype=torch.float, device=self.device) for item in batch], dim=0)
        
        return {
            "image_embeds":image_embeds,
            "text_embeds":text_embeds,
            "labels":labels
        }