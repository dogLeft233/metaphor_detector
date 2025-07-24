import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from tqdm import trange
from typing import Dict, List, Tuple
import torchvision.transforms as T
import torch

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
        
    def collate(self, batch):
        texts = [text for item in batch for text in item["captions"]]
        images = [item["image"] for item in batch]
        
        inputs = self.processor(text = texts, images = images, return_tensors = "pt", padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        return inputs