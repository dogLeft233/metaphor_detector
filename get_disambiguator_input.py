import pandas as pd
from typing import List
import torch
from PIL import Image
import re
import string
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from gensim.models import KeyedVectors
from pathlib import Path

device = 'cuda'

csv_path = "./data/flickr30k/flickr_annotations_30k.csv"
image_dir = Path("./data/flickr30k/flickr30k-images")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
model.eval()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
w2v = KeyedVectors.load_word2vec_format("./word2vec/GoogleNews-vectors-negative300.bin", binary=True)

data = pd.read_csv(csv_path, encoding="utf-8")

punct_re = f'[{re.escape(string.punctuation)}]'

def word2vec(word:str)->torch.Tensor:
    if word in w2v:
        return torch.tensor(np.array(w2v[word], dtype=np.float32), dtype=torch.float, device=device)
    return torch.zeros(300, dtype=torch.float, device=device)

def get_sentence_w2v(sentence:str)->torch.Tensor:
    word_list = sentence.split(' ')
    embeds = torch.stack([word2vec(word) for word in word_list], dim=0)
    return embeds.mean(dim=0, keepdim=False)

@torch.no_grad()
def process(sentences:List[str], file_name:str):
    assert len(sentences) == 5
    
    sentences = [re.sub(punct_re, '', s) for s in sentences]
    
    image_path = image_dir / file_name
    image = Image.open(image_path)
    
    inputs = processor(
        text= sentences,
        images=image,
        return_tensors="pt",
        padding= True, 
        truncation=True,
        max_length=77
    )
    
    inputs = {k:v.to(device) for k,v in inputs.items()}
    
    outputs = model(**inputs)
    
    image_embeds = outputs.image_embeds.view(-1).cpu().tolist()
    text_embeds = outputs.text_embeds.view(-1).cpu().tolist()
    
    w2v_embeds = torch.stack([get_sentence_w2v(s) for s in sentences],dim=0).view(-1).cpu().tolist()
    
    return pd.Series(
        {"image_embeds":image_embeds, 
         "text_embeds":text_embeds,
         "labels":w2v_embeds
        }
        )

# 对原始数据每一行使用process函数处理
from tqdm import tqdm

processed_data = []
for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing data"):
        # 获取5个caption
        sentences = eval(row["raw"])
        file_name = row["filename"]  # 假设图片文件名列名为'image_name'
        
        # 使用process函数处理
        result = process(sentences, file_name)
        processed_data.append(result)
    # except Exception as e:
    #     print(f"Error processing row {idx}: {e}")
    #     continue

# 将处理结果转换为DataFrame并保存
result_df = pd.DataFrame(processed_data)