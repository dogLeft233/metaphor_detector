import pandas as pd
from typing import List
import torch
from PIL import Image
import re
import string
import numpy as np
from gensim.models import KeyedVectors
from pathlib import Path
from tqdm import tqdm
from utils.clip_handler import CLIPHandler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#设置flickr 30k路径
csv_path = "./data/flickr30k/flickr_annotations_30k.csv"
image_dir = Path("./data/flickr30k/flickr30k-images")
save_path = "./data/flickr30k/processed_flickr30k_.csv"

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
    text_embeds = torch.stack([CLIPHandler.process_text(sentence, device=device) for sentence in sentences], dim=0)
    
    image_path = image_dir / file_name
    image_embeds = CLIPHandler.process_image(image_path, device=device)
    
    image_embeds = image_embeds.view(-1).cpu().tolist()
    text_embeds = text_embeds.view(-1).cpu().tolist()
    
    w2v_embeds = torch.stack([get_sentence_w2v(s) for s in sentences],dim=0).view(-1).cpu().tolist()
    
    return pd.Series({
            "image_embeds":image_embeds, 
            "text_embeds":text_embeds,
            "labels":w2v_embeds
        })

# 对原始数据每一行使用process函数处理

processed_data = []
for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing data"):
        sentences = eval(row["raw"])
        file_name = row["filename"]
        
        result = process(sentences, file_name)
        processed_data.append(result)

# 将处理结果转换为DataFrame并保存
result_df = pd.DataFrame(processed_data)
result_df.to_csv(save_path, encoding="utf-8")