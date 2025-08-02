import os
import pandas as pd
from model.clip_w2v_gpt import W2VDisambiguator
import torch
from tqdm import tqdm
from typing import Dict
from utils.clip_handler import CLIPHandler
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = W2VDisambiguator().to(device)
#设置训练得到的消歧器
model.load_state_dict(torch.load("checkpoint/exp_2025-08-01_14-51-15/model.pth", map_location=device))
model.eval()

# 图片文件夹路径
img_dir = "./data/archive/Eimages/Eimages/Eimages"
print(f"图片地址{img_dir}")

# 读取csv
csv_path = 'data/archive/avg_val_label_E.csv'

labels = pd.read_csv(csv_path, encoding="utf-8")
labels = labels.rename(columns={"images_name": "file_name"})
labels["file_name"] = labels["file_name"].apply(lambda x: x.replace("E_", ""))

text = pd.read_csv("data/archive/E_text.csv", encoding="gbk")

df = pd.merge(labels, text, on="file_name")

results_dict = {k: [] for k in ["text_embeds", "image_embeds", "shifted_image_embeds", "shifted_text_embeds"]}

for idx, row in tqdm(df.iterrows(), total=len(df), ncols=80):
    if pd.notnull(row['text']):
        text = str(row['text'])
    else:
        text = ""
        print(f"{row['file_name']} text missing!")
        
    img_name = row["file_name"]
    img_path = os.path.join(img_dir, img_name)
    
    text_embed = CLIPHandler.process_text(text, device=device)
    image_embed =  CLIPHandler.process_image(img_path, device=device)
    
    shifted_output = model(text_embed, image_embed)
    
    results_dict["text_embeds"].append(text_embed.view(-1).cpu().tolist())
    results_dict["image_embeds"].append(image_embed.view(-1).cpu().tolist())
    results_dict["shifted_image_embeds"].append(shifted_output["image_disambiguated_embeds"].view(-1).cpu().tolist())
    results_dict["shifted_text_embeds"].append(shifted_output["text_disambiguated_embeds"].view(-1).cpu().tolist())

new_features = {}
for k, arrs in results_dict.items():
    new_features[k] = arrs 
new_features_df = pd.DataFrame(new_features)

df = pd.concat([df, new_features_df], axis=1)

print(f"新数据集大小: {len(df)}")

path = "./data/archive/avg_val.csv"
df.to_csv(path, index=False, encoding="utf-8")
print(f"已处理并保存: {path}")