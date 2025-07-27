import os
import pandas as pd
from model.CLIPW2V import W2VDisambiguator
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from typing import Dict
# 设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_model.eval()

model = W2VDisambiguator().to(device)
model.load_state_dict(torch.load("train_log\exp_2025-07-26_12-34-03\model.pth", map_location=device))
model.eval()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 读取TSV
tsv_path = 'data/archive/avg_test_label_E.csv'
labels = pd.read_csv(tsv_path, encoding="utf-8")
labels = labels.rename(columns={"images_name": "file_name"})
labels["file_name"] = labels["file_name"].apply(lambda x: x.replace("E_", ""))

text = pd.read_csv("data/archive/E_text.csv", encoding="gbk")

df = pd.merge(labels, text, on="file_name")

# # 假设图片文件夹路径
img_dir = "./data/archive/Eimages/Eimages/Eimages"

print(f"图片地址{img_dir}")

results_dict = {k: [] for k in ["text_embeds", "image_embeds", "shifted_image_embeds", "shifted_text_embeds"]}

@torch.no_grad()
def encode(input_ids:torch.Tensor, attention_mask:torch.Tensor, pixel_values:torch.Tensor)->Dict[str,torch.Tensor]:
    clip_output = clip_model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
    text_embeds = clip_output.text_embeds
    image_embeds = clip_output.image_embeds
    
    shifted_output = model(text_embeds, image_embeds)
    return {
        "text_embeds":text_embeds,
        "image_embeds":image_embeds,
        "shifted_image_embeds":shifted_output["image_disambiguated_embeds"],
        "shifted_text_embeds":shifted_output["text_disambiguated_embeds"]
    }

with torch.no_grad():
    for idx, row in tqdm(df.iterrows(), total=len(df), ncols=80):
        if pd.notnull(row['text']):
            text = str(row['text'])
        else:
            text = ""
            print(f"{row['file_name']} text missing!")
            
        img_name = row["file_name"]
        img_path = os.path.join(img_dir, img_name)
        try:
           image = Image.open(img_path).convert('RGB')
        except:
            tqdm.write(f"{img_path} miss!")
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        
        inputs = processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,        # 开启截断
                max_length=77           # CLIP 文本最大长度
            )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output_dict = encode(**inputs)
        for k in results_dict.keys():
            v = output_dict[k]
            v_flat = v.view(-1).cpu().tolist()
            results_dict[k].append(v_flat)

# 将每个编码的展平向量作为一列
new_features = {}
for k, arrs in results_dict.items():
    new_features[k] = arrs  # 每行为一个np.ndarray
new_features_df = pd.DataFrame(new_features)

# 合并到原df
# 注意：如果需要保存为字符串，可用.apply(lambda x: ','.join(map(str, x)))
df = pd.concat([df, new_features_df], axis=1)

print(f"新数据集大小: {len(df)}")

path = "./data/archive/avg_test.csv"
df.to_csv(path, index=False, encoding="utf-8")
print(f"已处理并保存: {path}")