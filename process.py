import os
import pandas as pd
from model.CLIPW2V import CLIPEncoder
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor
from tqdm import tqdm

# 设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CLIPEncoder().to(device)
model.load_state_dict(torch.load("./train_log/exp_2025-07-24_18-10-17/model.pth", map_location=device))
model.eval()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 读取TSV
tsv_path = 'data/archive/test.csv'
df = pd.read_csv(tsv_path, encoding="utf-8")

# # 假设图片文件夹路径
img_dir = "./data/archive/Eimages/Eimages/Eimages"

print(f"图片地址{img_dir}")

results_dict = {k: [] for k in ["text_embeds", "image_embeds", "shifted_image_embeds", "shifted_text_embeds"]}

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
        with torch.no_grad():
            output_dict = model.encode(**inputs)
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

path = "./data/archive/new_test.csv"
df.to_csv(path, index=False, encoding="utf-8")
print(f"已处理并保存: {path}")