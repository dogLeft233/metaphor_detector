import os
import pandas as pd
from model.clip_w2v_gpt import W2VDisambiguator
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_model.eval()

model = W2VDisambiguator()

#使用训练得到的偏移器
model.load_state_dict(torch.load("./train_log/exp_2025-07-26_18-10-17/model.pth", map_location=device))
model.eval()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

csv_path = "./data/archive/avg_train_label_E.csv"
df = pd.read_csv(csv_path, sep='\t', encoding="utf-8")

img_dir = "./data/MultiMET/Facebook_pic_solved"

print(f"图片地址{img_dir}")

results_dict = {k: [] for k in ["text_embeds", "image_embeds", "shifted_image_embeds", "shifted_text_embeds"]}

with torch.no_grad():
    for idx, row in tqdm(df.iterrows(), total=len(df), ncols=80):
        text = str(row[' Text  ']) if pd.notnull(row[' Text  ']) else ""
        img_name = str(int(row['ID'])) + '.jpg'
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            tqdm.write(f"miss!")
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        else:
            image = Image.open(img_path).convert('RGB')
        
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_dict = model.encode(**inputs)
        for k in results_dict.keys():
            v = output_dict[k]
            v_flat = v.view(-1).cpu().tolist()
            results_dict[k].append(v_flat)

new_features = {}
for k, arrs in results_dict.items():
    new_features[k] = arrs
new_features_df = pd.DataFrame(new_features)

df = pd.concat([df, new_features_df], axis=1)

path = "./data/MultiMET/embedded_Facebook_pic_solved.tsv"
df.to_csv(path, sep='\t', index=False, encoding="utf-8")
print(f"已处理并保存: {path}")