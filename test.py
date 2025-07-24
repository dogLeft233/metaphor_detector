# from data.VUA18.VUA18_Dataset import VUA18_Dataset, VUA18_Collator
from torch.utils.data.dataloader import DataLoader
# from transformers import BertTokenizer
# from model.TransformerClassifier import TransformerClassifier

# # 初始化分词器
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # dataset = VUA18_Dataset("./data/VUA18/train.tsv", tokenizer)

# dataset = VUA18_Dataset("./data/VUA18/train.tsv", tokenizer)

# # 统计正负例数量和比例
# from collections import Counter
# labels = [item["label"] for item in dataset]
# counter = Counter(labels)
# num_pos = counter[1]
# num_neg = counter[0]
# total = num_pos + num_neg
# pos_ratio = num_pos / total if total > 0 else 0
# neg_ratio = num_neg / total if total > 0 else 0

# print(f"正例数量: {num_pos}")
# print(f"负例数量: {num_neg}")
# print(f"正例比例: {pos_ratio:.4f}")
# print(f"负例比例: {neg_ratio:.4f}")


# data = VUA18_Dataset_Embedded(
#     file_path="./data/VUA18/dev.tsv",
#     tokenizer=tokenizer,
#     device='cuda',
#     model=model
# )

from data_processing.flickr.flickr import FlickrDataset, FlickrCollator
import torchvision.transforms as T
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data.dataloader import DataLoader

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_root ="./data/flickr8k/Images"
ann_file = "./data/flickr8k/captions.csv"

dataset = FlickrDataset(
    csv_path=ann_file,
    images_dir=image_root
)

collator = FlickrCollator(processor)

dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator.collate)

for batch in dataloader:
    output = model(**batch)
    image_embeds = output.image_embeds
    text_embeds = output.text_embeds
    print(image_embeds.shape)
    print(text_embeds.shape)
    break

# output = model(**inputs)

# image_embeds = output.image_embeds
# text_embeds = output.text_embeds

# print(image_embeds.shape)
# print(text_embeds.shape)

# print(type(img), img.shape)

# print(type(captions), captions)

# print(type(img), img.shape)
# print(len(captions), captions[0])