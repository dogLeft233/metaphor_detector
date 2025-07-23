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

from transformers import BertModel, BertTokenizerFast
from data.VUA18.VUA18_Dataset import VUA18_Collator_Embed, VUA18_Dataset
from tqdm import trange
from gensim.models import KeyedVectors
import torch

w2v = KeyedVectors.load_word2vec_format("./word2vec/GoogleNews-vectors-negative300.bin", binary=True)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

collator = VUA18_Collator_Embed(tokenizer, w2v)

data = VUA18_Dataset(
    file_path="./data/VUA18/dev.tsv",
    tokenizer=tokenizer
)

dataloader = DataLoader(data, batch_size=2, collate_fn=collator.collate)

for batch in dataloader:
    print(batch["input_ids"].shape)
    print(batch["embeds"].shape)
    break

# data = VUA18_Dataset_Embedded(
#     file_path="./data/VUA18/dev.tsv",
#     tokenizer=tokenizer,
#     device='cuda',
#     model=model
# )