from torch.utils.data.dataset import Dataset
import pandas as pd
from transformers import PreTrainedTokenizer, BertTokenizerFast
from pathlib import Path
from gensim.models import KeyedVectors
import torch
from tqdm.auto import tqdm
from typing import Dict, List
import numpy as np
import re
import string
from torch.nn.utils.rnn import pad_sequence

class VUA18_Dataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        super().__init__()
        tqdm.pandas()
        
        file_path = Path(file_path)
        assert file_path.suffix.lower() == ".tsv"
        
        self.data = pd.read_csv(file_path, encoding="utf-8",sep='	')
        
        # 预处理：将标点符号转换为空格
        tqdm.pandas(desc="清理标点符号")
        punct_re = f'[{re.escape(string.punctuation)}]'
        self.data["sentence"] = self.data["sentence"].progress_apply(lambda s: re.sub(punct_re, ' ', s))

        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 丢弃分词后长度大于max_length的样本
        keep_indices = []
        for idx, sentence in enumerate(self.data["sentence"]):
            tokenized = self.tokenizer.encode(sentence, add_special_tokens=True)
            if len(tokenized) <= self.max_length:
                keep_indices.append(idx)
        self.data = self.data.iloc[keep_indices].reset_index(drop=True)
        
        print(f"成功读取{len(self)}条数据")
        
    def __len__(self)->int:
        return len(self.data)
    
    def __getitem__(self, index):
        return dict(self.data.iloc[index].items())
    
class VUA18_Collator:
    def __init__(self, tokenizer:PreTrainedTokenizer, device = "cpu"):
        self.device = device
        self.tokenizer = tokenizer
        
    def collate(self, batch:List[Dict])->Dict:
        sentences = [item["sentence"] for item in batch]
        sentences = self.tokenizer(
            text=sentences,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False
        )
        labels = torch.tensor([item["label"] for item in batch],device=self.device)
        
        return {
            "input_ids":sentences["input_ids"].to(self.device),
            "attention_mask":sentences["attention_mask"].to(self.device).bool(),
            "labels":labels.float()
        }

class VUA18_Collator_Embed:
    def __init__(self, tokenizer, w2v, embed_size=300, device='cpu'):
        """
        w2v: gensim 的 KeyedVectors
        embed_size: 词向量维度
        """
        self.tokenizer = tokenizer
        self.w2v = w2v
        self.embed_size = embed_size
        self.device = device

    def collate(self, batch):
        sentences = [b["sentence"] for b in batch]
        labels    = torch.tensor([b["label"] for b in batch],
                                dtype=torch.float, device=self.device)

        # 1) 批量分词，一次性拿到 offsets_mapping
        tok_out = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
            return_offsets_mapping=True
        )
        input_ids      = tok_out["input_ids"].to(self.device)
        attention_mask = tok_out["attention_mask"].to(self.device).bool()

        # 2) 循环调用 .word_ids(i) 拿 word_ids
        batch_word_embeds = []
        for i, sentence in enumerate(sentences):
            # 2.1 用 word_ids 映射子词 -> 单词索引
            word_ids: List[int|None] = tok_out.word_ids(batch_index=i)

            # 2.2 取出这条句子的 word2vec 序列
            words = sentence.lower().split()
            word_embeds = [
                self.w2v[w] if w in self.w2v
                else np.zeros(self.embed_size, dtype=np.float32)
                for w in words
            ]

            # 2.3 广播到子词级
            seq_len = len(word_ids)
            emb = np.zeros((seq_len, self.embed_size), dtype=np.float32)
            for tidx, widx in enumerate(word_ids):
                if widx is not None:
                    emb[tidx] = word_embeds[widx]

            batch_word_embeds.append(torch.from_numpy(emb))

        # 3) pad 并搬到 device
        padded_embeds = pad_sequence(
            batch_word_embeds, batch_first=True, padding_value=0.0
        ).to(self.device)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "embeds":         padded_embeds,
            "labels":         labels
        }