from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.clip_handler import CLIPHandler

def sliding_chunks(token_ids: List[int], max_len: int = 77, stride: int = 38):
    """
    将输入的 token ID 序列切分为多个重叠片段，使用 pad_sequence 填充，并构造 attention mask。

    Args:
        token_ids (List[int]): 原始 token ID 列表（不含 special tokens）
        max_len (int): 每个块的最大 token 数（包含特殊 tokens）
        stride (int): 滑动步长（下一个片段起点距离上一个片段起点的距离）

    Returns:
        Dict[str, torch.Tensor]: 包含 'input_ids' 和 'attention_mask'，
            shape 分别为 (num_chunks, max_len)
    """

    chunks = []
    length = len(token_ids)
    for start in range(0, length, stride):
        end = min(start + max_len, length)
        sub_ids = token_ids[start:end]
        chunks.append(torch.tensor(sub_ids, dtype=torch.long))
        if end == length:
            break

    padded = pad_sequence(chunks, batch_first=True, padding_value=CLIPHandler.TOKENIZER.pad_token_id)

    attention_mask = (padded != CLIPHandler.TOKENIZER.pad_token_id).long()

    return {"input_ids": padded, "attention_mask": attention_mask}