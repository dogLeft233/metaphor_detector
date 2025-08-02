from torch import nn
import torch
from typing import List

class FFNResidualBlock(nn.Module):
    def __init__(self, hidden_size:int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
    def forward(self, x)->torch.Tensor:
        return x + self.ffn(x)

class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        # 可学习的 query 向量，用于池化
        self.query = nn.Parameter(torch.randn(1, hidden_dim))
    
    def forward(self, H: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        H: Tensor[B, L, D]，序列隐藏状态
        mask: Tensor[B, L]，可选的 attention mask (1 = 有效，0 = pad)
        """
        # 计算每个位置的打分
        # scores: [B, L]
        scores = torch.einsum("bld,qd->bl", H, self.query)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # 归一化权重
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, L, 1]
        
        # 加权求和，得到 [B, D]
        pooled = torch.sum(weights * H, dim=1)
        return pooled
    
class LayerFusion(nn.Module):
    def __init__(self, num_layers: int, hidden_dim: int):
        """
        num_layers: 融合倒数多少层（如4 代表倒数第 1,2,3,4 层）
        hidden_dim: D
        """
        super().__init__()
        # Learnable 权重，初始化为相等分配
        init_val = 1.0 / num_layers
        self.weights = nn.Parameter(torch.full((num_layers,), init_val))
    
    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        hidden_states: list of Tensors [B, L, D], length >= num_layers
        取最后 num_layers 个进行加权融合
        """
        to_fuse = hidden_states[-len(self.weights):]  # list of [B, L, D]
        # 对权重做 softmax，保证归一化
        norm_w = torch.softmax(self.weights, dim=0)    # [num_layers]
        
        # 加权求和
        fused = 0
        for w, h in zip(norm_w, to_fuse):
            fused = fused + w * h   # broadcasting w [ ] -> scalar multiply
        return fused  # [B, L, D]