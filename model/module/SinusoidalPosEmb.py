from torch import nn
import torch
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000) -> None:
        super(SinusoidalPosEmb, self).__init__()
        assert dim % 2 == 0
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算频率
        div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                           (-math.log(10000.0) / dim))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度并注册为缓冲区
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 [seq_len, batch_size, embedding_dim]
        Returns:
            添加位置编码的张量
        """
        return x + self.pe[:x.size(0), :]