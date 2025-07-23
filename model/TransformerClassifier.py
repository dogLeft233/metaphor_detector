from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from model.module.SinusoidalPosEmb import SinusoidalPosEmb
import torch
import torch.nn.functional as F

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size:int, hidden_size:int, max_len:int, num_layers:int, nhead:int, dropout:float):
        super(TransformerClassifier, self).__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_embedding = SinusoidalPosEmb(hidden_size, max_len)
        
        self.transformer_layers =TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(hidden_size, nhead=nhead, batch_first=False, dropout=dropout),
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size)   
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
    def forward(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, labels:torch.Tensor)->torch.Tensor:
        input_ids.transpose_(0, 1)
        
        embed = self.token_embedding(input_ids)
        embed = self.positional_embedding(embed)
        
        encoder_outputs = self.transformer_layers(src=embed, src_key_padding_mask=~attention_mask)[0]
        pred = self.classifier(encoder_outputs).view(-1)
        
        return F.binary_cross_entropy_with_logits(pred, labels,pos_weight=torch.tensor(6.825, device=pred.device))