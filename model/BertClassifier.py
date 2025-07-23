from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from model.module.SinusoidalPosEmb import SinusoidalPosEmb
import torch
import torch.nn.functional as F
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, dropout:float):
        super(BertClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        hidden_size = self.bert.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
    def forward(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, labels:torch.Tensor)->torch.Tensor:
        
        encoder_outputs = self.bert(
            input_ids,
            attention_mask
        ).last_hidden_state
        pred = self.classifier(encoder_outputs.last_hidden_state[:,0,:]).view(-1)
        
        return F.binary_cross_entropy_with_logits(pred, labels,pos_weight=torch.tensor(6.825, device=pred.device))