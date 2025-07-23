import torch
import torch.nn as nn
from transformers import BertModel

class BertW2V(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', embed_size=300, hidden_size=768, num_classes=1):
        super().__init__()
        # 1. BERT编码器
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.hidden_size = hidden_size

        # 2. 线性层将word2vec映射到BERT同维度
        self.embed_proj = nn.Linear(embed_size, hidden_size)

        # 3. 卷积层（可根据需要调整参数）
        # 输入: (batch, hidden_size, 2, seq_len)
        self.conv = nn.Conv2d(
            in_channels=hidden_size, 
            out_channels=hidden_size, 
            kernel_size=(2, 3),  # 2表示融合BERT和w2v，3为卷积窗口长度
            padding=(0, 1)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, embeds, labels=None):
        # 1. BERT编码
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_hidden = bert_outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # 2. 线性层映射embeds
        embeds_proj = self.embed_proj(embeds)  # (batch, seq_len, hidden_size)

        # 3. 拼接
        # 先拼成 (batch, seq_len, hidden_size, 2)
        concat = torch.stack([bert_hidden, embeds_proj], dim=-1)
        # 转为 (batch, hidden_size, 2, seq_len)
        concat = concat.permute(0, 2, 3, 1)

        # 4. 卷积
        conv_out = self.conv(concat)  # (batch, hidden_size, 1, seq_len)
        conv_out = self.relu(conv_out)
        conv_out = conv_out.squeeze(2)  # (batch, hidden_size, seq_len)

        # 5. 池化（取最大池化，也可用平均池化）
        pooled = torch.max(conv_out, dim=2)[0]  # (batch, hidden_size)

        # 6. 分类
        logits = self.classifier(self.dropout(pooled)).squeeze(-1)  # (batch,)

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(6.825, device=logits.device))
            loss = loss_fn(logits, labels)
            return loss
        else:
            return torch.sigmoid(logits)