from transformers import CLIPModel
from torch import nn
from model.loss_fn.cca_loss import cca_loss
import torch

class CLIPEncoder(nn.Module):
    def __init__(self, dropout:float = 0.1, mlp_hidden_size:int = 400):
        super().__init__()
        
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        self.clip_output_dim = 512
        self.mlp_output_dim = 300
        
        self.mlp = nn.Sequential(
            nn.Linear(self.clip_output_dim, 400),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(400, self.mlp_output_dim)
        )
        
    def forward(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, pixel_values:torch.Tensor, sentence_embeds:torch.Tensor)->torch.Tensor:
        batch_size = pixel_values.size(0)
        outputs = self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        
        # 处理image_embeds
        image_embeds = outputs.image_embeds.unsqueeze(1)  # (B, 1, 512)
        text_embeds = outputs.text_embeds
        sentence_embeds = sentence_embeds
        
        # 通过MLP降维
        image_embeds = self.mlp(image_embeds)
        text_embeds = self.mlp(text_embeds)
        
        image_embeds_expand = image_embeds.expand((-1, 5, -1)).reshape(-1, self.mlp_output_dim)
        
        image_loss = cca_loss(image_embeds_expand, sentence_embeds)
        text_loss = cca_loss(text_embeds, sentence_embeds)
        
        loss = image_loss + text_loss
        
        return loss