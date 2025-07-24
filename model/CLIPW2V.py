from transformers import CLIPModel
from torch import nn

class CLIPEncoder(nn.Module):
    def __init__(self, dropout:float = 0.1, mlp_hidden_size:int = 400):
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        self.mlp = nn.Sequential(
            nn.Linear(512, 400),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(400, 300)
        )
        
    def forward(self, )