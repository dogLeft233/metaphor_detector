from transformers import CLIPModel
from torch import nn
from model.loss_fn.cca_loss import cca_loss
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer
from typing import Dict
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
            nn.Linear(400, self.mlp_output_dim),
            nn.LayerNorm(self.mlp_output_dim)
        )
        
    def encode(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, pixel_values:torch.Tensor)->Dict[str,torch.Tensor]:
        outputs = self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        
        shifted_image_embeds = self.mlp(image_embeds)
        shifted_text_embeds = self.mlp(text_embeds)
        
        return {
            "text_embeds":image_embeds,
            "image_embeds":image_embeds,
            "shifted_image_embeds":shifted_image_embeds,
            "shifted_text_embeds":shifted_text_embeds
        }
        
        
    def forward(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, pixel_values:torch.Tensor, sentence_embeds:torch.Tensor)->torch.Tensor:
        batch_size = pixel_values.size(0)
        outputs = self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        
        # 处理image_embeds
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        sentence_embeds = sentence_embeds
        
        # 通过MLP降维
        image_embeds = self.mlp(image_embeds)
        text_embeds = self.mlp(text_embeds)
        
        w2v_img_embeds = sentence_embeds.view(batch_size, 5, -1).mean(dim = 1, keepdim = False)
        text_img_embeds = text_embeds.view(batch_size, 5, -1).mean(dim = 1, keepdim = False)
        
        image_loss = (1 - F.cosine_similarity(image_embeds, w2v_img_embeds, dim=-1)).mean(dim=-1)
        text_loss = (1 - F.cosine_similarity(text_embeds, sentence_embeds, dim=-1)).mean(dim=-1)
        alignment_loss = (1 - F.cosine_similarity(image_embeds, text_img_embeds, dim=-1)).mean(dim=-1)
        
        loss = image_loss + text_loss + 0.5 * alignment_loss
        
        return loss
    
class GPTClassifier(nn.Module):
    def __init__(self, dropout:float = 0.1,num_classes = 2, img_feature_dim:int = 512, w2v_dim:int = 300, prompt_len:int = 10) -> None:
        super().__init__()
        self.prompt_len = prompt_len
        
        self.dropout = nn.Dropout(dropout)
        gpt2_model = GPT2Model.from_pretrained("gpt2")
        # 冻结GPT-2所有参数
        for param in gpt2_model.parameters():
            param.requires_grad = False
        # 解冻LayerNorm层
        for name, module in gpt2_model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                for param in module.parameters():
                    param.requires_grad = True
        # 再解冻Position Embedding
        gpt2_model.wpe.weight.requires_grad = True
        
        self.gpt2 = gpt2_model
        
        self.ffn = nn.Sequential(
            nn.Linear(2 * img_feature_dim + 2 * w2v_dim, img_feature_dim + w2v_dim),
            nn.ReLU(),
            nn.Linear(img_feature_dim + w2v_dim , img_feature_dim),
            nn.ReLU(),
            nn.Linear(img_feature_dim, gpt2_model.config.n_embd)
        )
        
        #软提示
        self.prompt_embedding = nn.Parameter(
            self.get_init_prompt_embedding(),
            requires_grad=False
        )
        
        self.scale = torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(1)))
        
        self.classifier = nn.Linear(gpt2_model.config.n_embd, num_classes)
    
    @torch.no_grad()
    def get_init_prompt_embedding(self) -> torch.Tensor:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        wte = self.gpt2.get_input_embeddings().weight

        prompts = "Sentiment: Intention: Offensiveness: Metaphor:"
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompts))
        emb = wte[torch.tensor(ids, dtype=torch.long, device=wte.device)]
        print(emb.shape)
            
        return emb
        
    def forward(self, text_embeds:torch.Tensor, image_embeds:torch.Tensor, shifted_image_embeds:torch.Tensor, shifted_text_embeds:torch.Tensor, labels:torch.Tensor = None):
        B = text_embeds.size(0)
        
        #(B, 2* I + 2* W)
        input_embed = torch.cat([text_embeds, image_embeds, shifted_text_embeds, shifted_image_embeds], dim=-1)
        input_embed = self.dropout(input_embed)
        
        #(B, 1, D)
        feature = self.ffn(input_embed).unsqueeze(1)
        
        combined_embeds = torch.cat([feature, self.prompt_embedding.expand(B, -1, -1)], dim=1)
        combined_embeds = self.dropout(combined_embeds)
        
        seq_len = combined_embeds.size(1)
        pos_ids = torch.arange(seq_len, device=combined_embeds.device).unsqueeze(0)
        pos_emb = self.gpt2.wpe(pos_ids)                  # (1, P+1, D)
        combined_embeds = combined_embeds + pos_emb                     #(B,P+1,D)
        
        outputs = self.gpt2(inputs_embeds = combined_embeds)
        alpha = torch.sigmoid(self.scale)
        h = alpha * outputs.last_hidden_state[:, -1, :] + (1 - alpha) * outputs.last_hidden_state[:, -2, :]
        logits = self.classifier(h)
        
        if labels != None:
            return F.cross_entropy(logits, labels)
        return logits
    
class Meteor(nn.Module):
    def __init__(self, num_classes=2, img_feature_dim=768):
        super(Meteor, self).__init__()
        gpt2_model = GPT2Model.from_pretrained("gpt2")
        # 冻结GPT-2所有参数
        for param in gpt2_model.parameters():
            param.requires_grad = False
        # 解冻LayerNorm层
        for name, module in gpt2_model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                for param in module.parameters():
                    param.requires_grad = True
        # 再解冻Position Embedding
        gpt2_model.wpe.weight.requires_grad = True

        self.num_classes = num_classes

        self.gpt2_model = gpt2_model

        self.feature_projection = torch.nn.Linear(img_feature_dim, gpt2_model.config.n_embd)  # 图像特征转换为GPT-2的嵌入维度

        self.adapter = torch.nn.Sequential(
            torch.nn.Linear(img_feature_dim * 2, img_feature_dim),
            torch.nn.GELU()
        )

        self.prompt_embedding = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(10, gpt2_model.config.n_embd)), requires_grad=False)

        self.scale = torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(1)))
        self.classifier = torch.nn.Linear(gpt2_model.config.n_embd, num_classes)

    def forward(self, image_embeds, text_embeds, labels = None, **kwargs):
        fused_embedding = self.adapter(torch.cat([image_embeds, text_embeds], dim=1))
        fused_embedding = self.feature_projection(fused_embedding).unsqueeze(1)
        prompt_embeddings =  self.prompt_embedding.unsqueeze(0).expand(image_embeds.shape[0], -1, -1)
        combined_embeddings = torch.cat([prompt_embeddings, fused_embedding], dim=1)
        outputs = self.gpt2_model(inputs_embeds=combined_embeddings)
        alpha = torch.sigmoid(self.scale)
        last_hidden_state = alpha * outputs.last_hidden_state[:, -1, :] + (1 - alpha) * outputs.last_hidden_state[:, -2, :]
        logits = self.classifier(last_hidden_state)  # 使用最后两个位置的token的线性插值的结果进行分类
        if labels != None:
            return F.cross_entropy(logits, labels)
        return logits