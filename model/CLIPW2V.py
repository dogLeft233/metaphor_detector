from transformers import CLIPModel
from torch import nn
from model.loss_fn.cca_loss import cca_loss
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from typing import Dict
import torch
from peft import LoraConfig, get_peft_model

class W2VDisambiguator(nn.Module):
    def __init__(self, dropout:float = 0.1,input_dim:int = 768, hidden_dim:int = 512, output_dim:int = 300):
        super().__init__()
        
        self.output_dim = output_dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def _calculate_loss(self, text_disambiguated_embeds:torch.Tensor, image_disambiguated_embeds:torch.Tensor, labels:torch.Tensor)->torch.Tensor:
        batch_size = image_disambiguated_embeds.size(0)
        
        image_labels = labels.view(batch_size, 5, -1).mean(dim=1, keepdim=False)
        text2image = text_disambiguated_embeds.view(batch_size, 5, -1).mean(dim=1, keepdim=False)
        
        image_loss = (1 - F.cosine_similarity(image_disambiguated_embeds, image_labels, dim=-1)).mean(dim=-1)
        text_loss = (1 - F.cosine_similarity(text_disambiguated_embeds, labels, dim=-1)).mean(dim=-1)
        alignment_loss = (1 - F.cosine_similarity(image_disambiguated_embeds, text2image, dim=-1)).mean(dim=-1)
        
        loss = image_loss + text_loss + alignment_loss
        return loss
        
    def forward(self, text_embeds:torch.Tensor, image_embeds:torch.Tensor, labels:torch.Tensor = None):
        #(5 * B, D)
        text_embeds = text_embeds.view(-1, image_embeds.size(-1))
        
        #(5 * B, H) (B, H)
        text_hidden_embeds = self.input_proj(text_embeds)
        image_hidden_embeds = self.input_proj(image_embeds)
        
        text_features = self.ffn(text_hidden_embeds) + text_hidden_embeds
        image_features = self.ffn(image_hidden_embeds) + image_hidden_embeds
        
        #(5 * B, O) (B, O)
        text_disambiguated_embeds = self.output(text_features)
        image_disambiguated_embeds = self.output(image_features)
        
        if labels != None:
            #(5 * B, O)
            labels = labels.view(-1, self.output_dim)
            
            return self._calculate_loss(text_disambiguated_embeds, image_disambiguated_embeds, labels)
        return {
            "text_disambiguated_embeds":text_disambiguated_embeds,
            "image_disambiguated_embeds":image_disambiguated_embeds
        }

class CLIPEncoder(nn.Module):
    def __init__(self, dropout:float = 0.1, mlp_hidden_size:int = 400):
        super().__init__()
        
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        self.clip_output_dim = self.clip.projection_dim
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
        # # 解冻LayerNorm层
        # for name, module in gpt2_model.named_modules():
        #     if isinstance(module, torch.nn.LayerNorm):
        #         for param in module.parameters():
        #             param.requires_grad = True
        # # 再解冻Position Embedding
        # gpt2_model.wpe.weight.requires_grad = True
        
        self.gpt2 = gpt2_model
        
        self.clip2gpt = nn.Sequential(
            nn.Linear(img_feature_dim , gpt2_model.config.n_embd),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.w2v2gpt = nn.Sequential(
            nn.Linear(w2v_dim , gpt2_model.config.n_embd),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.prompt_embedding = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(self.prompt_len, gpt2_model.config.n_embd)), requires_grad=False)
        
        self.scale = torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(1)))
        
        self.classifier = nn.Sequential(
            nn.Linear(gpt2_model.config.n_embd, gpt2_model.config.n_embd),
            nn.ReLU(),
            nn.Linear(gpt2_model.config.n_embd, num_classes)
        )
    
    @torch.no_grad()
    def get_init_prompt_embedding(self) -> torch.Tensor:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        wte = self.gpt2.get_input_embeddings().weight

        prompts = "Sentiment: Intention: Offensiveness: Metaphor:"
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompts))
        emb = wte[torch.tensor(ids, dtype=torch.long, device=wte.device)]
            
        return emb
        
    def forward(self, text_embeds:torch.Tensor, image_embeds:torch.Tensor, shifted_image_embeds:torch.Tensor, shifted_text_embeds:torch.Tensor, labels:torch.Tensor = None):
        B = text_embeds.size(0)
        image_gpt = self.clip2gpt(image_embeds).unsqueeze(1)
        text_gpt = self.clip2gpt(text_embeds).unsqueeze(1)
        shifted_image_gpt = self.w2v2gpt(shifted_image_embeds).unsqueeze(1)
        shifted_text_gpt = self.w2v2gpt(shifted_text_embeds).unsqueeze(1)
        
        feature = torch.cat([image_gpt, text_gpt, shifted_image_gpt, shifted_text_gpt], dim=1)
        
        # #(B, 2* I + 2* W)
        # input_embed = torch.cat([text_embeds, image_embeds, shifted_text_embeds, shifted_image_embeds], dim=-1)
        # input_embed = self.dropout(input_embed)
        
        # #(B, 1, D)
        # feature = self.ffn(input_embed).unsqueeze(1)
        
        combined_embeds = torch.cat([self.prompt_embedding.expand(B, -1, -1), feature], dim=1)
        combined_embeds = self.dropout(combined_embeds)
        
        outputs = self.gpt2(inputs_embeds = combined_embeds)
        alpha = torch.sigmoid(self.scale)
        
        h = alpha * outputs.last_hidden_state[:, -4: , :].mean(dim=1) + (1 - alpha) * outputs.last_hidden_state[:, -5, :]
        h = self.dropout(h)
        
        logits = self.classifier(h)
        
        if labels != None:
            return F.cross_entropy(logits, labels)
        return logits
    
class GPT2LoRAClassifier(nn.Module):
    def __init__(
        self,
        dropout: float = 0.1,
        num_classes: int = 2,
        img_feature_dim: int = 512,
        w2v_dim: int = 300,
        prompt_len: int = 10,
        lora_r: int = 4,
        lora_alpha: int = 16,
    ):
        super().__init__()
        self.prompt_len = prompt_len
        
        config = GPT2Config.from_pretrained("gpt2", output_hidden_states=True)
        base_model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
        for p in base_model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["c_attn"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.gpt2 = get_peft_model(base_model, lora_config)

        # 3) Soft prompt embeddings
        # self.prompt_embedding = nn.Parameter(
        #     torch.randn(prompt_len, self.gpt2.config.n_embd) * 0.02
        # )
        
        self.prompt_embedding = nn.Parameter(
            self.get_init_prompt_embedding()
        )

        self.clip2gpt = nn.Sequential(
            nn.Linear(img_feature_dim, self.gpt2.config.n_embd),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.w2v2gpt = nn.Sequential(
            nn.Linear(w2v_dim, self.gpt2.config.n_embd),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.scale = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.gpt2.config.n_embd, self.gpt2.config.n_embd),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.gpt2.config.n_embd, num_classes),
        )
        
    @torch.no_grad()
    def get_init_prompt_embedding(self) -> torch.Tensor:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        wte = self.gpt2.get_input_embeddings().weight

        prompts = "Sentiment: Intention: Offensiveness: Metaphor:"
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompts))
        emb = wte[torch.tensor(ids, dtype=torch.long, device=wte.device)]
            
        return emb

    def forward(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        shifted_image_embeds: torch.Tensor,
        shifted_text_embeds: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        B = text_embeds.size(0)
        image_gpt = self.clip2gpt(image_embeds).unsqueeze(1)
        text_gpt = self.clip2gpt(text_embeds).unsqueeze(1)
        shifted_image_gpt = self.w2v2gpt(shifted_image_embeds).unsqueeze(1)
        shifted_text_gpt = self.w2v2gpt(shifted_text_embeds).unsqueeze(1)
        feature = torch.cat([
            image_gpt,
            text_gpt,
            shifted_image_gpt,
            shifted_text_gpt,
        ], dim=1)

        prompt = self.prompt_embedding.unsqueeze(0).expand(B, -1, -1)
        inputs_embeds = torch.cat([prompt, feature], dim=1)
        inputs_embeds = self.dropout(inputs_embeds)

        outputs = self.gpt2(inputs_embeds=inputs_embeds)
        hidden = outputs.hidden_states[-1]  # (B, seq_len, D)
        alpha = torch.sigmoid(self.scale)
        h_last4 = hidden[:, -4:, :].mean(dim=1)
        h_pre = hidden[:, -5, :]
        h = alpha * h_last4 + (1 - alpha) * h_pre
        h = self.dropout(h)

        logits = self.classifier(h)
        if labels is not None:
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