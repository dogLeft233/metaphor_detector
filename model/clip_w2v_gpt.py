from transformers import CLIPModel, CLIPProcessor
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from typing import List
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model

def detect_metaphor(clip:CLIPModel, clip_processor:CLIPProcessor, disambiguator:nn.Module, classifier:nn.Module, texts:List[str], image:Image)->torch.Tensor:
    
    clip_inputs = clip_processor(
            text=texts,
            images = image,
            return_tensors="pt",
            padding = True,
            truncation=True,
            max_length=77
        )
    
    clip_output = clip(**clip_inputs)
    text_embeds = clip_output.text_embeds
    image_embeds = clip_output.image_embeds
    
    disambiguator_output = disambiguator(text_embeds = text_embeds, image_embeds = image_embeds)
    shifted_text_embeds = disambiguator_output["text_disambiguated_embeds"]
    shifted_image_embeds = disambiguator_output["image_disambiguated_embeds"]
    
    logits = classifier(
            text_embeds = text_embeds,
            image_embeds = image_embeds,
            shifted_image_embeds = shifted_image_embeds,
            shifted_text_embeds = shifted_text_embeds
        )
    
    preds = torch.argmax(logits, dim=-1)
    
    return preds
    

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
    
class GPT2LoRAClassifier(nn.Module):
    def __init__(
        self,
        dropout: float = 0.1,
        num_classes: int = 2,
        img_feature_dim: int = 768,
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
        
        self.prompt_embedding = nn.Parameter(
            self.get_init_prompt_embedding()
        )

        self.clip2gpt = nn.Sequential(
            nn.Linear(img_feature_dim * 2, img_feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(img_feature_dim, self.gpt2.config.n_embd)
        )
        self.w2v2gpt = nn.Sequential(
            nn.Linear(w2v_dim, self.gpt2.config.n_embd),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

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

        prompts = "Sentiment? Intention? Offensiveness? Metaphor?"
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
        
        context_feature = torch.cat([image_embeds, text_embeds], dim=-1)
        context_gpt_input = self.clip2gpt(context_feature).unsqueeze(1)
        
        shifted_image_gpt_input = self.w2v2gpt(shifted_image_embeds).unsqueeze(1)
        shifted_text_gpt_input = self.w2v2gpt(shifted_text_embeds).unsqueeze(1)
        feature = torch.cat([
            shifted_image_gpt_input,
            shifted_text_gpt_input,
            context_gpt_input,
        ], dim=1)

        prompt = self.prompt_embedding.unsqueeze(0).expand(B, -1, -1)
        inputs_embeds = torch.cat([prompt, feature], dim=1)
        inputs_embeds = self.dropout(inputs_embeds)

        outputs = self.gpt2(inputs_embeds=inputs_embeds)
        hidden = outputs.hidden_states[-1]  # (B, seq_len, D)
        h = hidden[:, -1, :]

        logits = self.classifier(h)
        if labels is not None:
            return F.cross_entropy(logits, labels)
        return logits