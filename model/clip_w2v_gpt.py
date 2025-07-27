"""
CLIP + Word2Vec + GPT 隐喻检测模型

该模块实现了一个基于CLIP、Word2Vec和GPT2的隐喻检测模型，包含以下主要组件：
1. detect_metaphor: 完整的隐喻检测流程函数
2. W2VDisambiguator: 基于Word2Vec的消歧器，用于处理多义词
3. GPT2LoRAClassifier: 基于GPT2和LoRA的分类器，用于最终分类

模型架构：
CLIP编码 -> W2V消歧器 -> GPT2分类器 -> 隐喻检测结果

主要功能：
- 多模态隐喻检测（文本+图像）
- 基于Word2Vec的语义消歧
- 使用LoRA进行高效微调
- 支持批量处理和推理
"""

from transformers import CLIPModel, CLIPProcessor
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from typing import List
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model

def detect_metaphor(clip: CLIPModel, clip_processor: CLIPProcessor, disambiguator: nn.Module, 
                   classifier: nn.Module, texts: List[str], image: Image) -> torch.Tensor:
    """
    完整的隐喻检测流程
    
    该函数实现了从原始输入到最终预测的完整隐喻检测流程：
    1. 使用CLIP编码文本和图像
    2. 使用消歧器处理多义词
    3. 使用分类器进行最终分类
    
    Args:
        clip (CLIPModel): 预训练的CLIP模型
        clip_processor (CLIPProcessor): CLIP预处理器
        disambiguator (nn.Module): 消歧器模型
        classifier (nn.Module): 分类器模型
        texts (List[str]): 输入文本列表
        image (Image): 输入图像
    
    Returns:
        torch.Tensor: 预测结果，形状为(batch_size,)，值为0或1
    """
    # 使用CLIP处理器处理输入
    clip_inputs = clip_processor(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    
    # 获取CLIP编码
    clip_output = clip(**clip_inputs)
    text_embeds = clip_output.text_embeds
    image_embeds = clip_output.image_embeds
    
    # 使用消歧器处理多义词
    disambiguator_output = disambiguator(text_embeds=text_embeds, image_embeds=image_embeds)
    shifted_text_embeds = disambiguator_output["text_disambiguated_embeds"]
    shifted_image_embeds = disambiguator_output["image_disambiguated_embeds"]
    
    # 使用分类器进行最终分类
    logits = classifier(
        text_embeds=text_embeds,
        image_embeds=image_embeds,
        shifted_image_embeds=shifted_image_embeds,
        shifted_text_embeds=shifted_text_embeds
    )
    
    # 获取预测结果
    preds = torch.argmax(logits, dim=-1)
    
    return preds


class W2VDisambiguator(nn.Module):
    """
    基于Word2Vec的语义消歧器
    
    该模块用于处理多义词的语义消歧，通过将CLIP编码转换为Word2Vec空间，
    并与预训练的Word2Vec向量进行对齐，实现语义消歧。
    
    主要功能：
    - 将CLIP编码投影到Word2Vec空间
    - 通过多层前馈网络进行特征提取
    - 计算与Word2Vec标签的对齐损失
    """
    
    def __init__(self, dropout: float = 0.1, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 300):
        """
        初始化消歧器
        
        Args:
            dropout (float): Dropout率，默认为0.1
            input_dim (int): 输入维度（CLIP编码维度），默认为768
            hidden_dim (int): 隐藏层维度，默认为512
            output_dim (int): 输出维度（Word2Vec维度），默认为300
        """
        super().__init__()
        
        self.output_dim = output_dim
        
        # 输入投影层：将CLIP编码投影到隐藏空间
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 前馈网络：多层特征提取
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出层：投影到Word2Vec空间
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def _calculate_loss(self, text_disambiguated_embeds: torch.Tensor, 
                       image_disambiguated_embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算消歧损失
        
        损失包含三个部分：
        1. 图像嵌入与标签的余弦相似度损失
        2. 文本嵌入与标签的余弦相似度损失
        3. 图像嵌入与文本嵌入的对齐损失
        
        Args:
            text_disambiguated_embeds (torch.Tensor): 消歧后的文本嵌入
            image_disambiguated_embeds (torch.Tensor): 消歧后的图像嵌入
            labels (torch.Tensor): Word2Vec标签嵌入
            
        Returns:
            torch.Tensor: 总损失值
        """
        batch_size = image_disambiguated_embeds.size(0)
        
        # 计算图像标签（5个文本标签的平均）
        image_labels = labels.view(batch_size, 5, -1).mean(dim=1, keepdim=False)
        text2image = text_disambiguated_embeds.view(batch_size, 5, -1).mean(dim=1, keepdim=False)
        
        # 计算各种损失
        image_loss = (1 - F.cosine_similarity(image_disambiguated_embeds, image_labels, dim=-1)).mean(dim=-1)
        text_loss = (1 - F.cosine_similarity(text_disambiguated_embeds, labels, dim=-1)).mean(dim=-1)
        alignment_loss = (1 - F.cosine_similarity(image_disambiguated_embeds, text2image, dim=-1)).mean(dim=-1)
        
        # 总损失
        loss = image_loss + text_loss + alignment_loss
        return loss
        
    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor, 
                labels: torch.Tensor = None):
        """
        前向传播
        
        Args:
            text_embeds (torch.Tensor): CLIP文本嵌入，形状为(5*B, D)
            image_embeds (torch.Tensor): CLIP图像嵌入，形状为(B, D)
            labels (torch.Tensor, optional): Word2Vec标签嵌入，用于训练时计算损失
            
        Returns:
            训练时返回损失值，推理时返回消歧后的嵌入字典
        """
        # 重塑文本嵌入
        text_embeds = text_embeds.view(-1, image_embeds.size(-1))
        
        # 输入投影
        text_hidden_embeds = self.input_proj(text_embeds)
        image_hidden_embeds = self.input_proj(image_embeds)
        
        # 前馈网络处理（残差连接）
        text_features = self.ffn(text_hidden_embeds) + text_hidden_embeds
        image_features = self.ffn(image_hidden_embeds) + image_hidden_embeds
        
        # 输出投影到Word2Vec空间
        text_disambiguated_embeds = self.output(text_features)
        image_disambiguated_embeds = self.output(image_features)
        
        if labels is not None:
            # 训练模式：计算损失
            labels = labels.view(-1, self.output_dim)
            return self._calculate_loss(text_disambiguated_embeds, image_disambiguated_embeds, labels)
        
        # 推理模式：返回消歧后的嵌入
        return {
            "text_disambiguated_embeds": text_disambiguated_embeds,
            "image_disambiguated_embeds": image_disambiguated_embeds
        }


class GPT2LoRAClassifier(nn.Module):
    """
    基于GPT2和LoRA的分类器
    
    该模块使用GPT2作为骨干网络，通过LoRA进行高效微调，
    结合CLIP编码和Word2Vec消歧结果进行最终的隐喻分类。
    
    主要特点：
    - 使用LoRA进行参数高效微调
    - 软提示（Soft Prompt）技术
    - 多模态特征融合
    """
    
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
        """
        初始化GPT2分类器
        
        Args:
            dropout (float): Dropout率，默认为0.1
            num_classes (int): 分类类别数，默认为2（二分类）
            img_feature_dim (int): 图像特征维度，默认为768
            w2v_dim (int): Word2Vec维度，默认为300
            prompt_len (int): 软提示长度，默认为10
            lora_r (int): LoRA的秩，默认为4
            lora_alpha (int): LoRA的缩放因子，默认为16
        """
        super().__init__()
        self.prompt_len = prompt_len
        
        # 加载预训练的GPT2模型
        config = GPT2Config.from_pretrained("gpt2", output_hidden_states=True)
        base_model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
        
        # 冻结基础模型参数
        for p in base_model.parameters():
            p.requires_grad = False

        # 配置LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["c_attn"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.gpt2 = get_peft_model(base_model, lora_config)
        
        # 初始化软提示嵌入
        self.prompt_embedding = nn.Parameter(
            self.get_init_prompt_embedding()
        )

        # CLIP特征到GPT2的投影层
        self.clip2gpt = nn.Sequential(
            nn.Linear(img_feature_dim * 2, img_feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(img_feature_dim, self.gpt2.config.n_embd)
        )
        
        # Word2Vec特征到GPT2的投影层
        self.w2v2gpt = nn.Sequential(
            nn.Linear(w2v_dim, self.gpt2.config.n_embd),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.gpt2.config.n_embd, self.gpt2.config.n_embd),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.gpt2.config.n_embd, num_classes),
        )
        
    @torch.no_grad()
    def get_init_prompt_embedding(self) -> torch.Tensor:
        """
        获取初始化的软提示嵌入
        
        使用预定义的提示文本初始化软提示嵌入，这些提示用于引导模型关注
        特定的语义方面（情感、意图、冒犯性、隐喻）。
        
        Returns:
            torch.Tensor: 初始化的软提示嵌入
        """
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        wte = self.gpt2.get_input_embeddings().weight

        # 预定义的提示文本
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
        """
        前向传播
        
        Args:
            text_embeds (torch.Tensor): CLIP文本嵌入
            image_embeds (torch.Tensor): CLIP图像嵌入
            shifted_image_embeds (torch.Tensor): 消歧后的图像嵌入
            shifted_text_embeds (torch.Tensor): 消歧后的文本嵌入
            labels (torch.Tensor, optional): 标签，用于训练时计算损失
            
        Returns:
            训练时返回损失值，推理时返回logits
        """
        B = text_embeds.size(0)
        
        # 融合CLIP特征
        context_feature = torch.cat([image_embeds, text_embeds], dim=-1)
        context_gpt_input = self.clip2gpt(context_feature).unsqueeze(1)
        
        # 处理Word2Vec特征
        shifted_image_gpt_input = self.w2v2gpt(shifted_image_embeds).unsqueeze(1)
        shifted_text_gpt_input = self.w2v2gpt(shifted_text_embeds).unsqueeze(1)
        
        # 拼接所有特征
        feature = torch.cat([
            shifted_image_gpt_input,
            shifted_text_gpt_input,
            context_gpt_input,
        ], dim=1)

        # 添加软提示
        prompt = self.prompt_embedding.unsqueeze(0).expand(B, -1, -1)
        inputs_embeds = torch.cat([prompt, feature], dim=1)
        inputs_embeds = self.dropout(inputs_embeds)

        # GPT2前向传播
        outputs = self.gpt2(inputs_embeds=inputs_embeds)
        hidden = outputs.hidden_states[-1]  # (B, seq_len, D)
        
        # 取最后一个token的隐藏状态
        h = hidden[:, -1, :]

        # 分类
        logits = self.classifier(h)
        
        if labels is not None:
            return F.cross_entropy(logits, labels)
        return logits