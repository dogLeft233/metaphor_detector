from transformers import CLIPModel, CLIPProcessor
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from typing import List
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from typing import Dict
from model.module import FFNResidualBlock

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
    
    def __init__(self, dropout: float = 0.1, input_dim: int = 768, hidden_dim: int = 768, output_dim: int = 300):
        """
        初始化消歧器
        
        Args:
            dropout (float): Dropout率
            input_dim (int): 输入维度（CLIP编码维度），默认为768
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出维度（Word2Vec维度），默认为300
        """
        super().__init__()
        
        self.output_dim = output_dim
        
        self.doprout = nn.Dropout(dropout)
        
        # 输入投影层：将CLIP编码投影到隐藏空间
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 前馈网络：多层特征提取
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            FFNResidualBlock(hidden_dim),
            FFNResidualBlock(hidden_dim),
            nn.Dropout(dropout),
        )
        
        # 输出层：投影到Word2Vec空间
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def _calculate_loss(self, solo_text_embeds: torch.Tensor, 
                       solo_image_embeds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算消歧损失
        
        损失包含三个部分：
        1. 图像嵌入与标签的余弦相似度损失
        2. 文本嵌入与标签的余弦相似度损失
        3. 图像嵌入与文本嵌入的对齐损失
        
        Args:
            solo_text_embeds (torch.Tensor): 消歧后的文本嵌入
            solo_image_embeds (torch.Tensor): 消歧后的图像嵌入
            labels (torch.Tensor): Word2Vec标签嵌入
            
        Returns:
            torch.Tensor: 总损失值
        """
        batch_size = solo_image_embeds.size(0)
        
        # 计算图像标签（5个文本标签的平均）
        image_labels = labels.view(batch_size, 5, -1).mean(dim=1, keepdim=False)
        text2image = solo_text_embeds.view(batch_size, 5, -1).mean(dim=1, keepdim=False)
        
        # 计算各种损失
        image_loss = (1 - F.cosine_similarity(solo_image_embeds, image_labels, dim=-1)).mean(dim=-1)
        text_loss = (1 - F.cosine_similarity(solo_text_embeds, labels, dim=-1)).mean(dim=-1)
        alignment_loss = (1 - F.cosine_similarity(solo_image_embeds, text2image, dim=-1)).mean(dim=-1)
        
        # 总损失
        loss = image_loss + text_loss + alignment_loss
        return loss
    
    def encode(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor, **kwargs)->Dict[str, torch.Tensor]:
         # 重塑文本嵌入
        text_embeds = text_embeds.view(-1, image_embeds.size(-1))
        
        # 输入投影
        text_hidden_embeds = self.input_proj(text_embeds)
        image_hidden_embeds = self.input_proj(image_embeds)
        
        # 输出投影到Word2Vec空间
        text_hidden_embeds = self.output(text_hidden_embeds)
        image_hidden_embeds = self.output(image_hidden_embeds)
        
        return {
            "solo_text_embeds": text_hidden_embeds,
            "solo_image_embeds": image_hidden_embeds
        }
        
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
        output = self.encode(text_embeds, image_embeds)
        
        if labels is not None:
            # 训练模式：计算损失
            labels = labels.view(-1, self.output_dim)
            return self._calculate_loss(output["solo_text_embeds"], output["solo_image_embeds"], labels)
        
        # 推理模式：返回消歧后的嵌入
        return {
            "solo_text_embeds": output["solo_text_embeds"],
            "solo_image_embeds": output["solo_image_embeds"]
        }
        
class ContrastiveEncoder(nn.Module):
    def __init__(self,dropout:float = 0.2, context_dim:int = 768, solo_dim:int = 300, hidden_dim:int = 768, outout_dim : int = 768, margin:float = 0.5) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        self.context_proj = nn.Sequential(
            nn.LayerNorm(context_dim * 2),
            nn.Linear(context_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            FFNResidualBlock(hidden_dim),
            nn.Dropout(dropout),
            FFNResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, outout_dim)
        )
        
        self.solo_proj = nn.Sequential(
            nn.LayerNorm(solo_dim * 2),
            nn.Linear(solo_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, outout_dim)
        )
        
        self.margin = margin
        
    def encode(self,
               text_embeds: torch.Tensor,
               image_embeds: torch.Tensor,
               solo_text_embeds: torch.Tensor,
               solo_image_embeds: torch.Tensor,
               **kwargs
            )->Dict[str, torch.Tensor]:
        context_input = torch.cat([text_embeds, image_embeds], dim=-1)
        context_input = self.dropout(context_input)
        context_feature = self.context_proj(context_input)
        
        solo_input = torch.cat([solo_image_embeds, solo_text_embeds], dim=-1)
        solo_input = self.dropout(solo_input)
        solo_feature = self.solo_proj(solo_input)
        
        distance = 1 - F.cosine_similarity(context_feature, solo_feature, dim=-1)
        
        return{
            "context_feature":context_feature,
            "solo_feature":solo_feature,
            "distance":distance   
        }
        
    def _calculate_loss(self, distances:torch.Tensor, labels:torch.Tensor)->torch.Tensor:
        pos_loss = labels * torch.clamp(self.margin - distances, min=0) ** 2
        neg_loss = (1 - labels) * distances ** 2
        return (pos_loss + neg_loss).mean()
        
    def forward(
            self,
            text_embeds: torch.Tensor,
            image_embeds: torch.Tensor,
            solo_image_embeds: torch.Tensor,
            solo_text_embeds: torch.Tensor,
            labels: torch.Tensor = None,
        ):
        output = self.encode(
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            solo_text_embeds=solo_text_embeds,
            solo_image_embeds=solo_image_embeds
        )
        
        if labels != None:
            # Contrastive Loss
            distance = output["distance"]
            return self._calculate_loss(distances=distance, labels=labels)
        return output
    
class ContrastiveEncoder2(nn.Module):
    def __init__(self,
                 dropout:float = 0.2,
                 context_dim:int = 768,
                 solo_dim:int = 300,
                 hidden_dim:int = 768,
                 outout_dim : int = 768,
                 margin:float = 0.5,
                 num_intention_classes:int = 5,
                 intention_coefficient:float = 0.0,
                 num_sentiment_classes:int = 7,
                 sentiment_coefficient:float = 0.0,
                 num_offensiveness_classes:int = 4,
                 offensiveness_coefficient:float = 0.0
                ) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        self.context_proj = nn.Sequential(
            nn.LayerNorm(context_dim * 2),
            nn.Linear(context_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            FFNResidualBlock(hidden_dim),
            nn.Dropout(dropout),
            FFNResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, outout_dim)
        )
        
        self.solo_proj = nn.Sequential(
            nn.LayerNorm(solo_dim * 2),
            nn.Linear(solo_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, outout_dim)
        )
        
        self.margin = margin
        
        self.intention_classifier = nn.Linear(outout_dim, num_intention_classes)
        self.intention_coefficient = intention_coefficient
        
        self.sentiment_classifier = nn.Linear(outout_dim, num_sentiment_classes)
        self.sentiment_coefficient = sentiment_coefficient
        
        self.offensiveness_classifier = nn.Linear(outout_dim, num_offensiveness_classes)
        self.offensiveness_coefficient = offensiveness_coefficient
        
    def encode(self,
               text_embeds: torch.Tensor,
               image_embeds: torch.Tensor,
               solo_text_embeds: torch.Tensor,
               solo_image_embeds: torch.Tensor,
               **kwargs
            )->Dict[str, torch.Tensor]:
        context_input = torch.cat([text_embeds, image_embeds], dim=-1)
        context_input = self.dropout(context_input)
        context_feature = self.context_proj(context_input)
        
        solo_input = torch.cat([solo_image_embeds, solo_text_embeds], dim=-1)
        solo_input = self.dropout(solo_input)
        solo_feature = self.solo_proj(solo_input)
        
        distance = 1 - F.cosine_similarity(context_feature, solo_feature, dim=-1)
        
        return{
            "context_feature":context_feature,
            "solo_feature":solo_feature,
            "distance":distance   
        }
        
    def _calculate_loss(self,
                        distances:torch.Tensor,
                        labels:torch.Tensor,
                        intention_logits:torch.Tensor,
                        intention_detection:torch.Tensor,
                        intention_coefficient:float,
                        sentiment_logits:torch.Tensor,
                        sentiment_category:torch.Tensor,
                        sentiment_coefficient:float,
                        offensiveness_logits:torch.Tensor,
                        offensiveness_detection:torch.Tensor,
                        offensiveness_coefficient:float
                    )->torch.Tensor:
        pos_loss = labels * torch.clamp(self.margin - distances, min=0) ** 2
        neg_loss = (1 - labels) * distances ** 2
        contrast_loss = (pos_loss + neg_loss).mean()
        
        intention_loss = F.cross_entropy(intention_logits, intention_detection)
        sentiment_loss = F.cross_entropy(sentiment_logits, sentiment_category)
        offensiveness_loss = F.cross_entropy(offensiveness_logits, offensiveness_detection)
        
        return contrast_loss + intention_coefficient * intention_loss + sentiment_coefficient * sentiment_loss + offensiveness_coefficient * offensiveness_loss
        
    def forward(
            self,
            text_embeds: torch.Tensor,
            image_embeds: torch.Tensor,
            solo_image_embeds: torch.Tensor,
            solo_text_embeds: torch.Tensor,
            labels: torch.Tensor = None,
            intention_detection:torch.Tensor = None,
            sentiment_category:torch.Tensor = None,
            offensiveness_detection:torch.Tensor = None
        ):
        output = self.encode(
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            solo_text_embeds=solo_text_embeds,
            solo_image_embeds=solo_image_embeds
        )
        
        context_feature = output["context_feature"]
        intention_logits = self.intention_classifier(context_feature)
        sentiment_logits = self.sentiment_classifier(context_feature)
        offensive_logits = self.offensiveness_classifier(context_feature)
        
        if labels != None:
            # Contrastive Loss
            distance = output["distance"]
            return self._calculate_loss(distances=distance,
                                        labels=labels,
                                        intention_logits=intention_logits,
                                        intention_detection=intention_detection,
                                        intention_coefficient= self.intention_coefficient,
                                        sentiment_logits=sentiment_logits,
                                        sentiment_category=sentiment_category,
                                        sentiment_coefficient=self.sentiment_coefficient,
                                        offensiveness_logits=offensive_logits,
                                        offensiveness_detection=offensiveness_detection,
                                        offensiveness_coefficient=self.offensiveness_coefficient
                                    )
        return output

class MLPclassifier2(nn.Module):
    def __init__(self, input_dim:int = 1, hidden_dim:int = 4, num_classes:int =2):
        super().__init__()
        
        self.encoder = ContrastiveEncoder2()
        
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self,
                text_embeds: torch.Tensor, 
                image_embeds: torch.Tensor,
                solo_image_embeds: torch.Tensor,
                solo_text_embeds: torch.Tensor,
                labels: torch.Tensor = None,
                **kwargs
            ):
        
        feature = self.encoder.encode(text_embeds, image_embeds, solo_image_embeds, solo_text_embeds)
        
        distance = feature["distance"].unsqueeze(1)
        logits = self.ffn(distance)
        
        if labels is not None:
            return F.cross_entropy(logits, labels)
        return logits
 
class MLPclassifier(nn.Module):
    def __init__(self,dropout:float = 0.2, input_dim:int = 768, hidden_dim:int = 768, num_classes:int =2):
        super().__init__()
        
        self.encoder = ContrastiveEncoder2(dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            FFNResidualBlock(hidden_dim),
            FFNResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        self.classifier = nn.Linear(2, num_classes)
        
    def forward(self,
                text_embeds: torch.Tensor, 
                image_embeds: torch.Tensor,
                solo_image_embeds: torch.Tensor,
                solo_text_embeds: torch.Tensor,
                labels: torch.Tensor = None,
                **kwargs
            ):
        
        feature = self.encoder.encode(text_embeds, image_embeds, solo_image_embeds, solo_text_embeds)
        
        distance = feature["distance"].unsqueeze(1)
        feature = torch.cat([feature["context_feature"], feature["solo_feature"]], dim=-1)
        feature = self.dropout(feature)
        feature = self.ffn(feature)
        feature = torch.cat([feature, distance], dim=-1)
        logits = self.classifier(feature)
        
        if labels is not None:
            return F.cross_entropy(logits, labels)
        return logits