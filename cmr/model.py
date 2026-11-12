"""双塔模型：图像编码器 + 文本编码器 + 可学习温度。

    图像 → ResNet 骨干 → 投影头 ─┐
                                 ├─→ L2 归一化 → 共享嵌入空间（对比学习对齐）
    文本 → 多语言 BERT → 投影头 ─┘

骨干网络用 ImageNet / 多语言预训练权重初始化（Flickr8k 规模的数据无法从随机
初始化训出有效嵌入），投影头、对比损失、训练与评测流程全部自行实现。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models as tv_models
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:  # 避免运行时循环导入
    from .config import ModelConfig


class ProjectionHead(nn.Module):
    """两层 MLP 投影头: in_dim → embed_dim → embed_dim。"""

    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def mean_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """按 attention mask 做平均池化。hidden: (B, L, H), mask: (B, L)。"""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class ImageEncoder(nn.Module):
    """torchvision ResNet 骨干 + 投影头。输出未归一化的嵌入。"""

    def __init__(
        self,
        arch: str = "resnet50",
        pretrained: bool = True,
        embed_dim: int = 512,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        builder = getattr(tv_models, arch, None)
        if builder is None:
            raise ValueError(f"不支持的图像骨干: {arch!r}（可选 resnet18/34/50/101）")
        weights = "DEFAULT" if pretrained else None  # torchvision 会自动下载预训练权重
        backbone = builder(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()  # 丢弃分类头，保留 2048 维特征
        self.backbone = backbone
        self.projection = ProjectionHead(feat_dim, embed_dim)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, images: Tensor) -> Tensor:
        return self.projection(self.backbone(images))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # 冻结时 BN 统计量不更新
        return self


class TextEncoder(nn.Module):
    """HuggingFace 预训练多语言编码器 + 投影头。输出未归一化的嵌入。"""

    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        embed_dim: int = 512,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_dim = self.backbone.config.hidden_size
        self.projection = ProjectionHead(hidden_dim, embed_dim)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        return self.projection(pooled)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # 冻结时 dropout 关闭、统计量不更新
        return self


class DualEncoder(nn.Module):
    """双塔封装：编码、归一化、按 CLIP 惯例计算带温度的相似度矩阵。"""

    def __init__(
        self,
        image_encoder: ImageEncoder,
        text_encoder: TextEncoder,
        init_temperature: float = 0.07,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.full((), _log(1.0 / init_temperature)))

    # ---- 编码（L2 归一化，点积即余弦相似度） ----
    def encode_image(self, images: Tensor) -> Tensor:
        return F.normalize(self.image_encoder(images), dim=-1)

    def encode_text(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return F.normalize(self.text_encoder(input_ids, attention_mask), dim=-1)

    def forward(self, images: Tensor, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """返回 (B, B) 的 logits 矩阵，logits[i, j] = scale * cos(image_i, text_j)。"""
        img = self.encode_image(images)
        txt = self.encode_text(input_ids, attention_mask)
        scale = self.logit_scale.exp().clamp(max=100.0)  # 与 CLIP 相同的上界
        return scale * img @ txt.t()

    # ---- 冻结控制由各编码器自身的 train() 覆写处理 ----

    def param_groups(self, backbone_lr: float, head_lr: float) -> list:
        """骨干与投影头/温度使用不同学习率的参数组。"""
        backbone_params, head_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (backbone_params if ".backbone." in f".{name}." else head_params).append(p)
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
        return groups

    @property
    def temperature(self) -> float:
        return 1.0 / self.logit_scale.exp().item()


def _log(x: float) -> float:
    import math

    return math.log(x)


def build_model(cfg: "ModelConfig") -> DualEncoder:
    image_encoder = ImageEncoder(
        arch=cfg.image_arch,
        pretrained=cfg.image_pretrained,
        embed_dim=cfg.embed_dim,
        freeze_backbone=cfg.freeze_image_backbone,
    )
    text_encoder = TextEncoder(
        model_name=cfg.text_model,
        embed_dim=cfg.embed_dim,
        freeze_backbone=cfg.freeze_text_backbone,
    )
    return DualEncoder(image_encoder, text_encoder)
