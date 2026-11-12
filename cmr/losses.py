"""对称 InfoNCE（CLIP 风格）对比损失。

logits 矩阵约定: `logits[i, j] = scale * <image_i, text_j>`，对角线位置为正样本对。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def clip_loss(logits_per_image: Tensor) -> Tensor:
    """图→文 与 文→图 两个方向交叉熵的平均。

    batch 内第 i 张图与第 i 条文本互为正样本，其余为负样本。
    """
    labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_image.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)


def batch_top1_accuracy(logits_per_image: Tensor) -> float:
    """训练批内图→文 top-1 命中率（仅用于监控）。"""
    labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    preds = logits_per_image.argmax(dim=1)
    return (preds == labels).float().mean().item()
