"""检索指标：R@K（Recall at K）与 MedR（正确结果中位排名）。

约定 sim[q, c] = 查询 q 与候选 c 的相似度；
ground_truth: 每个查询的正确候选下标集合。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import Tensor


def best_rank_per_query(sim: Tensor, ground_truth: Sequence[Sequence[int]]) -> Tensor:
    """返回每个查询「最靠前的正确候选」的名次（1-based；无正确候选则为 C+1）。"""
    n_query, n_cand = sim.shape
    order = sim.argsort(dim=1, descending=True)  # order[q, r] = 排在第 r 位的候选
    # inv[q, c] = 候选 c 在查询 q 结果中的名次（0-based）
    inv = torch.empty_like(order)
    inv.scatter_(
        1,
        order,
        torch.arange(n_cand, device=sim.device).unsqueeze(0).expand_as(order),
    )

    sizes = torch.tensor([len(g) for g in ground_truth], dtype=torch.long, device=sim.device)
    q_idx = torch.repeat_interleave(torch.arange(n_query, device=sim.device), sizes)
    c_idx = torch.tensor([c for g in ground_truth for c in g], dtype=torch.long, device=sim.device)
    ranks = inv[q_idx, c_idx] + 1

    best = torch.full((n_query,), n_cand + 1, dtype=torch.long, device=sim.device)
    best.scatter_reduce_(0, q_idx, ranks, reduce="amin", include_self=True)
    return best


def retrieval_metrics(sim: Tensor, ground_truth: Sequence[Sequence[int]], ks=(1, 5, 10)) -> Dict[str, float]:
    """计算 R@K 与 MedR。sim 行=查询，列=候选。"""
    best = best_rank_per_query(sim, ground_truth)
    out: Dict[str, float] = {}
    for k in ks:
        out[f"R@{k}"] = (best <= k).float().mean().item()
    # torch.median 对偶数长度返回下中位数；标准 MedR 用常规中位数（中间两值取平均）
    out["MedR"] = torch.quantile(best.float(), 0.5).item()
    return out


def image_to_text_metrics(sim_i2t: Tensor, gt_lists: List[List[int]], ks=(1, 5, 10)) -> Dict[str, float]:
    """图→文: 每张图的 5 条描述均为正确结果。"""
    return retrieval_metrics(sim_i2t, gt_lists, ks)


def text_to_image_metrics(sim_i2t: Tensor, gt_image_idx: List[int], ks=(1, 5, 10)) -> Dict[str, float]:
    """文→图: 每条描述只对应一张正确图片。传入的相似度矩阵与 i2t 共用（转置处理）。"""
    return retrieval_metrics(sim_i2t.t(), [[g] for g in gt_image_idx], ks)
