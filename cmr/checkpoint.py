"""checkpoint 保存 / 加载（训练保存完整状态，评测与演示只需模型权重）。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from .config import Config, config_from_dict
from .model import DualEncoder, build_model


def save_checkpoint(
    path: str | Path,
    model: DualEncoder,
    cfg: Config,
    epoch: int,
    best_metric: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": asdict(cfg),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # 原子替换，避免写一半损坏


def load_checkpoint(path: str | Path, device: torch.device) -> tuple:
    """加载训练好的模型: 返回 (model(eval 模式), cfg, 原始 checkpoint dict)。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ckpt["config"])
    model = build_model(cfg.model)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg, ckpt
