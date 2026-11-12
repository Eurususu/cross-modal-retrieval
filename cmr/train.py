"""训练入口。

用法:
    python -m cmr.train
    python -m cmr.train --set train.epochs=20 train.batch_size=32
    python -m cmr.train --set train.resume=checkpoints/last.pt   # 断点续训

流程: 构建 DataLoader → 双塔模型 → AdamW(分组学习率) + warmup 余弦调度 + AMP
→ 每 epoch 训练后在 val split 上快速评测（文→图 / 图→文 R@K），保存 best/last。
"""

from __future__ import annotations

import argparse
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import load_checkpoint, save_checkpoint
from .config import Config, apply_overrides, load_config
from .data import Flickr8kDataset, eval_caption_table, eval_transform, load_image, train_transform
from .embedding import encode_images, encode_texts
from .losses import batch_top1_accuracy, clip_loss
from .metrics import image_to_text_metrics, text_to_image_metrics
from .model import build_model
from .utils import count_trainable_params, describe_device, get_device, seed_everything, warmup_cosine


def validate(model, cfg: Config, device: torch.device) -> dict:
    """在 val split（英文标注）上计算双向 R@K。"""
    root = cfg.data.root_path
    names, caps, gt_image_idx = eval_caption_table(root, "val", cfg.data.captions_path)
    img_feats = encode_images(
        model,
        (load_image(cfg.data.images_path / n) for n in names),
        device,
        eval_transform(cfg.data.image_size),
        batch_size=cfg.eval.batch_size,
        desc="[val] 图像",
    )
    txt_feats = encode_texts(model, caps, device, max_len=cfg.data.max_text_len, desc="[val] 文本")
    sim = img_feats @ txt_feats.T

    gt_lists: list = [[] for _ in names]
    for cap_j, img_i in enumerate(gt_image_idx):
        gt_lists[img_i].append(cap_j)

    i2t = image_to_text_metrics(sim, gt_lists, cfg.eval.recall_ks)
    t2i = text_to_image_metrics(sim, gt_image_idx, cfg.eval.recall_ks)
    return {"i2t": i2t, "t2i": t2i}


def main(cfg: Config) -> None:
    seed_everything(cfg.seed)
    device = get_device()

    root = cfg.data.root_path
    if not (root / "Flickr8k.token.txt").exists():
        sys.exit(f"[train] 未找到数据集标注 {root}/Flickr8k.token.txt，请先运行: python -m cmr.download")

    zh_available = cfg.data.captions_zh_path.exists()
    if cfg.data.mix_chinese and not zh_available:
        print("[train] 提示: 未找到中文标注（可选）。生成后训练将自动混用中英文:")
        print("        python -m cmr.translate_captions")

    train_ds = Flickr8kDataset(
        root,
        split="train",
        images_dir=cfg.data.images_path,
        transform=train_transform(cfg.data.image_size),
        mix_chinese=cfg.data.mix_chinese,
        zh_captions_file=cfg.data.captions_zh_path,
    )
    loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=len(train_ds) > cfg.train.batch_size,
        persistent_workers=cfg.data.num_workers > 0,
    )

    model = build_model(cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.train.lr, cfg.train.head_lr),
        weight_decay=cfg.train.weight_decay,
    )
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup_steps = max(1, int(total_steps * cfg.train.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(warmup_steps, total_steps))
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    print(f"[train] 设备: {describe_device(device)} | AMP: {use_amp}")
    print(f"[train] 可训练参数: {count_trainable_params(model)/1e6:.1f}M")
    print(f"[train] 训练样本: {len(train_ds)} | batch={cfg.train.batch_size} × {steps_per_epoch} 步/epoch × {cfg.train.epochs} epochs")

    start_epoch = 0
    best_metric = -1.0
    if cfg.train.resume_path:
        ckpt_path = cfg.train.resume_path
        if not ckpt_path.exists():
            sys.exit(f"[train] resume 文件不存在: {ckpt_path}")
        _, _, ckpt = load_checkpoint(ckpt_path, device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt.get("best_metric", -1.0)
        print(f"[train] 从 {ckpt_path} 恢复: epoch {start_epoch}, best={best_metric:.4f}")

    ckpt_dir = cfg.train.checkpoint_dir_path
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        running = {"loss": 0.0, "acc": 0.0, "n": 0}
        pbar = tqdm(loader, desc=f"epoch {epoch+1}/{cfg.train.epochs}", unit="batch")
        for step, batch in enumerate(pbar):
            images = batch["image"].to(device, non_blocking=True)
            texts = batch["caption"]
            enc = model.text_encoder.tokenizer(
                list(texts), padding=True, truncation=True,
                max_length=cfg.data.max_text_len, return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device, non_blocking=True)
            attention_mask = enc["attention_mask"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images, input_ids, attention_mask)
                loss = clip_loss(logits)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running["loss"] += loss.item()
            running["acc"] += batch_top1_accuracy(logits.detach())
            running["n"] += 1
            pbar.set_postfix(
                loss=f"{running['loss']/running['n']:.4f}",
                acc=f"{running['acc']/running['n']:.3f}",
                T=f"{model.temperature:.3f}",
            )

        if cfg.train.val_each_epoch:
            metrics = validate(model, cfg, device)
            i2t, t2i = metrics["i2t"], metrics["t2i"]
            score = 0.5 * (i2t["R@1"] + t2i["R@1"])
            print(
                f"[val] epoch {epoch+1} | 图→文 R@1 {i2t['R@1']:.4f} R@5 {i2t['R@5']:.4f} | "
                f"文→图 R@1 {t2i['R@1']:.4f} R@5 {t2i['R@5']:.4f} | 平均R@1 {score:.4f}"
            )
            if score > best_metric:
                best_metric = score
                save_checkpoint(ckpt_dir / "best.pt", model, cfg, epoch, best_metric)
                print(f"[train] 🎯 新的最佳模型 → {ckpt_dir/'best.pt'}")

        save_checkpoint(ckpt_dir / "last.pt", model, cfg, epoch, best_metric, optimizer, scheduler)

    best_path = ckpt_dir / "best.pt"
    print("\n[train] ✅ 训练完成")
    if best_path.exists():
        print(f"[train] 最佳模型(val 平均R@1={best_metric:.4f}): {best_path}")
    print("[train] 下一步:")
    print("  评测:   python -m cmr.evaluate")
    print("  建索引: python -m cmr.index")
    print("  演示:   python -m cmr.app")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练跨模态双塔模型")
    parser.add_argument("--config", default=None, help="配置文件路径（默认 configs/default.yaml）")
    parser.add_argument("--set", nargs="*", default=[], help="覆盖配置项，如 --set train.epochs=20")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(apply_overrides(load_config(args.config), args.set))
