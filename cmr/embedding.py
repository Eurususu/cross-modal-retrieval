"""批量编码：把 PIL 图像 / 文本列表编码为归一化嵌入矩阵（CPU float32 张量）。"""

from __future__ import annotations

from typing import Iterable, List

import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm


@torch.no_grad()
def encode_images(
    model,
    images: Iterable[Image.Image],
    device: torch.device,
    transform,
    batch_size: int = 64,
    desc: str = "编码图像",
    show_progress: bool = True,
) -> Tensor:
    """images 为 PIL 图片可迭代对象；transform 必须提供（PIL → tensor）。"""
    model.eval()
    feats: List[Tensor] = []
    buf: List[Tensor] = []

    def flush():
        if not buf:
            return
        batch = torch.stack(buf).to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            f = model.encode_image(batch)
        feats.append(f.float().cpu())
        buf.clear()

    for img in tqdm(images, desc=desc, unit="img", disable=not show_progress):
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf.append(transform(img))
        if len(buf) >= batch_size:
            flush()
    flush()
    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_texts(
    model,
    texts: List[str],
    device: torch.device,
    batch_size: int = 256,
    max_len: int = 64,
    desc: str = "编码文本",
) -> Tensor:
    model.eval()
    feats: List[Tensor] = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch",
                  total=n_batches, disable=(n_batches <= 1)):
        chunk = [t if t.strip() else " " for t in texts[i : i + batch_size]]
        enc = model.text_encoder.tokenizer(
            chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            f = model.encode_text(input_ids, attention_mask)
        feats.append(f.float().cpu())
    return torch.cat(feats, dim=0)
