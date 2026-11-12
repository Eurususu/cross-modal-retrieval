"""图片库向量索引：把一个文件夹里的所有图片编码入库，支持文本/图片查询。

用法:
    建索引: python -m cmr.index [--checkpoint checkpoints/best.pt]
                          [--images data/flickr8k/images] [--out index/flickr8k.npz]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from .checkpoint import load_checkpoint
from .config import PROJECT_ROOT, apply_overrides, load_config
from .data import eval_transform, load_image
from .embedding import encode_images, encode_texts
from .model import DualEncoder
from .utils import describe_device, get_device

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(folder: Path) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        sys.exit(f"[index] 图片目录不存在: {folder}")
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not paths:
        sys.exit(f"[index] 目录中没有图片: {folder}")
    return paths


class TorchQueryEncoder:
    """PyTorch 查询编码器: 复用完整 DualEncoder（含 tokenizer）。"""

    def __init__(self, model: DualEncoder, device, transform, max_text_len: int = 64):
        self.model, self.device = model, device
        self.transform, self.max_text_len = transform, max_text_len

    def text(self, query: str) -> np.ndarray:
        feat = encode_texts(self.model, [query], self.device, max_len=self.max_text_len, desc=None)
        return feat[0].numpy()

    def image(self, image: Image.Image) -> np.ndarray:
        feat = encode_images(self.model, [image], self.device, self.transform,
                             desc=None, show_progress=False)
        return feat[0].numpy()


class EmbeddingIndex:
    """已编码的图片库 + 查询编码器（PyTorch 或 TRT 均可），提供检索接口。"""

    def __init__(self, feats: np.ndarray, names: List[str], image_dir: Path, query_encoder):
        self.feats = feats.astype(np.float32)
        self.names = names
        self.image_dir = Path(image_dir)
        self.query = query_encoder  # 需提供 .text(str) / .image(PIL) -> 归一化向量

    def __len__(self) -> int:
        return len(self.names)

    def _search(self, query_feat: np.ndarray, k: int) -> List[Tuple[Path, float]]:
        k = min(k, len(self))
        sims = self.feats @ query_feat.astype(np.float32)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self.image_dir / self.names[i], float(sims[i])) for i in top]

    def search_by_text(self, query: str, k: int = 12) -> List[Tuple[Path, float]]:
        return self._search(self.query.text(query), k)

    def search_by_image(self, image: Image.Image, k: int = 12) -> List[Tuple[Path, float]]:
        return self._search(self.query.image(image), k)


def build_index(
    ckpt_path: Path,
    images_dir: Path,
    out_path: Path,
    batch_size: int = 64,
    device=None,
) -> EmbeddingIndex:
    device = device or get_device()
    print(f"[index] checkpoint: {ckpt_path}")
    print(f"[index] 设备: {describe_device(device)}")
    model, cfg, _ = load_checkpoint(ckpt_path, device)

    paths = list_images(images_dir)
    print(f"[index] 待编码图片: {len(paths)} 张（来自 {images_dir}）")
    feats = encode_images(
        model,
        (load_image(p) for p in paths),
        device,
        eval_transform(cfg.data.image_size),
        batch_size=batch_size,
        desc="[index] 编码",
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        feats=feats.numpy().astype(np.float32),
        names=np.array([p.name for p in paths]),
        image_dir=str(images_dir),
        image_size=cfg.data.image_size,
        max_text_len=cfg.data.max_text_len,
    )
    print(f"[index] ✅ 索引已保存: {out_path}（{len(paths)} × {feats.shape[1]} 维）")
    encoder = TorchQueryEncoder(model, device, eval_transform(cfg.data.image_size),
                                cfg.data.max_text_len)
    return EmbeddingIndex(feats.numpy(), [p.name for p in paths], images_dir, encoder)


def load_index(npz_path: Path, ckpt_path: Optional[Path] = None,
               query_encoder=None) -> EmbeddingIndex:
    """加载索引。query_encoder 可传入 TRT 等自定义编码器；否则用 PyTorch（需 checkpoint）。"""
    device = get_device()
    data = np.load(npz_path, allow_pickle=False)
    if query_encoder is None:
        if ckpt_path is None:
            ckpt_path = Path("checkpoints/best.pt")
        model, _, _ = load_checkpoint(ckpt_path, device)
        query_encoder = TorchQueryEncoder(
            model, device, eval_transform(int(data["image_size"])), int(data["max_text_len"]))
    return EmbeddingIndex(
        data["feats"],
        [str(n) for n in data["names"]],
        Path(str(data["image_dir"])),
        query_encoder,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="构建图片库向量索引")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--images", default=None, help="图片目录（默认 data/flickr8k/images）")
    parser.add_argument("--out", default=None, help="输出 npz 路径（默认 index/flickr8k.npz）")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    images_dir = Path(args.images) if args.images else cfg.data.images_path
    out_path = Path(args.out) if args.out else PROJECT_ROOT / "index" / "flickr8k.npz"
    build_index(Path(args.checkpoint), images_dir, out_path, batch_size=cfg.eval.batch_size)


if __name__ == "__main__":
    main()
