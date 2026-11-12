"""Flickr8k 数据：标注解析、数据集、图像预处理。"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Flickr8k 个别图片文件有轻微截断，放宽限制避免读取失败
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 标注行格式: `100024841.jpg#0\tA child ...`
_CAPTION_RE = re.compile(
    r"^(?P<image>.+?\.(?:jpe?g|png))#(?P<idx>\d+)\s*\t(?P<caption>.*)$", re.IGNORECASE
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SPLIT_FILES = {
    "train": "Flickr_8k.trainImages.txt",
    "val": "Flickr_8k.devImages.txt",
    "test": "Flickr_8k.testImages.txt",
}


def parse_captions(path: Path) -> Dict[str, List[str]]:
    """解析 token 标注文件 → {图片名: [5 条描述]}（保持文件内顺序）。"""
    captions: Dict[str, List[str]] = {}
    bad = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _CAPTION_RE.match(raw.strip("\r"))
        if not m:
            bad += 1
            continue
        captions.setdefault(m.group("image"), []).append(m.group("caption").strip())
    if bad:
        print(f"[data] 警告: {path.name} 中 {bad} 行无法解析，已跳过")
    return captions


def load_split_names(root: Path, split: str) -> List[str]:
    """读取官方 split 文件（train / val(=dev) / test）中的图片名列表。"""
    try:
        fname = SPLIT_FILES[split]
    except KeyError:
        raise ValueError(f"未知 split: {split!r}，可选 {list(SPLIT_FILES)}")
    path = root / fname
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_transform(image_size: int = 224):
    """训练增强：轻量裁剪/翻转（过于激进会破坏描述语义）。"""
    return transforms.Compose([
        transforms.Resize(image_size + 32, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(image_size + 32, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


class Flickr8kDataset(Dataset):
    """训练用数据集：每个样本 = (一张图, 随机选中的一条描述)。

    - 训练时每张图的 5 条描述被等概率随机抽出（等价于按 caption 扩充采样）
    - 若提供中文标注文件且 mix_chinese=True，则每条样本 50% 概率使用中文描述
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        images_dir: Optional[Path] = None,
        transform=None,
        image_size: int = 224,
        mix_chinese: bool = False,
        zh_captions_file: Optional[Path] = None,
    ):
        self.root = Path(root)
        self.images_dir = Path(images_dir) if images_dir else self.root / "images"
        all_captions = parse_captions(self.root / "Flickr8k.token.txt")

        names = load_split_names(self.root, split)
        self.image_names = [n for n in names if n in all_captions and (self.images_dir / n).exists()]
        skipped = len(names) - len(self.image_names)
        if skipped:
            print(f"[data] 警告: split={split} 有 {skipped} 张图缺失或无标注，已跳过")
        self.captions = [all_captions[n] for n in self.image_names]

        self.zh_captions: Optional[List[Optional[List[str]]]] = None
        if mix_chinese and split == "train" and zh_captions_file and Path(zh_captions_file).exists():
            zh_all = parse_captions(Path(zh_captions_file))
            self.zh_captions = [zh_all.get(n) for n in self.image_names]
            n_have = sum(1 for c in self.zh_captions if c)
            print(f"[data] 中文增强已启用: {n_have}/{len(self.image_names)} 张图有中文标注，训练时按 50% 概率混用")

        self.transform = transform or (
            train_transform(image_size) if split == "train" else eval_transform(image_size)
        )

    def __len__(self) -> int:
        return len(self.image_names)

    def image_path(self, i: int) -> Path:
        return self.images_dir / self.image_names[i]

    def load_image(self, i: int) -> Image.Image:
        return load_image(self.image_path(i))

    def __getitem__(self, i: int) -> dict:
        caption = random.choice(self.captions[i])
        if (
            self.zh_captions is not None
            and self.zh_captions[i]
            and random.random() < 0.5
        ):
            caption = random.choice(self.zh_captions[i])
        return {
            "image": self.transform(self.load_image(i)),
            "caption": caption,
            "image_id": i,
        }


def eval_caption_table(root: Path, split: str, captions_path: Path) -> Tuple[List[str], List[str], List[int]]:
    """构造评测用的 (图片名列表, 描述列表, 每条描述对应的图片下标)。

    描述按图片顺序展开，因此图片 i 的正确描述是一段连续下标。
    """
    captions = parse_captions(Path(captions_path))
    names = [n for n in load_split_names(Path(root), split) if n in captions]
    caps: List[str] = []
    gt_image_idx: List[int] = []
    for idx, name in enumerate(names):
        for cap in captions[name]:
            caps.append(cap)
            gt_image_idx.append(idx)
    return names, caps, gt_image_idx
