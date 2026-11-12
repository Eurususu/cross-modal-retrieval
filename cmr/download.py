"""Flickr8k 数据集下载与目录归一化。

用法: python -m cmr.download [--root data/flickr8k]

下载源为 GitHub 上广泛使用的 jbrownlee/Datasets 镜像（官方原版需向
伊利诺伊大学申请，流程很慢）。下载失败时脚本会给出手动放置指引。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"  # ~1.1GB
TEXT_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"  # ~9MB

REQUIRED_TEXT_FILES = [
    "Flickr8k.token.txt",
    "Flickr_8k.trainImages.txt",
    "Flickr_8k.devImages.txt",
    "Flickr_8k.testImages.txt",
]

MANUAL_HINT = """
下载失败。请手动处理：
  1) 浏览器下载以下两个文件，放入 {archive_dir}/ :
     - {dataset_url}
     - {text_url}
     然后重新运行: python -m cmr.download
  2) 备选镜像: https://huggingface.co/datasets/nlphuji/flickr8k-dataset （目录结构不同，需自行整理）
"""


def download_file(url: str, dst: Path, chunk_mb: int = 1) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[download] 已存在，跳过: {dst}")
        return
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dst, "wb") as f:
            bar = tqdm(total=total, unit="B", unit_scale=True, desc=dst.name)
            for chunk in r.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))
            bar.close()


def extract_zip(zip_path: Path, dest: Path) -> None:
    print(f"[download] 解压 {zip_path.name} → {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:  # zipfile 提取时会过滤绝对路径与 ".."，安全
        zf.extractall(dest)


def normalize_layout(root: Path) -> None:
    """把解压出的各种目录结构归一化为: root/images/*.jpg + root/*.txt。"""
    root.mkdir(parents=True, exist_ok=True)
    archive_dir = root / "_archives"
    extracted = archive_dir / "extracted"
    if not extracted.exists():
        return

    # 1) 图片目录：找到包含 jpg 最多的顶层目录，移动为 root/images
    images_dir = root / "images"
    if not images_dir.exists() or not any(images_dir.glob("*.jpg")):
        candidates = {}
        for jpg in extracted.rglob("*.[jJ][pP][gG]"):
            candidates[jpg.parent] = candidates.get(jpg.parent, 0) + 1
        if candidates:
            best = max(candidates, key=candidates.get)
            # 空目录先删掉，否则 shutil.move 会把目录移动到其内部
            if images_dir.exists():
                shutil.rmtree(images_dir)
            shutil.move(str(best), str(images_dir))

    # 2) 文本文件：移动到 root 顶层
    for name in REQUIRED_TEXT_FILES:
        if (root / name).exists():
            continue
        found = list(extracted.rglob(name))
        if found:
            shutil.move(str(found[0]), str(root / name))

    # 3) 清理解压残留（保留 zip 压缩包，方便出错重试）
    shutil.rmtree(extracted, ignore_errors=True)


def check_complete(root: Path) -> bool:
    has_text = all((root / name).exists() for name in REQUIRED_TEXT_FILES)
    n_images = len(list((root / "images").glob("*.jpg"))) if (root / "images").exists() else 0
    return has_text and n_images >= 8000


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并整理 Flickr8k 数据集")
    parser.add_argument("--root", default=None, help="数据集根目录（默认读配置 data.root）")
    args = parser.parse_args()

    if args.root:
        root = Path(args.root).expanduser().resolve()
    else:
        from .config import load_config

        root = load_config().data.root_path

    if check_complete(root):
        n_images = len(list((root / "images").glob("*.jpg")))
        print(f"[download] 数据集已就绪: {root}（{n_images} 张图片），无需下载。")
        return

    archive_dir = root / "_archives"
    try:
        download_file(TEXT_URL, archive_dir / "Flickr8k_text.zip")
        download_file(DATASET_URL, archive_dir / "Flickr8k_Dataset.zip")
        extract_zip(archive_dir / "Flickr8k_text.zip", archive_dir / "extracted")
        extract_zip(archive_dir / "Flickr8k_Dataset.zip", archive_dir / "extracted")
        normalize_layout(root)
    except Exception as exc:  # noqa: BLE001
        print(f"[download] 出错: {exc}", file=sys.stderr)
        print(MANUAL_HINT.format(archive_dir=archive_dir, dataset_url=DATASET_URL, text_url=TEXT_URL))
        sys.exit(1)

    if check_complete(root):
        n_images = len(list((root / "images").glob("*.jpg")))
        print(f"[download] ✅ 数据集就绪: {n_images} 张图片 → {root/'images'}")
        print("[download] 下一步: python -m cmr.train")
    else:
        print("[download] ⚠️ 目录结构异常，请检查上述路径，或参照手动指引重新放置文件。")
        sys.exit(1)


if __name__ == "__main__":
    main()
