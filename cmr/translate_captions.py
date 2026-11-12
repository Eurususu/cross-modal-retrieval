"""把英文标注机器翻译成中文（可选的双语增强步骤）。

用法: python -m cmr.translate_captions

使用 Helsinki-NLP/opus-mt-en-zh 翻译全部 40k 条描述，输出与原文件
同格式的 Flickr8k.token.zh.txt，支持断点续跑（进度实时落盘）。
需要额外安装: pip install sentencepiece
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from .config import apply_overrides, load_config
from .data import _CAPTION_RE
from .utils import describe_device, get_device

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-zh"


def read_records(path: Path) -> list:
    records = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _CAPTION_RE.match(raw.strip("\r"))
        if m:
            records.append((m.group("image"), int(m.group("idx")), m.group("caption").strip()))
    return records


def load_progress(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_progress(path: Path, progress: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成中文标注（机器翻译）")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--batch-size", type=int, default=None, help="翻译批大小（默认按设备自动选择）")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    src = cfg.data.captions_path
    dst = cfg.data.captions_zh_path
    if not src.exists():
        sys.exit(f"[translate] 找不到英文标注: {src}（请先 python -m cmr.download）")
    if dst.exists():
        print(f"[translate] 中文标注已存在: {dst}（如需重新生成请先删除该文件）")
        return

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        sys.exit(f"[translate] 加载 transformers 失败: {exc}")

    device = get_device()
    print(f"[translate] 设备: {describe_device(device)} | 模型: {TRANSLATION_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL).to(device).eval()

    records = read_records(src)
    unique_texts = sorted({t for _, _, t in records})
    progress_path = dst.with_suffix(".progress.json")
    progress = load_progress(progress_path)
    todo = [t for t in unique_texts if t not in progress]
    print(f"[translate] 共 {len(unique_texts)} 条唯一描述，已完成 {len(progress)}，待翻译 {len(todo)}")

    batch_size = args.batch_size or (32 if device.type == "cuda" else 8)
    n_batches = (len(todo) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(todo), batch_size), total=n_batches, desc="翻译", unit="batch"):
            chunk = todo[i : i + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=128)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model.generate(
                **enc,
                max_new_tokens=96,
                num_beams=1,
                do_sample=False,
            )
            for text, ids in zip(chunk, out):
                progress[text] = tokenizer.decode(ids, skip_special_tokens=True).strip()
            if (i // batch_size) % 20 == 0:
                save_progress(progress_path, progress)
    save_progress(progress_path, progress)

    missing = [t for t in unique_texts if t not in progress]
    if missing:
        sys.exit(f"[translate] 异常: 仍有 {len(missing)} 条未翻译，请重跑本命令续传")

    lines = [f"{img}#{idx}\t{progress[text]}" for img, idx, text in records]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    progress_path.unlink(missing_ok=True)
    print(f"[translate] ✅ 已生成中文标注: {dst}（{len(lines)} 条）")
    print("[translate] 重新运行 python -m cmr.train 即可启用中英混合训练")


if __name__ == "__main__":
    main()
