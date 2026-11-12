"""评测入口：在 test split 上计算 R@1/5/10 与 MedR（图→文 / 文→图）。

用法:
    python -m cmr.evaluate                          # 默认 checkpoints/best.pt，英文
    python -m cmr.evaluate --checkpoint xxx.pt --lang both
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .checkpoint import load_checkpoint
from .config import apply_overrides, load_config
from .data import eval_caption_table, eval_transform, load_image
from .embedding import encode_images, encode_texts
from .metrics import image_to_text_metrics, text_to_image_metrics
from .utils import describe_device, get_device

DEFAULT_CKPT = Path("checkpoints/best.pt")


def evaluate_language(model, cfg, device, split: str = "test", language: str = "en") -> dict:
    root = cfg.data.root_path
    captions_path = cfg.data.captions_zh_path if language == "zh" else cfg.data.captions_path
    if not captions_path.exists():
        print(f"[eval] 跳过 {language}: 标注文件不存在 ({captions_path})")
        return {}

    names, caps, gt_image_idx = eval_caption_table(root, split, captions_path)
    img_feats = encode_images(
        model,
        (load_image(cfg.data.images_path / n) for n in names),
        device,
        eval_transform(cfg.data.image_size),
        batch_size=cfg.eval.batch_size,
        desc=f"[test:{language}] 图像",
    )
    txt_feats = encode_texts(model, caps, device, max_len=cfg.data.max_text_len, desc=f"[test:{language}] 文本")
    sim = img_feats @ txt_feats.T  # (n_img, n_cap)

    gt_lists: list = [[] for _ in names]
    for cap_j, img_i in enumerate(gt_image_idx):
        gt_lists[img_i].append(cap_j)

    i2t = image_to_text_metrics(sim, gt_lists, tuple(cfg.eval.recall_ks))
    t2i = text_to_image_metrics(sim, gt_image_idx, tuple(cfg.eval.recall_ks))
    return {
        "n_images": len(names),
        "n_captions": len(caps),
        "image_to_text": i2t,
        "text_to_image": t2i,
    }


def format_row(label: str, m: dict) -> str:
    return (
        f"  {label:<10s} R@1 {m['R@1']:.4f} | R@5 {m['R@5']:.4f} | "
        f"R@10 {m['R@10']:.4f} | MedR {m['MedR']:.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="评测训练好的双塔模型")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--lang", choices=["en", "zh", "both"], default="en",
                        help="en=英文标注; zh=机翻中文标注; both=两者都测")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    device = get_device()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = Path.cwd() / ckpt_path
    if not ckpt_path.exists():
        sys.exit(f"[eval] checkpoint 不存在: {ckpt_path}（请先训练: python -m cmr.train）")

    print(f"[eval] checkpoint: {ckpt_path}")
    print(f"[eval] 设备: {describe_device(device)}")
    model, cfg, _ = load_checkpoint(ckpt_path, device)

    languages = ["en", "zh"] if args.lang == "both" else [args.lang]
    results = {}
    for lang in languages:
        r = evaluate_language(model, cfg, device, split="test", language=lang)
        if not r:
            continue
        results[lang] = r
        print(f"\n[eval] ===== 测试集 · {('英文标注' if lang == 'en' else '机翻中文标注')} "
              f"({r['n_images']} 图 / {r['n_captions']} 描述) =====")
        print(format_row("图→文", r["image_to_text"]))
        print(format_row("文→图", r["text_to_image"]))

    if not results:
        sys.exit("[eval] 没有可用的评测结果")

    report_dir = cfg.eval.report_dir_path
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"eval_{ckpt_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(
        json.dumps({"checkpoint": str(ckpt_path), "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[eval] 📄 报告已保存: {report_path}")


if __name__ == "__main__":
    main()
