"""把训练好的双塔导出为 ONNX（TensorRT 部署的第一步）。

用法: python -m cmr.export_onnx [--checkpoint checkpoints/best.pt] [--out onnx]

导出两个独立模型（部署时两塔各自独立服务）:
  image_tower.onnx  输入 (B,3,224,224) float32       输出 (B,512)
  text_tower.onnx   输入 (B,64) int32 × 2(ids/mask)  输出 (B,512)

注意:
- 输出是投影头结果（未 L2 归一化），归一化留在部署侧做（numpy 一行）
- 文本输入固定长度 64（tokenizer pad 到 max_length），batch 动态
- 文本输入用 int32（词表 11.9 万 << 2^31），图内 Cast 成 int64，
  规避部分 TensorRT 版本对 int64 输入张量的支持问题
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .checkpoint import load_checkpoint
from .config import PROJECT_ROOT
from .model import mean_pool

MAX_TEXT_LEN = 64


class TextTowerForExport(nn.Module):
    """int32 输入 + 图内 Cast 的文本塔包装。"""

    def __init__(self, text_encoder: nn.Module):
        super().__init__()
        self.enc = text_encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.enc.backbone(
            input_ids=input_ids.long(), attention_mask=attention_mask.long()
        )
        pooled = mean_pool(out.last_hidden_state, attention_mask.long())
        return self.enc.projection(pooled)


def export(model: nn.Module, dummy_args, out_path: Path, input_names, output_names, dynamic_batch=True):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic = {name: {0: "batch"} for name in input_names} if dynamic_batch else None
    torch.onnx.export(
        model,
        dummy_args,
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic,
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"[export] ✅ {out_path.name}  {size_mb:.1f}MB")


def ort_parity(onnx_path: Path, torch_out: np.ndarray, feeds: dict, atol=2e-2) -> None:
    """用 onnxruntime 跑一遍导出的模型, 验证与 PyTorch 数值一致（导出正确性的黄金标准）。"""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (name,) = [i.name for i in sess.get_outputs()]
    ort_out = sess.run(None, feeds)[0]
    cos = np.sum(torch_out * ort_out, axis=1) / (
        np.linalg.norm(torch_out, axis=1) * np.linalg.norm(ort_out, axis=1) + 1e-9
    )
    diff = np.abs(torch_out - ort_out).max()
    assert cos.min() > 0.9999, f"{onnx_path.name} 余弦相似度过低: {cos.min()}"
    print(f"[export] ORT 对齐校验通过: max|Δ|={diff:.2e}  cos_min={cos.min():.8f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="导出双塔为 ONNX")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "best.pt"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "onnx"))
    args = parser.parse_args()

    device = torch.device("cpu")
    model, _, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    out_dir = Path(args.out)

    # ---- 图像塔 ----
    dummy_img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch_img = model.image_encoder(dummy_img).numpy()
    img_path = out_dir / "image_tower.onnx"
    export(model.image_encoder, (dummy_img,), img_path,
           input_names=["images"], output_names=["image_feat"])
    ort_parity(img_path, torch_img, {"images": dummy_img.numpy()})

    # ---- 文本塔 ----
    tok = model.text_encoder.tokenizer
    sample = tok(["a dog running on the beach"],
                 padding="max_length", truncation=True, max_length=MAX_TEXT_LEN,
                 return_tensors="pt")
    ids32 = sample["input_ids"].to(torch.int32)
    mask32 = sample["attention_mask"].to(torch.int32)
    wrapper = TextTowerForExport(model.text_encoder).eval()
    with torch.no_grad():
        torch_txt = wrapper(ids32, mask32).numpy()
    txt_path = out_dir / "text_tower.onnx"
    export(wrapper, (ids32, mask32), txt_path,
           input_names=["input_ids", "attention_mask"], output_names=["text_feat"])
    ort_parity(txt_path, torch_txt,
               {"input_ids": ids32.numpy(), "attention_mask": mask32.numpy()})

    print("[export] 全部完成 →", out_dir)


if __name__ == "__main__":
    main()
