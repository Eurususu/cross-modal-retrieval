"""TensorRT 部署: 构建 engine、精度校验、性能对比。

用法:
    python -m cmr.trt build    # onnx/*.onnx --fp16--> trt/*.engine (调用 trtexec)
    python -m cmr.trt verify   # TRT vs PyTorch 数值对齐 + 端到端检索一致性
    python -m cmr.trt bench    # 单查询/批量推理延迟对比 (PyTorch vs TRT)

架构说明: TensorRT 只接管两个塔的前向计算。tokenizer、图像预处理、向量检索
（feats @ q）不是神经网络，仍留在 Python/numpy 侧。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .config import PROJECT_ROOT
from .data import eval_transform

TRTEXEC = "/opt/TensorRT/bin/trtexec"
ONNX_DIR = PROJECT_ROOT / "onnx"
TRT_DIR = PROJECT_ROOT / "trt"
MAX_TEXT_LEN = 64


class TRTEngine:
    """单个 TensorRT engine 的最小推理封装（TRT 10 API, torch 张量做缓冲区）。"""

    def __init__(self, engine_path: Path):
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"engine 反序列化失败: {engine_path}")
        self.ctx = self.engine.create_execution_context()
        self.trt = trt
        self.device = torch.device("cuda")
        self.stream = torch.cuda.Stream(device=self.device)
        self._dtype_map = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
        }
        self.input_names, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            (self.input_names if mode == trt.TensorIOMode.INPUT else self.output_names).append(name)

    def __call__(self, *inputs: np.ndarray) -> np.ndarray:
        for name, arr in zip(self.input_names, inputs):
            t = torch.from_numpy(np.ascontiguousarray(arr)).to(self.device)
            self.ctx.set_input_shape(name, tuple(t.shape))
            self.ctx.set_tensor_address(name, int(t.data_ptr()))
        outputs = []
        for name in self.output_names:
            shape = tuple(self.ctx.get_tensor_shape(name))
            t = torch.empty(shape, dtype=self._dtype_map[self.engine.get_tensor_dtype(name)],
                            device=self.device)
            self.ctx.set_tensor_address(name, int(t.data_ptr()))
            outputs.append(t)
        ok = self.ctx.execute_async_v3(self.stream.cuda_stream)  # TRT 10.14 的方法名
        if not ok:
            raise RuntimeError("execute_async_v3 执行失败")
        self.stream.synchronize()
        return outputs[0].float().cpu().numpy()


class TRTBackend:
    """两塔 TRT 引擎集合, 接口与 DualEncoder.encode_* 对齐（输出已 L2 归一化）。"""

    def __init__(self, trt_dir: Path = TRT_DIR):
        self.image_engine = TRTEngine(trt_dir / "image_tower_fp16.engine")
        self.text_engine = TRTEngine(trt_dir / "text_tower_fp16.engine")

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

    def encode_image(self, images_np: np.ndarray) -> np.ndarray:
        return self._norm(self.image_engine(images_np.astype(np.float32)))

    def encode_text(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        return self._norm(self.text_engine(
            input_ids.astype(np.int32), attention_mask.astype(np.int32)))


class TRTQueryEncoder:
    """REST 服务用的 TRT 查询编码器: 与 TorchQueryEncoder 接口对齐。

    只从 checkpoint 读取配置（文本模型名）来加载 tokenizer, 不加载 PyTorch 权重。
    注意文本固定 pad 到 engine 的输入长度(64), 图像用与训练一致的 eval_transform。
    """

    def __init__(self, checkpoint: Path = PROJECT_ROOT / "checkpoints" / "best.pt",
                 trt_dir: Path = TRT_DIR, image_size: int = 224):
        from transformers import AutoTokenizer

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        text_model = ckpt["config"]["model"]["text_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.backend = TRTBackend(trt_dir)
        self.transform = eval_transform(image_size)
        self.max_text_len = MAX_TEXT_LEN
        print(f"[trt] 查询编码器就绪 (tokenizer={text_model}, TRT FP16 engines)")

    def text(self, query: str) -> np.ndarray:
        enc = self.tokenizer([query], padding="max_length", truncation=True,
                             max_length=self.max_text_len, return_tensors="np")
        return self.backend.encode_text(enc["input_ids"], enc["attention_mask"])[0]

    def image(self, image) -> np.ndarray:
        from PIL import Image as PILImage

        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = self.transform(image).numpy()[None]  # (1,3,224,224)
        return self.backend.encode_image(arr)[0]


def _build_engine(onnx: Path, engine: Path, shapes: str) -> None:
    cmd = [
        TRTEXEC,
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        "--fp16",
        "--memPoolSize=workspace:2048",
        f"--minShapes={shapes.format(b=1)}",
        f"--optShapes={shapes.format(b=16)}",
        f"--maxShapes={shapes.format(b=64)}",
    ]
    print("[trt] 构建:", engine.name)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(result.stdout[-2000:], result.stderr[-2000:])
        sys.exit(f"[trt] trtexec 失败: {onnx.name}")
    print(f"[trt] ✅ {engine.name}  {engine.stat().st_size/1e6:.1f}MB")


def cmd_build() -> None:
    TRT_DIR.mkdir(exist_ok=True)
    _build_engine(ONNX_DIR / "image_tower.onnx", TRT_DIR / "image_tower_fp16.engine",
                  "images:{b}x3x224x224")
    _build_engine(ONNX_DIR / "text_tower.onnx", TRT_DIR / "text_tower_fp16.engine",
                  "input_ids:{b}x64,attention_mask:{b}x64")


def cmd_verify() -> None:
    from .checkpoint import load_checkpoint
    from .data import load_image, eval_transform
    from .index import load_index

    device = torch.device("cuda")
    model, _, _ = load_checkpoint(PROJECT_ROOT / "checkpoints" / "best.pt", device)
    backend = TRTBackend()

    # ---- 数值对齐: 图像塔 ----
    img_tf = eval_transform(224)
    imgs = np.stack([
        img_tf(load_image(PROJECT_ROOT / "data/flickr8k/images" / n)).numpy()
        for n in ["3151365121_e2a685a666.jpg", "1541272333_1624b22546.jpg"]
    ])
    with torch.no_grad():
        ref = model.encode_image(torch.from_numpy(imgs).to(device)).cpu().numpy()
    out = backend.encode_image(imgs)
    cos = (ref * out).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(out, axis=1))
    print(f"[verify] 图像塔  FP16 vs FP32: cos_min={cos.min():.6f}  max|Δ|={np.abs(ref-out).max():.4f}")

    # ---- 数值对齐: 文本塔 ----
    texts = ["a dog running on the beach", "一只狗在沙滩上奔跑"]
    enc = model.text_encoder.tokenizer(texts, padding="max_length", truncation=True,
                                       max_length=MAX_TEXT_LEN, return_tensors="np")
    with torch.no_grad():
        ref_t = model.encode_text(
            torch.from_numpy(enc["input_ids"]).to(device),
            torch.from_numpy(enc["attention_mask"]).to(device),
        ).cpu().numpy()
    out_t = backend.encode_text(enc["input_ids"], enc["attention_mask"])
    cos_t = (ref_t * out_t).sum(1) / (np.linalg.norm(ref_t, axis=1) * np.linalg.norm(out_t, axis=1))
    print(f"[verify] 文本塔  FP16 vs FP32: cos_min={cos_t.min():.6f}  max|Δ|={np.abs(ref_t-out_t).max():.4f}")

    # ---- 端到端: TRT 文本向量 vs PyTorch 建的索引, Top1 是否一致 ----
    idx = load_index(PROJECT_ROOT / "index/flickr8k.npz",
                     PROJECT_ROOT / "checkpoints/best.pt")
    q = backend.encode_text(enc["input_ids"][:1], enc["attention_mask"][:1])[0]
    trt_top1 = idx.names[int(np.argmax(idx.feats @ q))]
    ref_top1 = idx.names[int(np.argmax(idx.feats @ ref_t[0]))]
    print(f"[verify] 端到端检索 Top1 一致: {trt_top1 == ref_top1} ({trt_top1})")


def _time(fn, warmup=10, iters=100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def cmd_bench() -> None:
    from .checkpoint import load_checkpoint

    device = torch.device("cuda")
    model, _, _ = load_checkpoint(PROJECT_ROOT / "checkpoints" / "best.pt", device)
    backend = TRTBackend()

    img1 = np.random.randn(1, 3, 224, 224).astype(np.float32)
    img32 = np.random.randn(32, 3, 224, 224).astype(np.float32)
    tok = model.text_encoder.tokenizer
    t1 = tok(["a dog"], padding="max_length", max_length=MAX_TEXT_LEN, return_tensors="np")
    t32 = tok(["a dog"] * 32, padding="max_length", max_length=MAX_TEXT_LEN, return_tensors="np")

    def torch_img(batch):
        with torch.no_grad():
            model.encode_image(torch.from_numpy(batch).to(device)).cpu()

    def torch_txt(ids, mask):
        with torch.no_grad():
            model.encode_text(torch.from_numpy(ids).to(device),
                              torch.from_numpy(mask).to(device)).cpu()

    rows = [
        ("图像塔 B=1 ", lambda: torch_img(img1), lambda: backend.encode_image(img1)),
        ("图像塔 B=32", lambda: torch_img(img32), lambda: backend.encode_image(img32)),
        ("文本塔 B=1 ", lambda: torch_txt(t1["input_ids"], t1["attention_mask"]),
         lambda: backend.encode_text(t1["input_ids"], t1["attention_mask"])),
        ("文本塔 B=32", lambda: torch_txt(t32["input_ids"], t32["attention_mask"]),
         lambda: backend.encode_text(t32["input_ids"], t32["attention_mask"])),
    ]
    print(f"{'':12s}{'PyTorch FP32':>14s}{'TRT FP16':>12s}{'加速比':>8s}")
    for name, torch_fn, trt_fn in rows:
        t_torch = _time(torch_fn)
        t_trt = _time(trt_fn)
        print(f"{name:12s}{t_torch:11.2f} ms{t_trt:9.2f} ms{t_torch/t_trt:7.1f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorRT 构建与验证")
    parser.add_argument("cmd", choices=["build", "verify", "bench"])
    args = parser.parse_args()
    {"build": cmd_build, "verify": cmd_verify, "bench": cmd_bench}[args.cmd]()


if __name__ == "__main__":
    main()
