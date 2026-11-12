"""REST 推理服务（FastAPI）——把训练好的双塔模型部署成可调用的 API。

用法:
    python -m cmr.serve                          # 默认 best.pt + index/flickr8k.npz, 端口 8000
    python -m cmr.serve --port 9000 --checkpoint checkpoints/best.pt

接口:
    GET  /health                          健康检查 + 图库规模
    POST /api/search/text                 {"query": "一只狗在沙滩上", "k": 10}
    POST /api/search/image                multipart 文件上传，字段名 file
    GET  /images/{文件名}                  图片静态文件（结果里的 URL 直接可访问）

模型与索引在服务启动时加载一次，请求只做一次前向 + 一次矩阵乘。
"""

from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import PROJECT_ROOT
from .index import EmbeddingIndex, load_index

DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "best.pt"
DEFAULT_INDEX = PROJECT_ROOT / "index" / "flickr8k.npz"

state: dict = {"index": None}  # lifespan 里填充


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时一次性加载模型+索引（避免每个请求重复加载）；关闭时释放。"""
    # 直接以 TestClient 驱动 app（不经 main）时使用默认路径
    from types import SimpleNamespace

    args = getattr(
        app.state,
        "args",
        SimpleNamespace(checkpoint=str(DEFAULT_CKPT), index=str(DEFAULT_INDEX), trt=False),
    )
    for name, path in [("checkpoint", args.checkpoint), ("index", args.index)]:
        if not Path(path).exists():
            sys.exit(f"[serve] {name} 不存在: {path}（请先训练/建索引）")
    print(f"[serve] 加载索引: {args.index}")
    if getattr(args, "trt", False):
        from .trt import TRTQueryEncoder

        encoder = TRTQueryEncoder(checkpoint=args.checkpoint)
        state["index"] = load_index(args.index, query_encoder=encoder)
        state["backend"] = "TensorRT FP16"
    else:
        print(f"[serve] 加载 checkpoint: {args.checkpoint}")
        state["index"] = load_index(args.index, args.checkpoint)
        state["backend"] = "PyTorch FP32"
    print(f"[serve] ✅ 就绪 | 后端 {state['backend']} | 图库 {len(state['index'])} 张")
    yield
    state["index"] = None
    state["backend"] = None


app = FastAPI(title="跨模态检索 API", version="0.1.0", lifespan=lifespan)


class TextQuery(BaseModel):
    query: str = Field(..., min_length=1, description="自然语言查询，中英文均可")
    k: int = Field(10, ge=1, le=100, description="返回 Top-K")


class SearchItem(BaseModel):
    image: str      # 图片 URL（配合 GET /images/...）
    name: str       # 文件名
    score: float    # 余弦相似度


class SearchResponse(BaseModel):
    results: List[SearchItem]


def _to_response(results) -> SearchResponse:
    return SearchResponse(
        results=[
            SearchItem(image=f"/images/{path.name}", name=path.name, score=round(score, 4))
            for path, score in results
        ]
    )


@app.get("/health")
def health():
    idx = state["index"]
    if idx is None:
        raise HTTPException(503, "服务尚未就绪")
    return {"status": "ok", "backend": state.get("backend"), "library_size": len(idx)}


@app.post("/api/search/text", response_model=SearchResponse)
def search_text(q: TextQuery):
    results = state["index"].search_by_text(q.query.strip(), q.k)
    return _to_response(results)


@app.post("/api/search/image", response_model=SearchResponse)
async def search_image(file: UploadFile = File(...)):
    from PIL import Image

    try:
        img = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(400, f"无法解析为图片: {file.filename}")
    results = state["index"].search_by_image(img, 10)
    return _to_response(results)


@app.get("/images/{name}")
def get_image(name: str):
    """按文件名返回图库图片。校验路径防止目录穿越（../../etc/passwd 之类）。"""
    idx = state["index"]
    safe = (idx.image_dir / name).resolve()
    if not safe.is_relative_to(idx.image_dir.resolve()) or not safe.exists():
        raise HTTPException(404, "图片不存在")
    return FileResponse(safe)


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="跨模态检索 REST 服务")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--trt", action="store_true",
                        help="用 TensorRT FP16 engine 做查询编码（需先 python -m cmr.trt build）")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
