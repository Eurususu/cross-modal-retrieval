"""Gradio Web 演示：文本搜图（中/英）与以图搜图。

用法:
    python -m cmr.app                       # 默认读取 index/flickr8k.npz + checkpoints/best.pt
    python -m cmr.app --index my.npz --checkpoint my.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

from .config import PROJECT_ROOT, apply_overrides, load_config
from .index import load_index


def _patch_gradio_info_bug() -> None:
    """修复 gradio<=4.43 的 /info 接口崩溃（上游已在 4.44 修复）。

    gradio_client.utils.json_schema_to_python_type 遇到布尔型 JSON schema
    （如 Gallery 组件 FileData 定义里的 additionalProperties: false）会抛
    TypeError，导致 /info 500、前端点按钮报 "no api found"。
    包装一层：异常时降级返回 "Any"（/info 仅用于 API 文档展示，无害）。
    """
    import gradio_client.utils as gcu

    if getattr(gcu, "_cmr_patched", False):
        return
    original = gcu.json_schema_to_python_type

    def safe(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        try:
            return original(schema, defs)
        except Exception:
            return "Any"

    gcu.json_schema_to_python_type = safe
    gcu._cmr_patched = True


_patch_gradio_info_bug()

DEFAULT_INDEX = PROJECT_ROOT / "index" / "flickr8k.npz"
DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "best.pt"

EXAMPLE_QUERIES = [
    ["a dog running on the beach"],
    ["一只狗在沙滩上奔跑"],
    ["a man riding a bike in a park"],
    ["两个孩子在草地上踢足球"],
    ["someone jumping into the water"],
    ["夕阳下的城市天际线"],
]


def make_search_fns(index):
    def text_search(query: str, top_k: int) -> List[Tuple[str, str]]:
        if not query or not query.strip():
            return []
        results = index.search_by_text(query.strip(), int(top_k))
        return [(str(p), f"#{i+1} 相似度 {s:.3f}") for i, (p, s) in enumerate(results)]

    def image_search(image, top_k: int) -> List[Tuple[str, str]]:
        if image is None:
            return []
        results = index.search_by_image(image, int(top_k))
        return [(str(p), f"#{i+1} 相似度 {s:.3f}") for i, (p, s) in enumerate(results)]

    return text_search, image_search


def build_ui(index) -> gr.Blocks:
    text_search, image_search = make_search_fns(index)

    with gr.Blocks(title="跨模态检索演示") as demo:
        gr.Markdown(
            f"""
            # 🔎 跨模态检索演示（CLIP 风格双塔模型）
            图像塔 ResNet + 文本塔多语言 DistilBERT，对比学习对齐到同一嵌入空间。
            **支持中文/英文混合查询**；图库共 **{len(index)}** 张图片。
            """
        )
        with gr.Tab("📝 文本搜图（中/英）"):
            with gr.Row():
                with gr.Column(scale=3):
                    query = gr.Textbox(
                        label="查询文本",
                        placeholder="输入自然语言描述，例如: a dog running on the beach / 一只狗在沙滩上奔跑",
                        lines=1,
                    )
                with gr.Column(scale=1):
                    top_k_text = gr.Slider(1, 48, value=12, step=1, label="返回数量 Top-K")
            btn_text = gr.Button("🔍 检索", variant="primary")
            gallery_text = gr.Gallery(
                label="检索结果（按相似度降序）", columns=6, rows=2, height="auto", preview=True
            )
            gr.Examples(examples=EXAMPLE_QUERIES, inputs=[query], label="示例查询")
            btn_text.click(text_search, inputs=[query, top_k_text], outputs=gallery_text)
            query.submit(text_search, inputs=[query, top_k_text], outputs=gallery_text)

        with gr.Tab("🖼️ 以图搜图"):
            with gr.Row():
                with gr.Column(scale=1):
                    query_img = gr.Image(label="查询图片", type="pil", height=320)
                    top_k_img = gr.Slider(1, 48, value=12, step=1, label="返回数量 Top-K")
                    btn_img = gr.Button("🔍 检索", variant="primary")
                with gr.Column(scale=2):
                    gallery_img = gr.Gallery(
                        label="检索结果（按相似度降序）", columns=5, rows=2, height="auto", preview=True
                    )
            btn_img.click(image_search, inputs=[query_img, top_k_img], outputs=gallery_img)

        gr.Markdown(
            "提示: 相似度为归一化嵌入的余弦相似度（0~1）。"
            "模型在 Flickr8k（8000 张图）上训练，检索质量有限，用于学习与演示。"
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 Gradio 检索演示")
    parser.add_argument("--index", default=str(DEFAULT_INDEX), help="索引 npz 路径")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT), help="模型 checkpoint 路径")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)

    index_path = Path(args.index)
    ckpt_path = Path(args.checkpoint)
    if not index_path.exists():
        sys.exit(f"[app] 索引不存在: {index_path}\n请先运行: python -m cmr.index")
    if not ckpt_path.exists():
        sys.exit(f"[app] checkpoint 不存在: {ckpt_path}\n请先运行: python -m cmr.train")

    print(f"[app] 加载索引: {index_path}")
    index = load_index(index_path, ckpt_path)
    print(f"[app] 图库规模: {len(index)} 张图片")

    demo = build_ui(index)
    demo.queue()
    demo.launch(server_name=cfg.app.host, server_port=cfg.app.port)


if __name__ == "__main__":
    main()
