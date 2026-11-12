# 跨模态检索 REST 服务 Docker 镜像模板
# 注意: 本文件为模板，未在当前机器构建验证过（构建需拉取基础镜像，走网络）。
# CPU 推理版本；GPU 推理把基础镜像换成 nvidia/cuda 运行时镜像并在 run 时加 --gpus。
#
# 构建:   docker build -t cmr-serve .
# 运行:   docker run -p 8000:8000 \
#           -v $(pwd)/checkpoints:/app/checkpoints \
#           -v $(pwd)/index:/app/index \
#           -v $(pwd)/data/flickr8k/images:/app/data/flickr8k/images \
#           cmr-serve
# （模型/索引/图片用 volume 挂载进容器，镜像本身只含代码和依赖）

FROM python:3.11-slim

WORKDIR /app

# PyTorch CPU 版（体积小很多；GPU 部署改用官方 cu12x wheel）
RUN pip install --no-cache-dir torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
      "transformers>=4.44" fastapi "uvicorn[standard]" python-multipart \
      pillow numpy pyyaml tqdm

# 代码
COPY cmr/ ./cmr/
COPY configs/ ./configs/

# 预训练权重缓存（可选: 也可在启动时在线拉取）
# COPY hf_cache/ /root/.cache/huggingface/

EXPOSE 8000
CMD ["python", "-m", "cmr.serve", "--host", "0.0.0.0", "--port", "8000"]
