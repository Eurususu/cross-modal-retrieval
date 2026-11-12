# 跨模态检索（Cross-Modal Retrieval）

从零实现 CLIP 风格的**双塔模型 + 对比学习**，在 Flickr8k 上训练，支持**中英文混合查询**的以文搜图 / 以图搜图，并附带 Gradio 演示界面。

```text
   图片                              文本（中文/英文）
    │                                  │
 ResNet50 骨干                   多语言 DistilBERT 骨干
    │                                  │
  投影头 MLP  ────── 共享嵌入空间 ──────  投影头 MLP
    │                                  │
    └────────── L2 归一化 ──────────────┘
                     │
        对称 InfoNCE 对比损失（可学习温度）
       拉近 正样本对（图, 描述）· 推远 batch 内负样本
```

- **图→文检索**：给一张图，找最匹配的文本描述；**文→图检索**：输入一句话，返回最相关的图片
- 评测采用标准协议：**R@1 / R@5 / R@10 / MedR**（双向）

> 关于「从零」的边界：双塔结构、投影头、对称 InfoNCE 损失、可学习温度、训练与评测流程全部自行实现。骨干网络用 ImageNet / 多语言预训练权重初始化——Flickr8k 只有 8k 图（CLIP 训练用了 4 亿对数据），随机初始化训不出有效嵌入。想体验对照实验可设 `--set model.image_pretrained=false`。

## 环境要求

- Python ≥ 3.9（Python 3.13 需要 PyTorch ≥ 2.5）
- 建议有 NVIDIA GPU（8GB 显存即可舒服训练）；纯 CPU 也能跑通但很慢
- 磁盘：数据集约 1.2GB，模型 checkpoint 约 500MB

## 安装

```bash
cd "cross_modal retrieval"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

已有现成 conda 环境也可直接复用（本机实测在含 torch 2.8+cu128 / transformers 4.57 / gradio 4.31 的环境下全部功能可用），只需补装 `pytest`。

## 快速开始（5 条命令）

```bash
# 1. 下载数据集（Flickr8k, ~1.1GB，来自 jbrownlee/Datasets 镜像）
python -m cmr.download

# 2. 训练（默认 10 epochs；GPU 上约 20~40 分钟）
python -m cmr.train

# 3. 评测（test 集 1000 图 / 5000 描述，输出 R@K 与 MedR）
python -m cmr.evaluate

# 4. 对整个图片库建立向量索引
python -m cmr.index

# 5. 启动 Web 演示，浏览器打开 http://127.0.0.1:7860
python -m cmr.app
```

单元测试（无需数据集与预训练权重）：

```bash
pytest -q
```

## 中文检索与双语增强（可选但推荐）

文本塔是多语言模型，**中文查询开箱即用**（跨语言零样本迁移）。想进一步提升中文效果，可把英文标注机翻成中文参与训练：

```bash
pip install sentencepiece        # 翻译模型依赖
python -m cmr.translate_captions # 40k 条描述，GPU 几分钟，支持断点续跑
python -m cmr.evaluate --lang both
```

生成 `Flickr8k.token.zh.txt` 后重新训练即可自动混用中英文（`data.mix_chinese: true`）。

## 详细说明

### 训练

```bash
python -m cmr.train --set train.epochs=20 train.batch_size=32
```

- 官方 split：6000 训练 / 1000 验证 / 1000 测试
- 每个 epoch 结束在验证集上评测，`R@1` 均值最高的保存为 `checkpoints/best.pt`，最新状态保存为 `checkpoints/last.pt`
- 断点续训：`python -m cmr.train --set train.resume=checkpoints/last.pt`
- 显存不足：调小 `train.batch_size`（对比学习批越大负样本越多、效果越好，尽量别低于 32）
- 监控指标：`loss`（训练损失）、`acc`（batch 内图→文 top1 命中率）、`T`（可学习温度）

### 评测

```bash
python -m cmr.evaluate --checkpoint checkpoints/best.pt --lang both
```

输出示例与实测参考值（默认配置：batch 64 × 10 epochs，RTX 3090；数值随超参/硬件浮动，重训后以 `reports/` 里的报告为准）：

```text
===== 测试集 · 英文标注 (1000 图 / 5000 描述) =====
  图→文      R@1 0.3040 | R@5 0.6010 | R@10 0.7220 | MedR 4
  文→图      R@1 0.2288 | R@5 0.5176 | R@10 0.6622 | MedR 5
```

- **图→文**：一张图的 5 条描述都算正确结果；**文→图**：一条描述只对应一张正确图
- 中文列为机翻标注上的指标，仅供参考
- JSON 报告保存在 `reports/`

### 索引与检索

```bash
python -m cmr.index --images data/flickr8k/images --out index/flickr8k.npz
python -m cmr.app --index index/flickr8k.npz
```

`--images` 可以指向**任何图片文件夹**（不限于 Flickr8k），即可用训练好的模型对自己的图片库做检索。局域网访问：`--set app.host=0.0.0.0`。

## 项目结构

```text
├── configs/default.yaml     # 全部超参（数据/模型/训练/评测/演示）
├── cmr/                     # 核心包（python -m cmr.<模块> 运行）
│   ├── config.py            # dataclass + YAML + 命令行覆盖（配置随 checkpoint 存档）
│   ├── model.py             # 双塔: ImageEncoder / TextEncoder / DualEncoder
│   ├── losses.py            # 对称 InfoNCE（CLIP 风格）损失
│   ├── data.py              # Flickr8k 解析、Dataset、图像增强
│   ├── download.py          # 数据集下载与目录归一化
│   ├── train.py             # 训练主循环（AMP、warmup+余弦、断点续训）
│   ├── evaluate.py          # R@K / MedR 双向评测
│   ├── translate_captions.py# 中英标注机翻（双语增强，可选）
│   ├── index.py             # 向量索引 + 可注入查询编码器（PyTorch/TRT 同接口）
│   ├── app.py               # Gradio 演示界面
│   ├── serve.py             # REST API 服务（FastAPI，双后端 --trt 可选）
│   ├── export_onnx.py       # 双塔导出 ONNX（int32 文本输入包装）
│   ├── trt.py               # TensorRT 构建/校验/基准 + TRTQueryEncoder
│   ├── embedding.py         # 批量编码工具
│   ├── metrics.py           # 检索指标（R@K / MedR）
│   ├── checkpoint.py        # checkpoint 保存/加载
│   └── utils.py             # 种子/设备/学习率调度
├── tests/                   # 单元测试 + REST API 集成测试
├── conftest.py              # 让 pytest 能 import cmr
├── .vscode/launch.json      # F5 调试配置（module 方式，双后端环境）
├── Dockerfile               # REST 服务容器模板（CPU 推理）
└── requirements.txt
```

## 测试

```bash
pytest -q                       # 全量 22 个（约 15~30 秒）
pytest -q --ignore=tests/test_api.py   # 只跑 16 个单元测试（无需训练产物）
```

- **16 个单元测试**：损失对称性/数值、检索指标（R@K/MedR 的手工构造用例）、模型结构（归一化、冻结、mean pooling）、标注解析——不需要数据集和预训练权重
- **6 个 API 集成测试**（`tests/test_api.py`）：进程内起服务打全部接口——参数校验 422、结果降序、图片自匹配 = 1.0、路径穿越攻击 404 等；**没有训练产物时自动跳过**（CI 友好）

## VSCode 调试

[.vscode/launch.json](.vscode/launch.json) 提供两个 F5 配置（Gradio 演示 / 训练），关键是用 `"module": "cmr.app"` 代替 `"program"`——保持 `python -m` 的包上下文，相对导入不失效；断点、单步、变量查看与普通脚本一致。

## 常见问题

**下载 HuggingFace 模型很慢或失败**

国内网络建议使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**torchvision 预训练权重（ResNet）下载失败**
权重来自 download.pytorch.org，一般可直接访问；也可手动下载后放到 `~/.cache/torch/hub/checkpoints/`。

**显存不足（CUDA out of memory）**
`--set train.batch_size=32`（或 16），或减小图片尺寸 `--set data.image_size=160`。

**只有 CPU 能训练吗？**
能，但预计每个 epoch 1 小时以上。建议减少 epoch（`--set train.epochs=3`）先跑通流程。

**sentencepiece 安装失败（Python 3.13）**
它是可选项，只影响 `translate_captions`；不装不影响其他功能。若需要，在 Python ≤3.12 的环境里运行翻译脚本。

**Flickr8k 下载失败**
脚本会打印手动指引：把两个 zip 放入 `data/flickr8k/_archives/` 后重新运行 `python -m cmr.download`。

**Gradio 页面能打开但点击按钮报 "no api found"**
gradio ≤4.43 的 `/info` 接口在 Gallery 等 schema 上崩溃（`app.py` 已内置猴子补丁自动修复）；若仍出现，升级到 gradio ≥4.44 可根治。

## TensorRT 部署（FP16 加速）

```bash
python -m cmr.export_onnx      # ① 双塔导出 ONNX（含 onnxruntime 数值校验）
python -m cmr.trt build        # ② trtexec 构建 FP16 engine（RTX 3090 实测）
python -m cmr.trt verify       # ③ 精度对齐 + 端到端检索一致性
python -m cmr.trt bench        # ④ 性能对比
```

RTX 3090 实测（单次前向延迟）：

| | PyTorch FP32 | TRT FP16 | 加速比 |
| --- | --- | --- | --- |
| 图像塔 B=1 | 5.01 ms | 1.39 ms | 3.6x |
| 图像塔 B=32 | 26.75 ms | 9.16 ms | 2.9x |
| 文本塔 B=1 | 2.94 ms | 0.97 ms | 3.0x |
| 文本塔 B=32 | 13.61 ms | 2.80 ms | 4.9x |

FP16 精度损失：两塔输出与 FP32 的余弦相似度 > 0.9999，端到端检索 Top1 不变。
注意事项：**engine 与 GPU 型号和 TensorRT 版本绑定**，换硬件/升级 TRT 需重新 `build`；tokenizer/图像预处理/向量检索不在 TRT 内，仍由 Python 侧完成。

## 部署（REST API 服务）

Gradio 是演示界面；对外提供服务用 FastAPI 版本（[cmr/serve.py](cmr/serve.py)）：

```bash
python -m cmr.serve --port 8000        # PyTorch 后端（默认），模型+索引启动时加载一次
python -m cmr.serve --trt              # TensorRT FP16 后端（需先 python -m cmr.trt build）
```

接口（实测可用；启动后浏览器打开 `http://127.0.0.1:8000/docs` 有可交互的 Swagger 文档，Try it out 直接调）：

| 接口 | 说明 |
| --- | --- |
| `GET /health` | 健康检查，返回后端类型与图库规模 |
| `POST /api/search/text` | JSON `{"query": "一只狗在沙滩上", "k": 10}` |
| `POST /api/search/image` | multipart 文件上传（字段 `file`） |
| `GET /images/{name}` | 结果图片的静态访问 |

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/api/search/text \
     -H "Content-Type: application/json" \
     -d '{"query": "a dog on the beach", "k": 5}'
```

更大规模的演进路径：`uvicorn --workers N` 多进程（注意每进程各加载一份模型）→ nginx 反代做 TLS/限流 → Docker 容器化（[Dockerfile](Dockerfile)，CPU 推理模板）→ 图库上百万时用 FAISS 替换 numpy 暴力检索 → 需要脱离 Python 依赖时导出 ONNX。

## 扩展方向

- 换更强的图像骨干（`--set model.image_arch=resnet101`）或文本骨干
- 增加 `Flickr30k / MSCOCO` 数据（只需仿照 `data.py` 增加解析器）
- 用 FAISS 替换 numpy 暴力检索，支持百万级图库
- 加 queue-based 对比损失（MoCo 风格）缓解小 batch 负样本不足
- Rerank：用 cross-encoder 对 top-K 精排

## 声明

Flickr8k 数据集图片与标注仅供学习研究使用；本项目代码可自由参考。
