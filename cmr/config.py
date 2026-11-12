"""配置管理：dataclass + YAML + 命令行覆盖。

用法:
    cfg = load_config("configs/default.yaml")
    apply_overrides(cfg, ["train.epochs=20", "train.batch_size=32"])
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class DataConfig:
    root: str = "data/flickr8k"
    image_dir: str = "images"
    captions_file: str = "Flickr8k.token.txt"
    captions_zh_file: str = "Flickr8k.token.zh.txt"
    mix_chinese: bool = True
    image_size: int = 224
    max_text_len: int = 64
    num_workers: int = 4

    @property
    def root_path(self) -> Path:
        return _resolve(self.root)

    @property
    def images_path(self) -> Path:
        return self.root_path / self.image_dir

    @property
    def captions_path(self) -> Path:
        return self.root_path / self.captions_file

    @property
    def captions_zh_path(self) -> Path:
        return self.root_path / self.captions_zh_file


@dataclass
class ModelConfig:
    image_arch: str = "resnet50"
    image_pretrained: bool = True
    text_model: str = "distilbert-base-multilingual-cased"
    freeze_image_backbone: bool = False
    freeze_text_backbone: bool = False
    embed_dim: int = 512


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 10
    lr: float = 3.0e-5
    head_lr: float = 1.0e-4
    weight_decay: float = 0.2
    warmup_ratio: float = 0.05
    grad_clip: Optional[float] = 1.0
    amp: bool = True
    log_every: int = 25
    val_each_epoch: bool = True
    checkpoint_dir: str = "checkpoints"
    resume: Optional[str] = None

    @property
    def checkpoint_dir_path(self) -> Path:
        return _resolve(self.checkpoint_dir)

    @property
    def resume_path(self) -> Optional[Path]:
        return _resolve(self.resume) if self.resume else None


@dataclass
class EvalConfig:
    recall_ks: List[int] = field(default_factory=lambda: [1, 5, 10])
    batch_size: int = 64
    report_dir: str = "reports"

    @property
    def report_dir_path(self) -> Path:
        return _resolve(self.report_dir)


@dataclass
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    top_k: int = 12


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    app: AppConfig = field(default_factory=AppConfig)


def _resolve(path: str | Path) -> Path:
    """相对路径一律锚定到项目根目录，保证在任何工作目录下运行都正确。"""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


_SECTION_TYPES = {
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
    "app": AppConfig,
}


def config_from_dict(raw: dict) -> Config:
    """由（嵌套）字典构造 Config，也用于从 checkpoint 恢复配置。"""
    kwargs: dict = {}
    for key, value in raw.items():
        if key in _SECTION_TYPES and isinstance(value, dict):
            known = {f.name for f in dataclasses.fields(_SECTION_TYPES[key])}
            kwargs[key] = _SECTION_TYPES[key](**{k: v for k, v in value.items() if k in known})
        elif hasattr(Config, key):
            kwargs[key] = value
    return Config(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return config_from_dict(raw)


def apply_overrides(cfg: Config, overrides: List[str]) -> Config:
    """按 `a.b=value` 形式覆盖配置，value 用 YAML 规则解析（支持数字/布尔/列表）。"""
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"[config] --set 参数需要 key=value 形式，收到: {item!r}")
        obj = cfg
        parts = [p for p in key.strip().split(".") if p]
        if not parts:
            raise SystemExit(f"[config] 非法的配置键: {item!r}")
        try:
            for part in parts[:-1]:
                obj = getattr(obj, part)
            current = getattr(obj, parts[-1])
        except AttributeError:
            raise SystemExit(f"[config] 未知的配置键: {key!r}")
        value = yaml.safe_load(raw)
        if isinstance(current, bool):
            value = bool(value)  # 兼容 --set model.image_pretrained=1 这类写法
        setattr(obj, parts[-1], value)
    return cfg
