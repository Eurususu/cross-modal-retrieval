import torch
import torch.nn.functional as F

from cmr.losses import batch_top1_accuracy, clip_loss


def _rand_normalized(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return F.normalize(x, dim=-1)


def test_clip_loss_symmetric():
    """图→文 与 文→图 两个方向对称: 转置 logits 后损失不变。"""
    img = _rand_normalized(8, 16, seed=1)
    txt = _rand_normalized(8, 16, seed=2)
    logits = 10.0 * img @ txt.t()
    assert torch.allclose(clip_loss(logits), clip_loss(logits.t()), atol=1e-6)


def test_clip_loss_near_zero_on_perfect_pairs():
    """对角线为强正样本时损失应接近 0。"""
    feats = _rand_normalized(8, 16, seed=3)
    logits = 100.0 * feats @ feats.t()  # 对角=100, 非对角=100*cos<1
    assert clip_loss(logits).item() < 1e-4


def test_clip_loss_high_on_random():
    """完全随机的嵌入（正样本不突出）损失应明显大于 0。"""
    img = _rand_normalized(64, 8, seed=4)
    txt = _rand_normalized(64, 8, seed=5)
    logits = img @ txt.t()  # 无温度放大
    assert clip_loss(logits).item() > 1.0  # 接近 ln(64)≈4.16


def test_batch_top1_accuracy():
    feats = _rand_normalized(4, 8, seed=6)
    logits = feats @ feats.t()  # 对角最大
    assert batch_top1_accuracy(logits) == 1.0
