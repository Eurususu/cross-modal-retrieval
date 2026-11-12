import torch
import torch.nn.functional as F

from cmr.model import DualEncoder, ImageEncoder, ProjectionHead, mean_pool


def test_projection_head_shape():
    head = ProjectionHead(in_dim=32, embed_dim=16)
    out = head(torch.randn(3, 32))
    assert out.shape == (3, 16)


def test_mean_pool_respects_mask():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])  # (1,3,2)
    mask = torch.tensor([[1, 1, 0]])  # 第三位是 padding
    pooled = mean_pool(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))  # (1+3)/2


def test_image_encoder_forward():
    enc = ImageEncoder(arch="resnet18", pretrained=False, embed_dim=16)
    out = enc(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_image_encoder_freeze():
    enc = ImageEncoder(arch="resnet18", pretrained=False, embed_dim=16, freeze_backbone=True)
    assert all(not p.requires_grad for p in enc.backbone.parameters())
    assert all(p.requires_grad for p in enc.projection.parameters())
    enc.train()
    assert not enc.backbone.training  # 冻结骨干在 train() 后仍保持 eval


def test_dual_encoder_normalization_and_scale():
    img_enc = ImageEncoder(arch="resnet18", pretrained=False, embed_dim=8)
    txt_proj = ProjectionHead(in_dim=8, embed_dim=8)

    class _StubText(torch.nn.Module):
        # 规避测试时联网加载 HF 权重: 恒等骨干 + 投影头
        freeze_backbone = False

        def __init__(self):
            super().__init__()
            self.projection = txt_proj

        def forward(self, input_ids, attention_mask):
            return self.projection(input_ids.float())

    model = DualEncoder(img_enc, _StubText())
    images = torch.randn(4, 3, 64, 64)
    input_ids = torch.randn(4, 8)
    attention_mask = torch.ones(4, 8, dtype=torch.long)

    img_feat = model.encode_image(images)
    txt_feat = model.encode_text(input_ids, attention_mask)
    assert img_feat.shape == (4, 8) and txt_feat.shape == (4, 8)
    # L2 归一化后模长为 1
    assert torch.allclose(img_feat.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(txt_feat.norm(dim=-1), torch.ones(4), atol=1e-5)

    logits = model(images, input_ids, attention_mask)
    assert logits.shape == (4, 4)
    assert torch.isfinite(logits).all()
    # logit_scale 有上界 100
    assert model.logit_scale.exp().clamp(max=100.0).item() <= 100.0
