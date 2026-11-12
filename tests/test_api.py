"""REST API 集成测试: 用 FastAPI TestClient 在进程内起服务（不占端口）。

与单元测试不同, 这些测试需要训练产物（best.pt + 索引）, 缺失时自动跳过:
    pytest -q                # 有产物就全跑
    pytest -q tests/test_api.py -v
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT = PROJECT_ROOT / "checkpoints" / "best.pt"
INDEX = PROJECT_ROOT / "index" / "flickr8k.npz"
IMAGES = PROJECT_ROOT / "data" / "flickr8k" / "images"

pytestmark = pytest.mark.skipif(
    not (CKPT.exists() and INDEX.exists()),
    reason="需要训练产物 checkpoints/best.pt 与 index/flickr8k.npz",
)


@pytest.fixture(scope="module")
def client():
    """module 级共享: 模型只加载一次（约 10~15 秒）, 该模块所有测试复用。"""
    from cmr.serve import app

    with TestClient(app) as c:  # with 语句触发 lifespan（加载模型）
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["library_size"] > 0


def test_text_search_returns_ranked_results(client):
    r = client.post("/api/search/text", json={"query": "a dog on the beach", "k": 3})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 3
    assert all(set(x) == {"image", "name", "score"} for x in results)
    scores = [x["score"] for x in results]
    assert scores == sorted(scores, reverse=True)  # 按相似度降序


def test_text_search_rejects_bad_params(client):
    assert client.post("/api/search/text", json={"query": "", "k": 3}).status_code == 422
    assert client.post("/api/search/text", json={"query": "cat", "k": 999}).status_code == 422
    assert client.post("/api/search/text", json={"k": 3}).status_code == 422


def test_image_search_self_match(client):
    """上传图库里的图, Top1 必须是它自己（余弦相似度 = 1.0）。"""
    sample = next(IMAGES.glob("*.jpg"))
    with open(sample, "rb") as f:
        r = client.post("/api/search/image", files={"file": ("q.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    top = r.json()["results"][0]
    assert top["name"] == sample.name
    assert top["score"] > 0.99


def test_image_search_rejects_garbage(client):
    r = client.post("/api/search/image", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400


def test_static_image_and_path_traversal(client):
    sample = next(IMAGES.glob("*.jpg"))
    assert client.get(f"/images/{sample.name}").status_code == 200
    # 目录穿越攻击必须被挡住
    assert client.get("/images/..%2F..%2F..%2Fetc%2Fpasswd").status_code == 404
    assert client.get("/images/nonexistent.jpg").status_code == 404
