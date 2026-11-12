import torch

from cmr.metrics import best_rank_per_query, retrieval_metrics


def _sim(rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_best_rank_single_gt():
    sim = _sim([
        [10.0, 9.0, 0.0],   # 查询0: 候选顺序 [0,1,2]
        [0.0, 10.0, 9.0],   # 查询1: 候选顺序 [1,2,0]
    ])
    best = best_rank_per_query(sim, [[0], [2]])
    assert best.tolist() == [1, 2]


def test_best_rank_multi_gt_takes_min():
    sim = _sim([[10.0, 9.0, 1.0]])  # 顺序 [0,1,2]
    best = best_rank_per_query(sim, [[1, 2]])
    assert best.tolist() == [2]


def test_recall_and_median():
    sim = _sim([
        [10.0, 9.0, 0.0],
        [0.0, 10.0, 9.0],
    ])
    m = retrieval_metrics(sim, [[0], [2]], ks=(1, 5, 10))
    assert m["R@1"] == 0.5  # 只有查询0的第一名正确
    assert m["R@5"] == 1.0
    assert m["R@10"] == 1.0
    assert m["MedR"] == 1.5  # median(1, 2)


def test_all_miss():
    sim = _sim([[1.0, 2.0, 3.0]])  # 顺序 [2,1,0]
    m = retrieval_metrics(sim, [[0]], ks=(1,))
    assert m["R@1"] == 0.0
    assert m["MedR"] == 3.0


def test_matrix_layout_matches_text_to_image():
    """文→图 指标 = 对 (n_img, n_cap) 相似度矩阵的转置逐行计算。"""
    g = torch.Generator().manual_seed(0)
    sim = torch.randn(5, 20, generator=g)  # 5 图 × 20 文
    gt_lists = [[4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3] for i in range(5)]
    gt_per_caption = [i for i in range(5) for _ in range(4)]

    i2t = retrieval_metrics(sim, gt_lists, ks=(1, 5))
    t2i = retrieval_metrics(sim.t(), [[g_] for g_ in gt_per_caption], ks=(1, 5))
    assert 0.0 <= i2t["R@1"] <= 1.0 and 0.0 <= t2i["R@1"] <= 1.0
