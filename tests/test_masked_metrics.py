"""Masked-metric math: hand-built cases where agents vanish mid-future."""

import numpy as np
import pytest
import torch

import dd_data


def test_masked_ade_fde_np_partial_future():
    # One agent, 4 future steps, valid at steps 0,1 only.
    gt = np.array([[[1.0, 0.0], [2.0, 0.0], [99.0, 99.0], [99.0, 99.0]]])  # (1,4,2)
    pred = np.array([[[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]]])
    mask = np.array([[True, True, False, False]])
    ade, fde, path = dd_data.masked_ade_fde_np(pred, gt, mask)
    # errors at valid steps: 1.0 and 2.0 -> ADE 1.5; FDE at last valid step -> 2.0
    assert ade[0] == pytest.approx(1.5)
    assert fde[0] == pytest.approx(2.0)
    # path over valid consecutive pairs: |(2,0)-(1,0)| = 1
    assert path[0] == pytest.approx(1.0)


def test_masked_ade_fde_np_invalid_steps_ignored():
    """Garbage values at masked steps must not affect anything."""
    gt = np.array([[[1.0, 0.0], [1e6, 1e6], [3.0, 0.0]]])
    pred = np.zeros((1, 3, 2))
    mask = np.array([[True, False, True]])
    ade, fde, _ = dd_data.masked_ade_fde_np(pred, gt, mask)
    assert ade[0] == pytest.approx((1.0 + 3.0) / 2)
    assert fde[0] == pytest.approx(3.0)


def _torch_batch(gt_np, pred_np, fut_mask_np):
    """Wrap (M,T,2)-style arrays as a (B=1, M, 2, T) torch batch."""
    gt = torch.from_numpy(np.transpose(gt_np, (0, 2, 1))[None]).float()
    pred = torch.from_numpy(np.transpose(pred_np, (0, 2, 1))[None]).float()
    fm = torch.from_numpy(fut_mask_np[None])
    am = torch.ones((1, gt_np.shape[0]), dtype=torch.bool)
    return pred, gt, am, fm


def test_per_agent_errors_torch_matches_numpy():
    rng = np.random.default_rng(3)
    gt = rng.normal(size=(4, 6, 2))
    pred = rng.normal(size=(4, 6, 2))
    mask = rng.random((4, 6)) > 0.3
    mask[:, 0] = True  # every agent has at least one valid step
    pred_t, gt_t, am, fm = _torch_batch(gt, pred, mask)
    ade_t, fde_t, path_t = dd_data.per_agent_errors(pred_t, gt_t, am, fm)
    ade_n, fde_n, path_n = dd_data.masked_ade_fde_np(
        pred.astype(np.float32), gt.astype(np.float32), mask)
    assert np.allclose(ade_t.numpy(), ade_n, atol=1e-5)
    assert np.allclose(fde_t.numpy(), fde_n, atol=1e-5)
    assert np.allclose(path_t.numpy(), path_n, atol=1e-5)


def test_bestofk_oracle_never_worse_than_any_mode():
    rng = np.random.default_rng(5)
    B, M, K, T = 1, 3, 4, 5
    gt = torch.from_numpy(rng.normal(size=(B, M, 2, T))).float()
    pred_k = torch.from_numpy(rng.normal(size=(B, M, K, 2, T))).float()
    am = torch.ones((B, M), dtype=torch.bool)
    fm = torch.ones((B, M, T), dtype=torch.bool)
    ade_best, _, _ = dd_data.per_agent_errors_bestofk(pred_k, gt, am, fm)
    for k in range(K):
        ade_k, _, _ = dd_data.per_agent_errors(pred_k[:, :, k], gt, am, fm)
        assert (ade_best <= ade_k + 1e-6).all()


def test_masked_variety_loss_ignores_invalid_displacements():
    B, M, K, T = 1, 1, 2, 4
    gt_rel = torch.zeros((B, M, 2, T))
    pred_k = torch.zeros((B, M, K, 2, T))
    # Mode 0 is perfect on valid steps but wild on the masked step;
    # mode 1 is slightly off everywhere. Mask must make mode 0 win.
    fut_mask = torch.tensor([[[True, True, False, True]]])
    pred_k[0, 0, 0, :, 2] = 100.0
    pred_k[0, 0, 1] = 0.5
    loss, best_mode = dd_data.masked_variety_loss(pred_k, gt_rel,
                                                  torch.ones((B, M), dtype=torch.bool),
                                                  fut_mask)
    assert best_mode[0, 0].item() == 0
    assert loss.item() == pytest.approx(0.0, abs=1e-9)


def test_ewta_top_w_reduces_to_wta_and_full_mean():
    rng = np.random.default_rng(11)
    B, M, K, T = 2, 3, 4, 5
    gt = torch.from_numpy(rng.normal(size=(B, M, 2, T))).float()
    pred_k = torch.from_numpy(rng.normal(size=(B, M, K, 2, T))).float()
    am = torch.ones((B, M), dtype=torch.bool)
    fm = torch.rand(B, M, T) > 0.3
    fm[..., 0] = True

    loss_default, bm_default = dd_data.masked_variety_loss(pred_k, gt, am, fm)
    loss_w1, bm_w1 = dd_data.masked_variety_loss(pred_k, gt, am, fm, top_w=1)
    assert torch.allclose(loss_default, loss_w1)
    assert (bm_default == bm_w1).all()

    # top_w=K averages ALL modes: equals mean of per-mode masked SSE / (n*2*K)
    loss_wK, _ = dd_data.masked_variety_loss(pred_k, gt, am, fm, top_w=K)
    rv = am.unsqueeze(-1) & dd_data.rel_valid_mask(fm)
    m = rv.unsqueeze(2).unsqueeze(3).float()
    diff = (pred_k - gt.unsqueeze(2)) * m
    expected = (diff * diff).sum() / (rv.float().sum() * 2.0 * K)
    assert torch.allclose(loss_wK, expected, atol=1e-6)

    # the best-w average can never exceed the all-K average
    for w in range(1, K + 1):
        lw, _ = dd_data.masked_variety_loss(pred_k, gt, am, fm, top_w=w)
        assert lw.item() <= loss_wK.item() + 1e-6


def test_ewta_masking_with_relaxed_winners():
    """Garbage at masked steps must not leak even when w > 1."""
    B, M, K, T = 1, 1, 3, 4
    gt_rel = torch.zeros((B, M, 2, T))
    pred_k = torch.zeros((B, M, K, 2, T))
    fut_mask = torch.tensor([[[True, True, False, True]]])
    pred_k[0, 0, 0, :, 2] = 1e6   # only at the masked step
    pred_k[0, 0, 1, :, 2] = 1e6
    pred_k[0, 0, 2, :, 2] = 1e6
    loss, _ = dd_data.masked_variety_loss(pred_k, gt_rel,
                                          torch.ones((B, M), dtype=torch.bool),
                                          fut_mask, top_w=2)
    assert loss.item() == pytest.approx(0.0, abs=1e-9)


def test_per_class_summary_consistency():
    rng = np.random.default_rng(7)
    n = 300
    ade = rng.uniform(1, 5, n)
    fde = rng.uniform(2, 10, n)
    path = rng.uniform(5, 40, n)
    types = np.array(["Car"] * 150 + ["Pedestrian"] * 100 + ["Animal"] * 50, dtype=object)
    s = dd_data.per_class_summary(ade, fde, path, types, min_class_n=50)
    total_n = sum(v["n"] for v in s["classes"].values())
    assert total_n == n
    # pooled ADE must be reproducible from per-class means
    pooled = sum(v["n"] * v["ade"] for v in s["classes"].values()) / n
    assert pooled == pytest.approx(float(ade.mean()))
    assert s["ade_disparity_worst_over_best"] >= 1.0
