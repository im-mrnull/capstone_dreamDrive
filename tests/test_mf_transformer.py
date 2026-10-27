"""Multi-future map-free transformer (scripts/model8_mf_transformer_train_eval.py)."""

import torch

from conftest import load_script

m8 = load_script("model8_mf_transformer_train_eval.py")


def make(k=4):
    torch.manual_seed(0)
    return m8.MultiFutureTransformer(obs_len=20, pred_len=12, k=k, d_model=32, n_heads=4,
                                     n_temporal_layers=1, n_social_layers=1, n_decoder_layers=1).eval()


def test_shapes_and_probabilities():
    model = make(k=4)
    obs = torch.randn(2, 3, 2, 20) * 0.3
    mask = torch.tensor([[True, True, False], [True, True, True]])
    pred, logits = model(obs, mask)
    assert pred.shape == (2, 3, 4, 2, 12) and logits.shape == (2, 3, 4)
    assert torch.isfinite(pred).all() and torch.isfinite(logits).all()


def test_rotation_equivariance_of_every_mode():
    """Rotating the observed motion rotates every predicted future the same way."""
    model = make(k=3)
    obs = torch.randn(1, 2, 2, 20) * 0.3 + torch.tensor([0.5, 0.0]).view(1, 1, 2, 1)
    mask = torch.ones(1, 2, dtype=torch.bool)
    th = torch.tensor(0.7)
    R = torch.tensor([[torch.cos(th), -torch.sin(th)], [torch.sin(th), torch.cos(th)]])
    p1, l1 = model(obs, mask)
    p2, l2 = model(torch.einsum("ij,bmjt->bmit", R, obs), mask)
    assert torch.allclose(torch.einsum("ij,bmkjt->bmkit", R, p1), p2, atol=1e-4)
    assert torch.allclose(l1, l2, atol=1e-4)


def test_variety_loss_trains_the_heads():
    model = make(k=4).train()
    obs = torch.randn(2, 3, 2, 20) * 0.3
    gt = torch.randn(2, 3, 2, 12) * 0.3
    mask = torch.ones(2, 3, dtype=torch.bool)
    fm = torch.ones(2, 3, 12, dtype=torch.bool)
    from dd_data import masked_variety_loss
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(30):
        pred, logits = model(obs, mask)
        loss, best = masked_variety_loss(pred, gt, mask, fm)
        loss = loss + 0.1 * torch.nn.functional.cross_entropy(logits[mask], best[mask])
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else float(loss)
    assert float(loss) < first
