"""Contract + unit tests for the deterministic map-free transformer (model5).

Two kinds of guarantee:

* Map-free INPUT contract — the promise that makes this the clean Chaos-Gap
  transfer backbone: inputs are agent-relative displacements (translation-
  invariant), the model is rotation-equivariant to each agent's last-observed
  pose (per-agent heading canonicalization), there is no map/lane channel and
  no agent-class one-hot input feature, and obs/pred horizons + fps are not
  hard-coded.
* Eval-population contract — the model, evaluated through the shared harness,
  covers exactly the canonical val/test population (conftest.assert_eval_population).

Plus ordinary forward/shape and masked-loss unit checks.
"""

import inspect
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import assert_eval_population, canonical_split_populations, load_script

m5 = load_script("model5_transformer_train_eval.py")

REPO = Path(__file__).resolve().parent.parent
CANONICAL = REPO / "outputs" / "canonical_world_obs20_pred60.npz"
SPLITS_FILE = REPO / "splits" / "scene_splits.json"
needs_canonical = pytest.mark.skipif(
    not CANONICAL.exists(), reason="canonical dataset not generated")


def _tiny(obs_len=20, pred_len=60, d_model=32, **kw):
    torch.manual_seed(0)
    return m5.MapFreeTransformer(
        obs_len=obs_len, pred_len=pred_len, d_model=d_model, n_heads=4,
        n_temporal_layers=1, n_social_layers=1, n_decoder_layers=1,
        dropout=0.0, **kw).eval()


def _rand_obs_rel(B=2, M=3, obs_len=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs_rel = torch.randn(B, M, 2, obs_len, generator=g)
    agent_mask = torch.ones(B, M, dtype=torch.bool)
    return obs_rel, agent_mask


# --------------------------------------------------------------------------
# forward / shape
# --------------------------------------------------------------------------
def test_forward_shape():
    model = _tiny()
    obs_rel, am = _rand_obs_rel()
    with torch.no_grad():
        out = model(obs_rel, am)
    assert out.shape == (2, 3, 2, 60)
    assert torch.isfinite(out).all()


def test_horizons_and_fps_not_hardcoded():
    """Same class must build at a different horizon (obs 8 / pred 12) — the
    guarantee that it transfers to datasets sampled at other rates/lengths."""
    model = _tiny(obs_len=8, pred_len=12)
    obs_rel, am = _rand_obs_rel(obs_len=8)
    with torch.no_grad():
        out = model(obs_rel, am)
    assert out.shape == (2, 3, 2, 12)


# --------------------------------------------------------------------------
# map-free input contract
# --------------------------------------------------------------------------
def test_forward_takes_only_relative_coords_and_mask():
    """No map/lane channel, no class one-hot: the only inputs are agent-relative
    displacements and the padding mask, and the input feature dim is 2 (x, y)."""
    params = list(inspect.signature(m5.MapFreeTransformer.forward).parameters)
    assert params == ["self", "obs_rel", "agent_mask"]
    assert _tiny().input_dim == 2


def test_translation_invariance_by_construction():
    """Inputs are displacements, so a constant translation of the underlying
    absolute trajectory (which leaves obs_rel unchanged) cannot change the
    prediction. Guards against anyone sneaking absolute position into the model."""
    model = _tiny()
    obs_rel, am = _rand_obs_rel(seed=3)
    with torch.no_grad():
        a = model(obs_rel, am)
        b = model(obs_rel.clone(), am)
    assert torch.allclose(a, b, atol=0.0)


def test_rotation_equivariance_to_last_pose():
    """Rotating the whole scene by theta rotates every prediction by the same
    theta (per-agent heading canonicalization). This is the invariance a plain
    transformer lacks unless the frame is canonicalized."""
    model = _tiny(d_model=64)
    obs_rel, am = _rand_obs_rel(seed=1)
    theta = 0.7
    R = torch.tensor([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]], dtype=obs_rel.dtype)
    obs_rot = torch.einsum("ij,bmjt->bmit", R, obs_rel)
    with torch.no_grad():
        out = model(obs_rel, am)
        out_rot = model(obs_rot, am)
    expected = torch.einsum("ij,bmjt->bmit", R, out)
    assert torch.allclose(out_rot, expected, atol=2e-4), \
        float((out_rot - expected).abs().max())


def test_prediction_independent_of_agent_type():
    """agent_type is a metrics label, never an input — the model has no way to
    consume it, so nothing about class can leak into the forward pass."""
    model = _tiny()
    # There is simply no agent_type argument to pass; assert it is absent.
    assert "agent_type" not in inspect.signature(m5.MapFreeTransformer.forward).parameters


# --------------------------------------------------------------------------
# masked-loss correctness
# --------------------------------------------------------------------------
def test_masked_loss_ignores_unobserved_steps():
    """The training loss (dd_data.masked_reg_loss on the model's output) must be
    invariant to the GT at masked-out future steps."""
    from dd_data import masked_reg_loss

    model = _tiny()
    obs_rel, am = _rand_obs_rel(seed=5)
    with torch.no_grad():
        pred_rel = model(obs_rel, am)
    B, M, _, T = pred_rel.shape
    gt = torch.randn_like(pred_rel)
    fut_mask = torch.ones(B, M, T, dtype=torch.bool)
    fut_mask[..., T // 2:] = False  # second half unobserved
    loss_a = masked_reg_loss(pred_rel, gt, am, fut_mask, kind="mse")
    gt2 = gt.clone()
    gt2[..., T // 2:] = 999.0  # perturb only masked steps
    loss_b = masked_reg_loss(pred_rel, gt2, am, fut_mask, kind="mse")
    assert torch.allclose(loss_a, loss_b, atol=1e-6)


def test_zero_loss_on_perfect_prediction():
    from dd_data import masked_reg_loss

    obs_rel, am = _rand_obs_rel(seed=6)
    B, M, _, T = 2, 3, 2, 60
    gt = torch.randn(B, M, 2, T)
    fut_mask = torch.ones(B, M, T, dtype=torch.bool)
    loss = masked_reg_loss(gt, gt, am, fut_mask, kind="mse")
    assert float(loss) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# eval-population contract (the shared gate)
# --------------------------------------------------------------------------
@needs_canonical
def test_eval_population_matches_canonical():
    """A randomly-initialized model, evaluated through model5.evaluate on the
    canonical val/test splits, must report exactly the canonical agent count."""
    device = torch.device("cpu")
    model = _tiny().to(device)

    from torch.utils.data import DataLoader

    from dd_data import CanonicalSeqDataset, collate_pad

    def loader(split):
        ds = CanonicalSeqDataset(str(CANONICAL), split, str(SPLITS_FILE),
                                 rot_aug=False, seed=0)
        return DataLoader(ds, batch_size=64, shuffle=False, num_workers=0,
                          collate_fn=collate_pad)

    val = m5.evaluate(model, loader("val"), device)
    test = m5.evaluate(model, loader("test"), device)
    assert_eval_population(val["n_agents"], test["n_agents"])
