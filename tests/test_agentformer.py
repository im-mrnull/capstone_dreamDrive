"""Contract + unit tests for the upstream-AgentFormer integration (model6).

Two kinds of guarantee:

* Map-free adapter contract — the config is genuinely map-free (no map channel,
  no per-agent heading input), and scripts/dd_agentformer.seq_to_af_data turns a
  canonical sequence into AgentFormer's data dict with one row per agent, masked
  future steps copy-forward-filled.
* Eval-population contract — the AgentFormer eval iterates EXACTLY the canonical
  val/test population. Proved two ways that are cheap on CPU (AgentFormer's 60-step
  autoregressive decode is too slow to run the whole population in a unit test):
  (1) structurally, the per-split scene sizes sum to the canonical agent counts;
  (2) on a few scenes, evaluate() scores every agent it is handed (no drop/dup).

Plus forward/inference shape checks and the best-of-K oracle invariant. Every
model-dependent test skips cleanly if the AgentFormer stack is unavailable
(repo not cloned/patched, or easydict/glob2 not installed).
"""

import math
from pathlib import Path

import numpy as np
import pytest

from conftest import canonical_split_populations, load_script

REPO = Path(__file__).resolve().parent.parent
CANONICAL = REPO / "outputs" / "canonical_world_obs20_pred60.npz"
SPLITS_FILE = REPO / "splits" / "scene_splits.json"

needs_canonical = pytest.mark.skipif(
    not CANONICAL.exists(), reason="canonical dataset not generated")

# Import the AgentFormer stack once; skip model tests cleanly if unavailable.
_af_err = None
try:
    import torch  # noqa: F401
    import dd_agentformer as dda
    dda.add_af_to_path()
    from model.model_lib import model_dict  # noqa: F401
    m6 = load_script("model6_agentformer_train_eval.py")
except Exception as e:  # pragma: no cover - environment-dependent
    _af_err = repr(e)
needs_af = pytest.mark.skipif(_af_err is not None, reason=f"AgentFormer stack unavailable: {_af_err}")

OBS, PRED = 20, 60


def _tiny_model(nz=8):
    import torch
    torch.manual_seed(0)
    cfg = dda.build_map_free_cfg(OBS, PRED, nz=nz, sample_k=2, tf_model_dim=32,
                                 tf_ff_dim=64, tf_nhead=4, nlayer=1,
                                 out_mlp_dim=None, rand_rot_scene=False)
    model = model_dict["agentformer"](cfg)
    model.set_device(torch.device("cpu"))
    model.eval()
    return model, cfg


def _synth_scene(M=4, seed=0):
    g = np.random.default_rng(seed)
    obs = g.standard_normal((M, 2, OBS)).astype(np.float32)
    pred = g.standard_normal((M, 2, PRED)).astype(np.float32)
    fm = np.ones((M, PRED), dtype=bool)
    return obs, pred, fm


# --------------------------------------------------------------------------
# map-free adapter contract
# --------------------------------------------------------------------------
@needs_af
def test_config_is_map_free():
    cfg = dda.build_map_free_cfg(OBS, PRED)
    assert cfg.use_map is False
    assert "map" not in cfg.input_type and "map" not in cfg.fut_input_type
    assert "heading" not in cfg.input_type and "heading" not in cfg.fut_input_type
    assert cfg.dec_input_type == []          # decoder consumes only its own output + z
    assert cfg.learn_prior is False          # N(0,I) prior -> z=0 is the most-likely mode
    assert cfg.past_frames == OBS and cfg.future_frames == PRED


@needs_af
def test_seq_to_af_data_shapes_and_scaling():
    import torch
    obs, pred, fm = _synth_scene(M=3)
    data = dda.seq_to_af_data(obs, pred, fm, traj_scale=10.0)
    assert data["heading"] is None and data["scene_map"] is None
    assert len(data["pre_motion_3D"]) == 3 == len(data["fut_motion_3D"])
    assert tuple(data["pre_motion_3D"][0].shape) == (OBS, 2)
    assert tuple(data["fut_motion_3D"][0].shape) == (PRED, 2)
    assert tuple(data["fut_motion_mask"][0].shape) == (PRED,)
    # positions are scaled into world/traj_scale units
    expected = torch.from_numpy(np.transpose(obs, (0, 2, 1))[0] / 10.0).float()
    assert torch.allclose(data["pre_motion_3D"][0], expected, atol=1e-6)


@needs_af
def test_masked_future_steps_copy_forward():
    obs, pred, fm = _synth_scene(M=2)
    fm[:, 30:] = False                       # second half unobserved
    data = dda.seq_to_af_data(obs, pred, fm, traj_scale=1.0)
    fut = data["fut_motion_3D"][0].numpy()
    # every masked step equals the last valid position (copy-forward), so no NaNs
    assert np.isfinite(fut).all()
    assert np.allclose(fut[30:], fut[29])


@needs_af
def test_keyframes_native_2hz():
    obs_idx, fut_idx = dda.keyframe_indices(OBS, PRED, 5)
    assert obs_idx.tolist() == [4, 9, 14, 19]      # ends at the prediction anchor
    assert fut_idx.tolist() == list(range(4, 60, 5))  # t+0.5 s ... t+6.0 s, 12 frames
    o1, f1 = dda.keyframe_indices(OBS, PRED, 1)
    assert o1.tolist() == list(range(OBS)) and f1.tolist() == list(range(PRED))
    with pytest.raises(ValueError):
        dda.keyframe_indices(OBS, PRED, 7)


@needs_af
def test_seq_to_af_data_with_stride_keeps_keyframes():
    obs, pred, fm = _synth_scene(M=2)
    fm[1, 12:] = False
    data = dda.seq_to_af_data(obs, pred, fm, traj_scale=1.0, stride=5)
    assert tuple(data["pre_motion_3D"][0].shape) == (4, 2)
    assert tuple(data["fut_motion_3D"][0].shape) == (12, 2)
    assert np.allclose(data["pre_motion_3D"][0].numpy()[-1], obs[0, :, -1])
    assert np.allclose(data["fut_motion_3D"][0].numpy()[0], pred[0, :, 4])
    # agent 1 is valid only through frame 11 -> keyframes 4 and 9 valid, rest masked
    assert data["fut_motion_mask"][1].numpy().tolist() == [1, 1] + [0] * 10


@needs_af
def test_upsample_keyframes_is_exact_linear_interpolation():
    import torch
    anchor = torch.tensor([[0.0, 0.0], [10.0, 5.0]])               # (M=2, 2)
    steps = torch.arange(5, 61, 5, dtype=torch.float32)             # keyframe times
    kf = torch.stack([torch.stack([steps, 2 * steps], -1),          # straight lines
                      torch.stack([10 + steps, 5 - steps], -1)])    # (2, 12, 2)
    up = dda.upsample_keyframes(kf, anchor, 5)
    assert tuple(up.shape) == (2, 60, 2)
    t = torch.arange(1, 61, dtype=torch.float32)
    assert torch.allclose(up[0], torch.stack([t, 2 * t], -1))       # exact on a line
    assert torch.allclose(up[1], torch.stack([10 + t, 5 - t], -1))
    assert torch.allclose(up[:, 4::5], kf)                           # keyframes reproduced
    # sample axis broadcasts against a per-agent anchor
    up_k = dda.upsample_keyframes(kf[:, None].expand(2, 3, 12, 2), anchor[:, None, :], 5)
    assert tuple(up_k.shape) == (2, 3, 60, 2) and torch.allclose(up_k[:, 1], up)


@needs_af
def test_predict_seq_with_stride_returns_full_rate():
    import torch
    torch.manual_seed(0)
    cfg = dda.build_map_free_cfg(4, 12, nz=8, sample_k=2, tf_model_dim=32, tf_ff_dim=64,
                                 tf_nhead=4, nlayer=1, out_mlp_dim=None, rand_rot_scene=False)
    model = model_dict["agentformer"](cfg)
    model.set_device(torch.device("cpu")); model.eval()
    obs, pred, fm = _synth_scene(M=3)
    data = dda.seq_to_af_data(obs, pred, fm, cfg.traj_scale, stride=5)
    ml, samp = dda.predict_seq(model, data, nz=cfg.nz, sample_k=4,
                               traj_scale=cfg.traj_scale, stride=5)
    assert tuple(ml.shape) == (3, PRED, 2) and tuple(samp.shape) == (3, 4, PRED, 2)


# --------------------------------------------------------------------------
# forward / inference
# --------------------------------------------------------------------------
@needs_af
def test_predict_seq_shapes():
    import torch
    model, cfg = _tiny_model()
    obs, pred, fm = _synth_scene(M=4)
    data = dda.seq_to_af_data(obs, pred, fm, cfg.traj_scale)
    ml, samp = dda.predict_seq(model, data, nz=cfg.nz, sample_k=5, traj_scale=cfg.traj_scale)
    assert tuple(ml.shape) == (4, PRED, 2)
    assert tuple(samp.shape) == (4, 5, PRED, 2)
    assert torch.isfinite(ml).all() and torch.isfinite(samp).all()


@needs_af
def test_most_likely_is_deterministic():
    """z=0 rollout is deterministic — two calls give identical predictions."""
    import torch
    model, cfg = _tiny_model()
    obs, pred, fm = _synth_scene(M=3)
    data = dda.seq_to_af_data(obs, pred, fm, cfg.traj_scale)
    a, _ = dda.predict_seq(model, data, nz=cfg.nz, sample_k=2, traj_scale=cfg.traj_scale)
    b, _ = dda.predict_seq(model, data, nz=cfg.nz, sample_k=2, traj_scale=cfg.traj_scale)
    assert torch.allclose(a, b, atol=1e-6)


@needs_af
def test_bestofk_is_oracle_min():
    """Oracle best-of-K ADE must be <= the ADE of any single one of the K samples."""
    import torch
    from dd_data import per_agent_errors, per_agent_errors_bestofk

    model, cfg = _tiny_model()
    obs, pred, fm = _synth_scene(M=4, seed=1)
    data = dda.seq_to_af_data(obs, pred, fm, cfg.traj_scale)
    _, samp = dda.predict_seq(model, data, nz=cfg.nz, sample_k=6, traj_scale=cfg.traj_scale)

    gt = torch.from_numpy(pred).float().unsqueeze(0)
    am = torch.ones(1, 4, dtype=torch.bool)
    fmt = torch.from_numpy(fm).unsqueeze(0)
    ade_k, _, _ = per_agent_errors_bestofk(samp.permute(0, 1, 3, 2).unsqueeze(0), gt, am, fmt)
    ade_0, _, _ = per_agent_errors(samp[:, 0].permute(0, 2, 1).unsqueeze(0), gt, am, fmt)
    assert float(ade_k.mean()) <= float(ade_0.mean()) + 1e-5


# --------------------------------------------------------------------------
# eval-population contract
# --------------------------------------------------------------------------
@needs_canonical
@needs_af
def test_scene_sizes_partition_canonical_population():
    """The per-split scene sizes sum to the canonical agent counts, so iterating
    every sequence scores exactly the canonical val/test population."""
    from dd_data import CanonicalSeqDataset

    pops = canonical_split_populations()
    for split in ("val", "test"):
        ds = CanonicalSeqDataset(str(CANONICAL), split, str(SPLITS_FILE), rot_aug=False, seed=0)
        assert int(m6.scene_sizes(ds).sum()) == pops[split]["agents"]


@needs_canonical
@needs_af
def test_evaluate_scores_every_agent_it_is_given():
    """On a handful of scenes, evaluate() reports exactly the number of agents in
    those scenes (no survivor filter, no drop/dup) — the per-scene invariant the
    full-population contract is built from."""
    import torch
    from dd_data import CanonicalSeqDataset

    model, cfg = _tiny_model()
    ds = CanonicalSeqDataset(str(CANONICAL), "val", str(SPLITS_FILE), rot_aug=False, seed=0)
    idx = m6.pick_indices(ds, limit=4, smallest=True)      # cheap: smallest scenes
    expected = int(m6.scene_sizes(ds)[idx].sum())
    out = m6.evaluate(model, ds, cfg, eval_k=2, device=torch.device("cpu"), indices=idx)
    assert out["most_likely"]["n_agents"] == expected
    assert out["best_of_k"]["n_agents"] == expected
