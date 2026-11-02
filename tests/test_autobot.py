"""Contract + unit tests for the upstream AutoBot-Joint integration (model9).

* Batching: a batch holds sequences of one size, so no padded agent reaches the
  model. Upstream sums the mode-probability logits over every agent slot, so
  padding would change a scene's mode probabilities (shown below).
* Batch independence: a scene's predictions do not depend on the rest of its batch.
* Loss masking: future steps outside fut_mask do not change the loss.
* Frame: scene-frame outputs map back to world positions.
* Eval-population contract: run_split scores every agent it is handed, once, and
  dumps them in sequence order.
* Import hygiene: loading AutoBot leaves no `models` package behind (LED has one).

Skips cleanly if the AutoBots repo is not cloned into third_party/autobots.
"""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import load_script

_ab_err = None
try:
    import torch
    m9 = load_script("model9_autobot_train_eval.py")
    AutoBotJoint = m9.load_autobot_joint()
    from dd_data import CanonicalSeqDataset, collate_pad
except Exception as e:  # pragma: no cover - environment-dependent
    _ab_err = repr(e)
needs_ab = pytest.mark.skipif(_ab_err is not None, reason=f"AutoBot stack unavailable: {_ab_err}")

OBS, PRED, K = 8, 12, 3


def _model():
    torch.manual_seed(0)
    return AutoBotJoint(k_attr=2, d_k=32, _M=1, c=K, T=PRED, L_enc=1, dropout=0.0, num_heads=4, L_dec=1,
                        tx_hidden_size=64, use_map_lanes=False, map_attr=0, num_agent_types=1).eval()


def _seq(m, seed, fut_steps=None):
    """One canonical sequence item of m agents (dd_data item layout)."""
    rng = np.random.default_rng(seed)
    start = rng.uniform(-20, 20, (m, 2, 1)).astype(np.float32)
    vel = rng.uniform(-1, 1, (m, 2, 1)).astype(np.float32)
    obs = start + vel * np.arange(OBS, dtype=np.float32)
    pred = obs[:, :, -1:] + vel * np.arange(1, PRED + 1, dtype=np.float32)
    fm = np.ones((m, PRED), bool)
    if fut_steps is not None:
        fm[:, fut_steps:] = False
        pred[:, :, fut_steps:] = 0.0
    return obs, pred, np.zeros_like(obs), np.zeros_like(pred), fm, np.asarray(["Car"] * m, dtype=object)


@needs_ab
def test_batches_hold_one_size_and_respect_the_caps():
    sizes = np.asarray([3, 5, 3, 3, 5, 2, 3])
    out = list(m9.batches(sizes, range(len(sizes)), batch_seqs=2, max_slots=9))
    assert sorted(i for b in out for i in b) == list(range(len(sizes)))
    for b in out:
        assert len({int(sizes[i]) for i in b}) == 1
        assert len(b) <= 2 and len(b) * sizes[b[0]] <= 9 or len(b) == 1
    assert [b for b in out if sizes[b[0]] == 3] == [[0, 2], [3, 6]]       # order kept within a size


@needs_ab
def test_padding_would_change_the_mode_probabilities():
    """Why batches hold one size: upstream's mode logits sum over every agent slot."""
    model = _model()
    alone = collate_pad([_seq(3, 1)])
    padded = collate_pad([_seq(3, 1), _seq(5, 2)])                          # sequence 0 padded to 5 slots
    with torch.no_grad():
        p_alone = m9.forward(model, alone, 1.0)[1][0]
        p_padded = m9.forward(model, padded, 1.0)[1][0]
    assert not torch.allclose(p_alone, p_padded, atol=1e-4)


@needs_ab
def test_a_scene_does_not_depend_on_the_rest_of_its_batch():
    model = _model()
    with torch.no_grad():
        pa, qa, *_ = m9.forward(model, collate_pad([_seq(4, 1)]), 1.0)
        pb, qb, *_ = m9.forward(model, collate_pad([_seq(4, 1), _seq(4, 2), _seq(4, 3)]), 1.0)
    assert torch.allclose(pa[:, :, 0], pb[:, :, 0], atol=1e-5) and torch.allclose(qa[0], qb[0], atol=1e-5)


@needs_ab
def test_loss_ignores_future_steps_outside_the_mask():
    model = _model()
    b = collate_pad([_seq(3, 1, fut_steps=7), _seq(3, 2)])
    pred, probs, fut, fut_m, _ = m9.forward(model, b, 1.0)

    def loss(f):
        return m9.autobot_joint_loss(pred, probs, f, fut_m, b.agent_mask, 40.0, 20.0)[0].item()
    base = loss(fut)
    hidden = fut.clone(); hidden[0, 7:] += 50.0                             # steps outside sequence 0's mask
    shown = fut.clone(); shown[0, 3] += 50.0                                # a valid step
    assert loss(hidden) == pytest.approx(base, rel=1e-6)
    assert loss(shown) != pytest.approx(base, rel=1e-3)


@needs_ab
def test_scene_frame_maps_back_to_world_positions():
    b = collate_pad([_seq(3, 1), _seq(3, 2)])
    scale = 10.0
    _, _, fut, fut_m, origin = m9.scene_inputs(b, scale)
    assert torch.allclose(origin, b.obs_abs[:, :, :, -1].mean(1))
    pred = torch.zeros((K, PRED, 2, 3, 5))
    pred[..., :2] = fut.permute(1, 0, 2, 3).unsqueeze(0)                    # every mode = the ground truth
    world = m9.world_modes(pred, origin, scale)                             # (B, M, K, 2, T)
    assert torch.allclose(world[:, :, 1], b.pred_abs, atol=1e-4)


@needs_ab
def test_run_split_scores_every_agent_once_in_sequence_order(tmp_path):
    seqs = [_seq(3, 1), _seq(2, 2), _seq(3, 3, fut_steps=5), _seq(4, 4)]
    n = np.cumsum([0] + [s[0].shape[0] for s in seqs])
    np.savez(tmp_path / "d.npz", obs_traj=np.concatenate([s[0] for s in seqs]),
             pred_traj=np.concatenate([s[1] for s in seqs]), obs_traj_rel=np.concatenate([s[2] for s in seqs]),
             pred_traj_rel=np.concatenate([s[3] for s in seqs]), fut_mask=np.concatenate([s[4] for s in seqs]),
             agent_type=np.concatenate([s[5] for s in seqs]), seq_start_end=np.stack([n[:-1], n[1:]], 1),
             scene_id=np.asarray(["s0", "s1", "s2", "s3"], dtype=object))
    (tmp_path / "split.json").write_text(json.dumps({"splits": {"train": [], "val": ["s0", "s1", "s2", "s3"],
                                                                "test": []}}))
    ds = CanonicalSeqDataset(tmp_path / "d.npz", "val", tmp_path / "split.json")
    args = SimpleNamespace(batch_seqs=64, max_slots=2048, pos_scale=1.0)
    summary, dump = m9.run_split(_model(), ds, m9.seq_sizes(ds), args, torch.device("cpu"), collect=True)
    assert summary["most_likely"]["n_agents"] == summary["best_of_k"]["n_agents"] == n[-1]
    assert list(dump["scene_id"]) == ["s0"] * 3 + ["s1"] * 2 + ["s2"] * 3 + ["s3"] * 4
    assert np.array_equal(dump["gt"], np.transpose(np.concatenate([s[1] for s in seqs]), (0, 2, 1)))
    assert dump["samples"].shape == (n[-1], K, PRED, 2) and np.allclose(dump["probs"].sum(1), 1.0, atol=1e-5)
    best = dump["probs"].argmax(1)
    assert np.allclose(dump["pred"], dump["samples"][np.arange(n[-1]), best])


@needs_ab
def test_loading_autobot_leaves_no_models_package_behind():
    m9.load_autobot_joint()
    assert not any(k == "models" or k.startswith("models.") for k in sys.modules if "autobots" in
                   str(getattr(sys.modules[k], "__file__", "") or ""))
