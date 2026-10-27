"""Contract + unit tests for the upstream-LED integration (model7).

* Adapter contract: scripts/dd_led.build_batch turns canonical sequences into one
  LED batch -- 6-dim past features, futures relative to each agent's last
  observation, and a block-diagonal mask so agents only attend within their scene.
* Shape generalization: the upstream modules hard-code NBA shapes (10/20 frames,
  K = 10 + 10); the adapter's core runs at IDD-3D's 20/60 frames for any K.
* Loss fidelity: the uncertainty term pairs each agent with its own error (the
  upstream (N,1) x (N,) broadcast is not reproduced).
* Eval-population contract: run_split scores every agent it is handed, once.

Skips cleanly if the LED repo is not cloned into third_party/led.
"""

import numpy as np
import pytest

from conftest import load_script

_led_err = None
try:
    import torch
    import dd_led
    dd_led.add_led_to_path()
    m7 = load_script("model7_led_train_eval.py")
except Exception as e:  # pragma: no cover - environment-dependent
    _led_err = repr(e)
needs_led = pytest.mark.skipif(_led_err is not None, reason=f"LED stack unavailable: {_led_err}")

OBS, PRED = 20, 60


def _seq(M, seed, offset=0.0):
    g = np.random.default_rng(seed)
    obs = (np.cumsum(g.normal(size=(M, 2, OBS)), axis=-1) + offset).astype(np.float32)
    pred = (obs[..., -1:] + np.cumsum(g.normal(size=(M, 2, PRED)), axis=-1)).astype(np.float32)
    fm = np.ones((M, PRED), dtype=bool)
    at = np.array(["Car"] * M, dtype=object)
    return obs, pred, fm, at


@needs_led
def test_build_batch_features_and_block_mask():
    items = [_seq(3, 0, offset=100.0), _seq(2, 1, offset=-50.0)]
    items[1][2][1, 30:] = False                       # agent 4: future ends at step 29
    b = dd_led.build_batch(items, traj_scale=10.0, device="cpu")
    assert tuple(b["past"].shape) == (5, OBS, 6) and tuple(b["fut"].shape) == (5, PRED, 2)
    m = b["mask"].numpy()
    assert m[:3, :3].all() and m[3:, 3:].all() and not m[:3, 3:].any() and not m[3:, :3].any()
    # rel channel ends at 0 (own last obs); abs channel is scene-centered (mean 0 at t=-1)
    assert np.allclose(b["past"][:, -1, 2:4].numpy(), 0.0, atol=1e-6)
    assert np.allclose(b["past"][:3, -1, :2].numpy().mean(0), 0.0, atol=1e-5)
    # future is relative to own last obs, scaled; masked steps copy-forward filled
    obs, pred, _, _ = items[0]
    assert np.allclose(b["fut"][0, 0].numpy(), (pred[0, :, 0] - obs[0, :, -1]) / 10.0, atol=1e-5)
    assert np.allclose(b["fut"][4, 30:].numpy(), b["fut"][4, 29].numpy())
    assert b["fut_mask"][4].numpy().sum() == 30
    assert np.allclose(b["last"].numpy()[0], obs[0, :, -1])


@needs_led
def test_core_generalizes_shapes_beyond_nba():
    torch.manual_seed(0)
    core, init = dd_led.build_models(OBS, PRED, k=6)
    core.eval(); init.eval()
    b = dd_led.build_batch([_seq(3, 0), _seq(2, 1)], 10.0, "cpu")
    diff = dd_led.Diffusion("cpu")
    with torch.no_grad():
        loc, mean, logvar = dd_led.leapfrog_loc(init, b["past"], b["mask"])
        assert tuple(loc.shape) == (5, 6, PRED, 2) and tuple(logvar.shape) == (5, 1)
        y = diff.denoise(core, b["past"], b["mask"], loc, n_tau=2)
        assert tuple(y.shape) == (5, 6, PRED, 2) and torch.isfinite(y).all()
        loss = diff.noise_estimation_loss(core, b["past"], b["fut"], b["fut_mask"], b["mask"])
        assert torch.isfinite(loss)


@needs_led
def test_uncertainty_loss_pairs_each_agent_with_its_own_error():
    """Swapping two agents' predicted log-variances must change the loss when
    their errors differ (it would not under the upstream outer-product broadcast)."""
    N, K, T = 2, 3, 5
    fut = torch.zeros(N, T, 2)
    gen = torch.zeros(N, K, T, 2)
    gen[1] += 4.0                                     # agent 1 has a large error
    fm = torch.ones(N, T, dtype=torch.bool)
    lv = torch.tensor([[0.0], [2.0]])
    a = dd_led.initializer_loss(gen, fut, fm, lv)[2]
    b = dd_led.initializer_loss(gen, fut, fm, lv.flip(0))[2]
    assert not torch.isclose(a, b)
    # matched (high variance on the high-error agent) is the lower loss
    assert a < b


@needs_led
def test_masked_steps_do_not_enter_the_distance_loss():
    N, K, T = 1, 2, 6
    fut = torch.zeros(N, T, 2)
    gen = torch.zeros(N, K, T, 2)
    gen[:, :, 3:] = 100.0                              # huge error only on masked steps
    fm = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    _, dist, _ = dd_led.initializer_loss(gen, fut, fm, torch.zeros(N, 1))
    assert float(dist) == pytest.approx(0.0)


@needs_led
def test_run_split_scores_every_agent_once():
    class DS:
        """Minimal CanonicalSeqDataset stand-in: 3 sequences, 7 agents."""
        def __init__(self):
            self.items = [_seq(3, 0), _seq(2, 1), _seq(2, 2)]
            self.seq_indices = np.arange(3)
            self.scene_id = np.array(["s0", "s1", "s1"], dtype=object)

        def __len__(self):
            return 3

        def __getitem__(self, i):
            o, p, f, a = self.items[i]
            return o, p, None, None, f, a

    class Args:
        batch_seqs, traj_scale, n_tau = 2, 10.0, 2

    torch.manual_seed(0)
    core, init = dd_led.build_models(OBS, PRED, k=4)
    out, dump = m7.run_split(core, init, dd_led.Diffusion("cpu"), DS(), Args(), "cpu", collect=True)
    assert out["most_likely"]["n_agents"] == 7 == out["best_of_k"]["n_agents"]
    assert np.isfinite(out["most_likely"]["ade"]) and np.isfinite(out["best_of_k"]["ade"])
    assert dump["pred"].shape == (7, PRED, 2)
    assert dump["scene_id"].tolist() == ["s0"] * 3 + ["s1"] * 4
