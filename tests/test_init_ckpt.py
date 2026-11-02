"""--init-ckpt (fine-tuning) in ST-GAT, the map-free transformer and LED: a run started from a
checkpoint with a zero learning rate saves exactly the checkpoint's weights, so the weights
were loaded, and its metrics record where they came from. (The MLP and AutoBot have their own
tests: test_mlp_init_ckpt.py, test_autobot.py.)"""

import json
import sys

import numpy as np
import pytest

from conftest import load_script

_err = None
try:
    import torch
except Exception as e:  # pragma: no cover - environment-dependent
    _err = repr(e)
needs_torch = pytest.mark.skipif(_err is not None, reason=f"torch unavailable: {_err}")

TO, TF = 20, 60
MODELS = {
    "stgat": ("model3_stgat_train_eval.py", ["--epochs", "1"], ["--lr", "0"], ("model",)),
    "transformer": ("model5_transformer_train_eval.py", ["--epochs", "1"], ["--lr", "0"], ("model",)),
    "led": ("model7_led_train_eval.py", ["--core-epochs", "1", "--init-epochs", "1", "--k", "4"],
            ["--core-lr", "0", "--init-lr", "0"], ("core", "init")),
}


def _data(tmp_path):
    """Six 3-agent sequences moving at constant velocity; scenes s0-s3 train, s4 val, s5 test."""
    rng = np.random.default_rng(0)
    n, sizes = 18, [3] * 6
    start = rng.uniform(-10, 10, (n, 2, 1)); vel = rng.uniform(-1, 1, (n, 2, 1))
    obs = (start + vel * np.arange(TO)).astype(np.float32)
    pred = (obs[:, :, -1:] + vel * np.arange(1, TF + 1)).astype(np.float32)
    se = np.cumsum([0] + sizes)
    np.savez(tmp_path / "d.npz", obs_traj=obs, pred_traj=pred, obs_traj_rel=np.diff(obs, axis=2, prepend=obs[:, :, :1]),
             pred_traj_rel=np.diff(pred, axis=2, prepend=obs[:, :, -1:]), fut_mask=np.ones((n, TF), bool),
             agent_id=np.arange(n, dtype=np.int32), agent_type=np.asarray(["Car"] * n, dtype=object),
             seq_start_end=np.stack([se[:-1], se[1:]], 1).astype(np.int64),
             scene_id=np.asarray([f"s{i}" for i in range(6)], dtype=object), t0=np.zeros(6, np.int32))
    (tmp_path / "s.json").write_text(json.dumps({"splits": {"train": ["s0", "s1", "s2", "s3"], "val": ["s4"],
                                                            "test": ["s5"]}}))


@needs_torch
@pytest.mark.parametrize("name", list(MODELS))
def test_init_ckpt_loads_the_weights(name, monkeypatch, tmp_path):
    script, epochs, zero_lr, keys = MODELS[name]
    _data(tmp_path)
    mod = load_script(script)

    def run(tag, extra):
        monkeypatch.setattr(sys, "argv", [script, "--device", "cpu", "--seed", "0", "--data", str(tmp_path / "d.npz"),
                                          "--splits-file", str(tmp_path / "s.json"), *epochs,
                                          "--save-ckpt", str(tmp_path / f"{tag}.pt"),
                                          "--save-metrics-json", str(tmp_path / f"{tag}.json"), *extra])
        assert mod.main() == 0
        return torch.load(tmp_path / f"{tag}.pt", map_location="cpu", weights_only=False)

    pre = run("pre", [])
    ft = run("ft", ["--init-ckpt", str(tmp_path / "pre.pt"), *zero_lr])
    fresh = run("fresh", zero_lr)                                          # control: same seed, nothing loaded
    for key in keys:
        for k, v in pre[key].items():
            assert torch.equal(ft[key][k], v), (name, key, k)
        assert any(not torch.equal(fresh[key][k], v) for k, v in pre[key].items()), (name, key)
    metrics = json.loads((tmp_path / "ft.json").read_text())
    assert metrics["init_ckpt"] == str(tmp_path / "pre.pt") and "init_ckpt" not in metrics["hyper"]
