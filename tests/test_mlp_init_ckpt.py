"""MLP --init-ckpt (fine-tuning, scripts/model2_baseline_mlp_train_eval.py): the weights and the
standardizers come from the checkpoint, not from the new train split; without it the
standardizers are fit on the train split as before."""

import json
import sys

import numpy as np

from conftest import load_script

m2 = load_script("model2_baseline_mlp_train_eval.py")
TO, TF = 4, 6


def _windows(path):
    """Per-agent windows for scenes a..e; scene e moves 10x faster (different statistics)."""
    rng = np.random.default_rng(0)
    obs, fut, sid = [], [], []
    for s, speed in zip("abcde", (1, 1, 1, 1, 10)):
        for _ in range(20):
            v = rng.normal(0, speed, 2)
            obs.append(v * np.arange(-TO + 1, 1)[:, None]); fut.append(v * np.arange(1, TF + 1)[:, None]); sid.append(s)
    n = len(sid)
    np.savez(path, obs=np.asarray(obs, np.float32), fut=np.asarray(fut, np.float32), fut_mask=np.ones((n, TF), bool),
             scene_id=np.asarray(sid, dtype=object), agent_type=np.asarray(["Car"] * n, dtype=object))


def _run(monkeypatch, tmp_path, tag, train, extra=()):
    sp = tmp_path / f"{tag}.json"
    sp.write_text(json.dumps({"splits": {"train": train, "val": ["c"], "test": ["d"]}}))
    out = tmp_path / f"{tag}.npz"
    monkeypatch.setattr(sys, "argv", ["m2", "--input", str(tmp_path / "w.npz"), "--splits-file", str(sp),
                                      "--save-best", str(out), "--save-metrics-json", str(tmp_path / f"{tag}.json.out"),
                                      "--epochs", "3", "--hidden-dim", "8", *extra])
    assert m2.main() == 0
    return np.load(out)


def test_init_ckpt_keeps_the_pretrained_weights_and_standardizers(monkeypatch, tmp_path):
    _windows(tmp_path / "w.npz")
    pre = _run(monkeypatch, tmp_path, "pre", ["a", "b"])
    ft = _run(monkeypatch, tmp_path, "ft", ["e"], ("--init-ckpt", str(tmp_path / "pre.npz"), "--lr", "0"))
    for k in pre.files:                                     # lr 0: nothing moves, nothing is refit on scene e
        assert np.array_equal(ft[k], pre[k]), k
    metrics = json.loads((tmp_path / "ft.json.out").read_text())
    assert metrics["init_ckpt"] == str(tmp_path / "pre.npz")


def test_without_init_ckpt_the_standardizers_are_fit_on_the_train_split(monkeypatch, tmp_path):
    _windows(tmp_path / "w.npz")
    d = np.load(tmp_path / "w.npz", allow_pickle=True)
    e = d["scene_id"] == "e"
    got = _run(monkeypatch, tmp_path, "e", ["e"])
    assert np.allclose(got["x_mean"], d["obs"][e].reshape(e.sum(), -1).mean(0, keepdims=True), atol=1e-5)
    assert np.allclose(got["y_std"], d["fut"][e].reshape(e.sum(), -1).std(0, keepdims=True), atol=1e-4)
