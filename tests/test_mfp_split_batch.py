"""MFP (the K-mode decoder) --max-batch-slots: a batch over the cap runs as parts of whole sequences with
accumulated gradients, each part's loss weighted by its share of the batch, so training is the same as on
the whole batch (one optimizer step per batch, the same loss). Sequences of 1-5 agents with partly missing
futures, so the parts carry unequal shares; dropout off, so the two runs draw the same randomness."""

import json
import sys

import numpy as np
import pytest

from conftest import load_script

torch = pytest.importorskip("torch")
TO, TF = 20, 60
SIZES = [1, 5, 2, 4, 3, 2, 3, 2]                                       # s0-s5 train, s6 val, s7 test


def _data(tmp_path):
    rng = np.random.default_rng(1)
    n = sum(SIZES)
    start = rng.uniform(-10, 10, (n, 2, 1)); vel = rng.uniform(-1, 1, (n, 2, 1))
    obs = (start + vel * np.arange(TO)).astype(np.float32)
    pred = (obs[:, :, -1:] + vel * np.arange(1, TF + 1) + rng.normal(0, 0.3, (n, 2, TF))).astype(np.float32)
    fm = np.ones((n, TF), bool)
    for i in range(0, n, 3):
        fm[i, rng.integers(10, TF):] = False                              # some futures end early
    se = np.cumsum([0] + SIZES)
    np.savez(tmp_path / "d.npz", obs_traj=obs, pred_traj=pred, obs_traj_rel=np.diff(obs, axis=2, prepend=obs[:, :, :1]),
             pred_traj_rel=np.diff(pred, axis=2, prepend=obs[:, :, -1:]), fut_mask=fm,
             agent_id=np.arange(n, dtype=np.int32), agent_type=np.asarray(["Car"] * n, dtype=object),
             seq_start_end=np.stack([se[:-1], se[1:]], 1).astype(np.int64),
             scene_id=np.asarray([f"s{i}" for i in range(len(SIZES))], dtype=object),
             t0=np.zeros(len(SIZES), np.int32))
    (tmp_path / "s.json").write_text(json.dumps({"splits": {"train": [f"s{i}" for i in range(6)], "val": ["s6"],
                                                            "test": ["s7"]}}))


def _run(mod, tmp_path, monkeypatch, tag, extra):
    monkeypatch.setattr(sys, "argv", ["model4_mfp_train_eval.py", "--device", "cpu", "--seed", "0",
                                      "--data", str(tmp_path / "d.npz"), "--splits-file", str(tmp_path / "s.json"),
                                      "--epochs", "3", "--batch-size", "6", "--k", "3", "--dropout", "0",
                                      "--hidden-dim", "32", "--emb-dim", "16", "--attn-dim", "16", "--keep", "last",
                                      "--save-ckpt", str(tmp_path / f"{tag}.pt"),
                                      "--save-metrics-json", str(tmp_path / f"{tag}.json"), *extra])
    assert mod.main() == 0
    return (json.loads((tmp_path / f"{tag}.json").read_text()),
            torch.load(tmp_path / f"{tag}.pt", map_location="cpu", weights_only=False)["model"])


def test_split_batch_keeps_whole_sequences_in_order_under_the_cap():
    mod = load_script("model4_mfp_train_eval.py")
    from dd_data import collate_pad
    items = []
    for m in (2, 5, 1, 3):
        o = np.random.default_rng(m).normal(size=(m, 2, TO)).astype(np.float32)
        p = np.random.default_rng(m + 9).normal(size=(m, 2, TF)).astype(np.float32)
        items.append((o, p, o, p, np.ones((m, TF), bool), np.asarray(["Car"] * m, dtype=object)))
    b = collate_pad(items)                                                 # 4 sequences x 5 slots = 20
    assert mod.split_batch(b, 0) == [b] and mod.split_batch(b, 20) == [b]
    parts = mod.split_batch(b, 6)
    assert [tuple(p.agent_mask.shape) for p in parts] == [(1, 2), (1, 5), (2, 3)]
    assert torch.equal(torch.cat([p.agent_mask.sum(1) for p in parts]), b.agent_mask.sum(1))
    row = 0
    for p in parts:                                                        # the same agents, padding trimmed
        for r in range(p.agent_mask.shape[0]):
            m = int(p.agent_mask[r].sum())
            assert torch.equal(p.obs_rel[r, :m], b.obs_rel[row, :m]) and torch.equal(p.pred_rel[r, :m], b.pred_rel[row, :m])
            assert torch.equal(p.fut_mask[r, :m], b.fut_mask[row, :m])
            row += 1


@pytest.mark.parametrize("target", ["hard", "soft"])
def test_split_training_equals_whole_batch_training(target, tmp_path, monkeypatch):
    """Weights compared under SGD: AdamW turns float noise in a zero gradient (the mode scorer's last bias,
    shared by every mode, so the softmax ignores it) into lr-sized steps, which would hide the comparison."""
    _data(tmp_path)
    mod = load_script("model4_mfp_train_eval.py")
    monkeypatch.setattr(torch.optim, "AdamW", lambda params, lr, weight_decay: torch.optim.SGD(params, lr=lr))
    whole, w0 = _run(mod, tmp_path, monkeypatch, "whole", ["--mode-target", target, "--lr", "0.05"])
    split, w1 = _run(mod, tmp_path, monkeypatch, "split", ["--mode-target", target, "--lr", "0.05",
                                                           "--max-batch-slots", "6"])
    assert [(e["steps"], e["agent_windows"]) for e in split["history"]] == \
        [(e["steps"], e["agent_windows"]) for e in whole["history"]]
    assert split["hyper"]["max_batch_slots"] == 6 and whole["hyper"]["max_batch_slots"] == 0
    for k in w0:
        assert torch.allclose(w0[k], w1[k], atol=1e-5, rtol=1e-4), k
    assert split["val"]["top1"]["ade"] == pytest.approx(whole["val"]["top1"]["ade"], abs=1e-5)
    # the cap actually split: with it at the whole batch's size, nothing changes bit for bit
    same, w2 = _run(mod, tmp_path, monkeypatch, "same", ["--mode-target", target, "--lr", "0.05",
                                                         "--max-batch-slots", "30"])
    assert all(torch.equal(w0[k], w2[k]) for k in w0)
