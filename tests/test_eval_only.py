"""--eval-only (cross-domain checkpoint evaluation): evaluating a saved checkpoint on the data it
was trained with reproduces the training run's metrics exactly, and the flat prediction dump holds
one row per evaluated agent, val then test, in canonical order."""

import json
import sys

import numpy as np
import pytest

from conftest import load_script

_err = None
try:
    import torch  # noqa: F401
except Exception as e:  # pragma: no cover - environment-dependent
    _err = repr(e)
needs_torch = pytest.mark.skipif(_err is not None, reason=f"torch unavailable: {_err}")

TO, TF = 20, 60
SIZES = [3, 4, 2, 3, 4, 3]                                              # scenes s0-s3 train, s4 val, s5 test


def _data(tmp_path):
    """Six sequences moving at constant velocity; some futures end early (fut_mask)."""
    rng = np.random.default_rng(0)
    n = sum(SIZES)
    start = rng.uniform(-10, 10, (n, 2, 1))
    vel = rng.uniform(-1, 1, (n, 2, 1))
    obs = (start + vel * np.arange(TO)).astype(np.float32)
    pred = (obs[:, :, -1:] + vel * np.arange(1, TF + 1)).astype(np.float32)
    fut_mask = np.ones((n, TF), bool)
    fut_mask[::4, 45:] = False
    pred[~np.repeat(fut_mask[:, None], 2, 1)] = 0.0                     # masked steps hold 0, as in the real file
    se = np.cumsum([0] + SIZES)
    np.savez(tmp_path / "d.npz", obs_traj=obs, pred_traj=pred, obs_traj_rel=np.diff(obs, axis=2, prepend=obs[:, :, :1]),
             pred_traj_rel=np.diff(pred, axis=2, prepend=obs[:, :, -1:]), fut_mask=fut_mask,
             agent_id=np.arange(n, dtype=np.int32), agent_type=np.asarray((["Car", "Pedestrian"] * n)[:n], dtype=object),
             seq_start_end=np.stack([se[:-1], se[1:]], 1).astype(np.int64),
             scene_id=np.asarray([f"s{i}" for i in range(6)], dtype=object), t0=np.zeros(6, np.int32))
    (tmp_path / "s.json").write_text(json.dumps({"splits": {"train": ["s0", "s1", "s2", "s3"], "val": ["s4"],
                                                            "test": ["s5"]}}))
    return fut_mask


def _run(mod, script, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", [script, *argv])
    assert mod.main() == 0


@needs_torch
def test_mfp_eval_only_reproduces_training_and_dumps_every_mode(monkeypatch, tmp_path):
    fut_mask = _data(tmp_path)
    script = "model4_mfp_train_eval.py"
    mod = load_script(script)
    common = ["--device", "cpu", "--seed", "0", "--data", str(tmp_path / "d.npz"), "--splits-file", str(tmp_path / "s.json"),
              "--save-ckpt", str(tmp_path / "m.pt")]
    _run(mod, script, monkeypatch, common + ["--epochs", "2", "--k", "3", "--emb-dim", "8", "--hidden-dim", "16",
                                              "--attn-dim", "8", "--no-amp", "--select", "top1",
                                              "--save-metrics-json", str(tmp_path / "train.json")])
    _run(mod, script, monkeypatch, common + ["--eval-only", "--no-amp", "--save-metrics-json", str(tmp_path / "eval.json"),
                                              "--save-pred-npz", str(tmp_path / "p.npz")])
    train, ev = (json.loads((tmp_path / f).read_text()) for f in ("train.json", "eval.json"))
    assert ev["eval_only"] and ev["val"] == train["val"] and ev["test"] == train["test"]
    assert ev["best_epoch"] == train["best_epoch"]

    p = np.load(tmp_path / "p.npz", allow_pickle=True)
    rows = slice(sum(SIZES[:4]), sum(SIZES))                            # the val then test agents, in order
    assert list(p["split"]) == ["val"] * SIZES[4] + ["test"] * SIZES[5]
    assert list(p["scene_id"]) == ["s4"] * SIZES[4] + ["s5"] * SIZES[5]
    assert p["modes"].shape == (SIZES[4] + SIZES[5], 3, TF, 2) and p["mode_logits"].shape == (SIZES[4] + SIZES[5], 3)
    top = p["mode_logits"].argmax(1)
    assert np.allclose(p["pred"], p["modes"][np.arange(len(top)), top], atol=1e-5)    # top-1 = best-scored mode
    assert np.array_equal(p["fut_mask"], fut_mask[rows])


@needs_torch
@pytest.mark.parametrize("name,script", [("stgat", "model3_stgat_train_eval.py"),
                                         ("transformer", "model5_transformer_train_eval.py")])
def test_dump_predictions_scores_a_checkpoint_as_its_training_run_did(name, script, monkeypatch, tmp_path):
    """scripts/dump_predictions.py --ckpt --metrics-json is the evaluate-only path for ST-GAT and the
    transformer: on the training data it reproduces the run's own val/test metrics."""
    fut_mask = _data(tmp_path)
    data = ["--data", str(tmp_path / "d.npz"), "--splits-file", str(tmp_path / "s.json")]
    _run(load_script(script), script, monkeypatch, ["--device", "cpu", "--seed", "0", "--epochs", "2", "--no-amp", *data,
                                                    "--save-ckpt", str(tmp_path / "m.pt"),
                                                    "--save-metrics-json", str(tmp_path / "train.json")])
    _run(load_script("dump_predictions.py"), "dump_predictions.py", monkeypatch,
         ["--device", "cpu", "--models", name, "--ckpt", str(tmp_path / "m.pt"), *data, "--out-dir", str(tmp_path),
          "--out-name", "p.npz", "--metrics-json", str(tmp_path / "eval.json")])
    train, ev = (json.loads((tmp_path / f).read_text()) for f in ("train.json", "eval.json"))
    for split in ("val", "test"):
        assert ev[split]["n_agents"] == train[split]["n_agents"]
        assert ev[split]["ade"] == pytest.approx(train[split]["ade"], abs=1e-4)
        assert ev[split]["fde"] == pytest.approx(train[split]["fde"], abs=1e-4)
    p = np.load(tmp_path / "p.npz", allow_pickle=True)
    assert list(p["scene_id"]) == ["s4"] * SIZES[4] + ["s5"] * SIZES[5]
    assert np.array_equal(p["fut_mask"], fut_mask[sum(SIZES[:4]):])


def _windows(tmp_path):
    """The per-agent windows format of scripts/9 (obs (N,20,2), fut (N,60,2)) for CV and the MLP."""
    rng = np.random.default_rng(1)
    n = 40
    vel = rng.uniform(-1, 1, (n, 1, 2))
    obs = (vel * np.arange(-TO + 1, 1)[None, :, None]).astype(np.float32)   # centred on the last observed frame
    fut = (vel * np.arange(1, TF + 1)[None, :, None]).astype(np.float32)
    fm = np.ones((n, TF), bool)
    fm[::5, 50:] = False
    scene = np.asarray([f"s{i % 6}" for i in range(n)], dtype=object)
    np.savez(tmp_path / "w.npz", obs=obs, fut=fut, fut_mask=fm, scene_id=scene,
             agent_type=np.asarray(["Car"] * n, dtype=object))
    (tmp_path / "s.json").write_text(json.dumps({"splits": {"train": ["s0", "s1", "s2", "s3"], "val": ["s4"],
                                                            "test": ["s5"]}}))
    return scene, fm


def test_mlp_eval_only_reproduces_training_and_cv_dumps_its_predictions(monkeypatch, tmp_path):
    scene, fm = _windows(tmp_path)
    s2, s1 = "model2_baseline_mlp_train_eval.py", "model1_baseline_cv_eval.py"
    mlp, cv = load_script(s2), load_script(s1)
    common = ["--input", str(tmp_path / "w.npz"), "--splits-file", str(tmp_path / "s.json"), "--hidden-dim", "16"]
    _run(mlp, s2, monkeypatch, common + ["--epochs", "3", "--save-best", str(tmp_path / "best.npz"),
                                         "--save-metrics-json", str(tmp_path / "train.json")])
    before = (tmp_path / "best.npz").read_bytes()
    _run(mlp, s2, monkeypatch, common + ["--eval-only", "--init-ckpt", str(tmp_path / "best.npz"),
                                         "--save-best", str(tmp_path / "best.npz"),
                                         "--save-metrics-json", str(tmp_path / "eval.json"),
                                         "--save-pred-npz", str(tmp_path / "p.npz")])
    assert (tmp_path / "best.npz").read_bytes() == before                # evaluate-only never overwrites
    train, ev = (json.loads((tmp_path / f).read_text()) for f in ("train.json", "eval.json"))
    assert ev["eval_only"] and ev["val"] == train["val"] and ev["test"] == train["test"]
    rows = np.concatenate([np.flatnonzero(scene == "s4"), np.flatnonzero(scene == "s5")])
    p = np.load(tmp_path / "p.npz", allow_pickle=True)
    assert list(p["scene_id"]) == list(scene[rows]) and np.array_equal(p["fut_mask"], fm[rows])

    _run(cv, s1, monkeypatch, ["--input-npz", str(tmp_path / "w.npz"), "--splits-file", str(tmp_path / "s.json"),
                               "--save-metrics-json", str(tmp_path / "cv.json"), "--save-pred-npz", str(tmp_path / "cv.npz")])
    c = np.load(tmp_path / "cv.npz", allow_pickle=True)
    w = np.load(tmp_path / "w.npz", allow_pickle=True)
    assert np.allclose(c["pred"], cv.compute_cv_predictions(w["obs"][rows], TF, 5), atol=1e-5)
    assert np.allclose(c["pred"], w["fut"][rows], atol=1e-4)            # exact for constant velocity
