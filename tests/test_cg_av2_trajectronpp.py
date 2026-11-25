"""Trajectron++ export of the chaos-gap datasets (scripts/cg_av2_trajectronpp.py) and its scoring
path: a node holds exactly an agent's observed frames plus its valid future prefix (a masked step,
stored as 0, never enters a node), placed so the canonical window's last observed frame is the
scene timestep the canonical evaluator predicts from; the test split is the IDD export's pickle,
byte for byte; the evaluator scores the windows of the file given by --canonical; and the
train.py memory-cap patch is idempotent. Tests that build scenes skip without third_party/."""

import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import load_script

cg = load_script("cg_av2_trajectronpp.py")
REPO = Path(__file__).resolve().parent.parent
TRAJECTRON = REPO / "third_party" / "trajectron-plus-plus" / "trajectron"
needs_tpp = pytest.mark.skipif(not TRAJECTRON.is_dir(), reason="Trajectron++ repo not cloned into third_party/")
TO, TF = 20, 60
TPP_TOP = {"environment", "evaluation", "model", "utils", "visualization", "argument_parser"}


@contextlib.contextmanager
def _tpp():
    """Trajectron++'s packages, imported in isolation: LED and AgentFormer in third_party/ also ship
    `model` / `utils` / `visualization`, so theirs are set aside and restored afterwards, and no
    Trajectron++ module is left behind for the other tests."""
    clash = lambda k: k.split(".")[0] in TPP_TOP                       # noqa: E731
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if clash(k)}
    sys.path.insert(0, str(TRAJECTRON))
    try:
        import dill
        import pandas as pd
        from environment import Environment, Node, Scene, derivative_of
        yield SimpleNamespace(dill=dill, pd=pd, Environment=Environment, Scene=Scene, Node=Node,
                              derivative_of=derivative_of)
    finally:
        sys.path.remove(str(TRAJECTRON))
        for k in [k for k in sys.modules if clash(k)]:
            del sys.modules[k]
        sys.modules.update(saved)


def _canonical():
    """Three AV2 sequences (t0 = 30; train, train, val) and one IDD sequence (t0 = 6; test). Agent 1
    ends its future early, agent 4's future has a hole; masked steps hold 0, as in the real file."""
    rng = np.random.default_rng(0)
    sizes, scenes, t0 = [2, 2, 2, 2], ["av2_a", "av2_b", "av2_c", "idd_v"], [30, 30, 30, 6]
    n = sum(sizes)
    start, vel = rng.uniform(-10, 10, (n, 2, 1)), rng.uniform(-2, 2, (n, 2, 1))
    obs = (start + vel * np.arange(TO)).astype(np.float32)
    pred = (obs[:, :, -1:] + vel * np.arange(1, TF + 1)).astype(np.float32)
    fm = np.ones((n, TF), bool)
    fm[1, 35:] = False                                                  # ends early: 35 valid steps
    fm[4, 10:20] = False                                                # a hole: cut after 10 steps
    pred[~np.repeat(fm[:, None], 2, 1)] = 0.0
    se = np.cumsum([0] + sizes)
    return {"obs_traj": obs, "pred_traj": pred, "fut_mask": fm, "agent_id": np.arange(100, 100 + n, dtype=np.int32),
            "agent_type": np.asarray(["Car", "Pedestrian", "Bus", "Car", "Car", "BicycleRider", "Car", "ScooterRider"],
                                     dtype=object),
            "seq_start_end": np.stack([se[:-1], se[1:]], 1).astype(np.int64),
            "scene_id": np.asarray(scenes, dtype=object), "t0": np.asarray(t0, np.int32)}


def test_node_tracks_hold_only_the_observed_frames_and_the_valid_future_prefix():
    d = _canonical()
    tracks, holes = cg.node_tracks(d, 0)
    assert [t[0] for t in tracks] == ["100", "101"] and [t[2] for t in tracks] == ["Car", "Pedestrian"]
    full, early = tracks[0][1], tracks[1][1]
    assert full.shape == (TO + TF, 2) and early.shape == (TO + 35, 2) and holes == 0
    assert np.array_equal(full, np.concatenate([d["obs_traj"][0].T, d["pred_traj"][0].T]))
    assert np.array_equal(early, np.concatenate([d["obs_traj"][1].T, d["pred_traj"][1].T[:35]]))
    tracks, holes = cg.node_tracks(d, 2)                                # agent 4: cut at its hole
    assert tracks[0][1].shape == (TO + 10, 2) and holes == 1
    assert not (tracks[0][1] == 0).all(1).any()                         # no masked zero ever enters a node


def _build(t, d, seqs):
    return cg.build_env(d, seqs, cg.export6(), t.Environment, t.Scene, t.Node, t.derivative_of, t.pd)


@needs_tpp
def test_scene_layout_puts_the_last_observed_frame_at_t0_plus_19():
    d = _canonical()
    with _tpp() as t:
        _check_layout(d, *_build(t, d, [0, 1]))


def _check_layout(d, env, types, stats):
    assert stats == {"scenes": 2, "nodes": 4, "futures_cut_at_a_hole": 0}
    assert types == {"av2_a": {"100": "Car", "101": "Pedestrian"}, "av2_b": {"102": "Bus", "103": "Car"}}
    assert env.standardization == cg.export6().STANDARDIZATION_PED_ONLY
    assert list(env.attention_radius.values()) == [cg.ATTENTION_RADIUS]
    for scene, q in zip(env.scenes, (0, 1)):
        assert scene.timesteps == 30 + TO + TF and scene.dt == cg.DT
        a = int(d["seq_start_end"][q][0])
        for j, node in enumerate(scene.nodes):
            i = a + j
            n_fut = int(d["fut_mask"][i].sum())
            assert node.first_timestep == 30 and node.timesteps == TO + n_fut
            hist = node.get(np.array([30, 30 + TO - 1]), {"position": ["x", "y"]})
            assert np.allclose(hist, d["obs_traj"][i].T, atol=1e-5)     # the window's 20 observed frames
            fut = node.get(np.array([30 + TO, 30 + TO + TF - 1]), {"position": ["x", "y"]})
            assert np.allclose(fut[:n_fut], d["pred_traj"][i].T[:n_fut], atol=1e-5)
            assert np.isnan(fut[n_fut:]).all()                          # beyond the valid future: absent, not 0


@needs_tpp
def test_export_copies_the_idd_pickle_and_config_and_merges_the_class_map(tmp_path, monkeypatch):
    with _tpp() as t:
        _check_export(t, tmp_path, monkeypatch)


def _check_export(t, tmp_path, monkeypatch):
    dill = t.dill
    d = _canonical()
    np.savez(tmp_path / "d.npz", **d)
    (tmp_path / "s.json").write_text(json.dumps({"splits": {"train": ["av2_a", "av2_b"], "val": ["av2_c"],
                                                            "test": ["idd_v"]}}))
    idd = tmp_path / "idd_export"
    idd.mkdir()
    env, types, _ = _build(t, d, [3])                                   # stands in for the IDD export's val.pkl
    with open(idd / "val.pkl", "wb") as f:
        dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
    (idd / "config.json").write_text(json.dumps({"prediction_horizon": 60, "maximum_history_length": 19}))
    (idd / "node_type_map.json").write_text(json.dumps(types))
    out = tmp_path / "out"
    argv = ["cg_av2_trajectronpp.py", "--name", "x", "--data", str(tmp_path / "d.npz"), "--splits-file",
            str(tmp_path / "s.json"), "--idd-export", str(idd), "--out-dir", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert cg.main() == 0
    assert (out / "test.pkl").read_bytes() == (idd / "val.pkl").read_bytes()
    assert (out / "config.json").read_bytes() == (idd / "config.json").read_bytes()
    tmap = json.loads((out / "node_type_map.json").read_text())
    assert set(tmap) == {"av2_a", "av2_b", "av2_c", "idd_v"} and tmap["idd_v"] == types["idd_v"]
    meta = json.loads((out / "export_meta.json").read_text())
    assert meta["dt"] == 0.1 and meta["hetero"] is False and meta["splits"]["val"]["futures_cut_at_a_hole"] == 1
    with open(out / "train.pkl", "rb") as f:
        assert [s.name for s in dill.load(f).scenes] == ["av2_a", "av2_b"]
    (tmp_path / "s.json").write_text(json.dumps({"splits": {"train": ["av2_a"], "val": ["av2_c"], "test": ["av2_b"]}}))
    with pytest.raises(SystemExit, match="does not hold exactly the test scenes"):
        cg.main()                                                       # the IDD pickle must match the split's test


EXPORT = REPO / "outputs" / "chaos_gap" / "tpp_av2_matched_dev"


@needs_tpp
@pytest.mark.skipif(not (EXPORT / "val.pkl").exists(), reason="AV2-matched Trajectron++ export not built")
def test_real_export_has_a_node_for_every_av2_window_with_the_exact_history():
    with _tpp() as t:
        _check_real_export(t.dill)


def _check_real_export(dill):
    d = np.load(REPO / "outputs" / "chaos_gap" / "av2_matched_dev.npz", allow_pickle=True)
    from dd_split import load_splits
    mapping = load_splits(REPO / "splits" / "chaos_gap" / "av2_matched_dev.json")
    for split in ("train", "val"):
        with open(EXPORT / f"{split}.pkl", "rb") as f:
            scenes = {s.name: s for s in dill.load(f, encoding="latin1").scenes}
        seqs = [q for q, s in enumerate(d["scene_id"]) if mapping.get(str(s)) == split]
        assert sorted(scenes) == sorted(str(d["scene_id"][q]) for q in seqs)
        for q in seqs[::25]:                                            # every 25th sequence: about 1 s
            scene, t0 = scenes[str(d["scene_id"][q])], int(d["t0"][q])
            nodes = {n.id: n for n in scene.nodes}
            for i in range(*d["seq_start_end"][q]):
                node = nodes[str(d["agent_id"][i])]
                assert node.first_timestep == t0 and node.timesteps == TO + int(d["fut_mask"][i].sum())
                hist = node.get(np.array([t0, t0 + TO - 1]), {"position": ["x", "y"]})
                assert np.allclose(hist, d["obs_traj"][i].T, atol=1e-4)


def test_evaluator_scores_the_windows_of_the_canonical_file_it_is_given(tmp_path, monkeypatch):
    ev = load_script("eval_trajectronpp_canonical.py")
    assert ev.CANONICAL == REPO / "outputs" / "canonical_world_obs20_pred60.npz"   # the default is unchanged
    d = _canonical()
    np.savez(tmp_path / "d.npz", **d)
    (tmp_path / "s.json").write_text(json.dumps({"splits": {"train": ["av2_a", "av2_b"], "val": ["av2_c"],
                                                            "test": ["idd_v"]}}))
    monkeypatch.setattr(ev, "CANONICAL", tmp_path / "d.npz")
    monkeypatch.setattr(ev, "SPLITS_FILE", tmp_path / "s.json")
    assert ev.windows("val") == [("av2_c", 30, 4), ("av2_c", 30, 5)]
    assert ev.windows("test") == [("idd_v", 6, 6), ("idd_v", 6, 7)]


def test_memory_cap_patch_is_idempotent(tmp_path):
    patch = load_script("patch_trajectronpp.py")
    rel, old, new, guard = next(p for p in patch.SPECIFIC if p[3] == "DD_GPU_MEM_FRAC")
    f = tmp_path / "train.py"
    f.write_text("import os\nimport torch\ntorch.cuda.set_device(args.device)\nprint('go')\n", encoding="utf-8")
    assert rel == "trajectron/train.py"
    assert patch.patch_file(f, [(old, new, guard)]) == 1
    once = f.read_text(encoding="utf-8")
    assert patch.patch_file(f, [(old, new, guard)]) == 0 and f.read_text(encoding="utf-8") == once
    assert once.count("set_per_process_memory_fraction") == 1 and once.index("set_device") < once.index("DD_GPU")
