"""Fine-tuning split files (scripts/cg_ft_splits.py): nested hash-ordered k-scene draws that skip
empty scenes; IDD val split once into the most balanced halves (half A selects, half B reports in
development); the frozen IDD test only with --final; and files that the benchmark loader reads."""

import itertools
import json
import sys
from collections import Counter

import numpy as np

from conftest import load_script

ft = load_script("cg_ft_splits.py")

IDD = {**{f"tr{i}": "train" for i in range(12)}, **{f"va{i}": "val" for i in range(4)}, "te0": "test", "te1": "test"}
STATS = {s: Counter(n=3, moving=2, vehicle=2, pedestrian=1) for s in IDD}
STATS["tr0"] = Counter()                                                # one empty train scene, as in IDD-3D
STATS["va0"] = Counter(n=6, moving=6, vehicle=6)                        # val halves differ, so balance matters


def test_draws_are_nested_exact_and_skip_empty_scenes():
    out = ft.build(IDD, STATS, ks=[2, 5, 11], draws=3, final=False)
    for r in range(3):
        a, b, c = (set(out[(k, r)]["train"]) for k in (2, 5, 11))
        assert len(a) == 2 and len(b) == 5 and a < b < c
        assert c == {f"tr{i}" for i in range(1, 12)}                    # k = all non-empty; tr0 never drawn
    assert len({tuple(out[(5, r)]["train"]) for r in range(3)}) == 3     # the draws differ


def test_balanced_halves_is_the_minimax_partition_with_the_larger_half_selecting():
    stats = {"a": Counter(n=10, moving=10, vehicle=10), "b": Counter(n=10, moving=0, pedestrian=10),
             "c": Counter(n=10, moving=10, vehicle=10), "d": Counter(n=10, moving=0, pedestrian=10),
             "e": Counter(n=4, moving=2, vehicle=2, pedestrian=2), "f": Counter(n=30, moving=15, vehicle=15, pedestrian=15)}
    sel, rep = ft.balanced_halves(stats, stats)
    best = min(ft.imbalance(A, [s for s in stats if s not in A], stats) for A in itertools.combinations(sorted(stats), 3))
    assert ft.imbalance(sel, rep, stats) == best                        # brute force agrees
    assert sorted(sel + rep) == sorted(stats) and not set(sel) & set(rep)
    assert sum(stats[s]["n"] for s in sel) >= sum(stats[s]["n"] for s in rep)


def test_dev_halves_val_and_final_keeps_half_a_with_the_frozen_test():
    dev = ft.build(IDD, STATS, ks=[5], draws=1, final=False)[(5, 0)]
    assert set(dev["val"]) | set(dev["test"]) == {f"va{i}" for i in range(4)}
    assert len(dev["val"]) == len(dev["test"]) == 2 and not set(dev["val"]) & set(dev["test"])
    assert not any(s.startswith("te") for v in dev.values() for s in v)
    final = ft.build(IDD, STATS, ks=[5], draws=1, final=True)[(5, 0)]
    assert final["val"] == dev["val"] and final["test"] == ["te0", "te1"]  # half A selects in both
    assert final["train"] == dev["train"]                               # the same draw either way


def test_hash_order_depends_on_the_ids_not_positions():
    ids = [f"s{i}" for i in range(9)]
    assert ft.hash_sorted(ids, "x") == ft.hash_sorted(list(reversed(ids)), "x")


def test_scene_stats_counts_agents_groups_and_moving():
    obs = np.zeros((3, 2, 20), np.float32)
    obs[0, 0] = np.arange(20)                                           # 10 m in the last second: moving
    obs[1, 0] = 0.01 * np.arange(20)                                    # 0.1 m/s: stationary
    d = {"seq_start_end": np.array([[0, 3]]), "scene_id": np.array(["s"], dtype=object), "obs_traj": obs,
         "agent_type": np.array(["Car", "ScooterRider", "Animal"], dtype=object)}
    assert ft.scene_stats(d)["s"] == Counter(n=3, moving=1, vehicle=1, **{"two-wheeler": 1, "other": 1})


def test_written_files_select_exactly_their_scenes(tmp_path, monkeypatch):
    from dd_data import CanonicalSeqDataset
    scenes = list(IDD)
    sizes = [STATS[s]["n"] for s in scenes]
    keep = [i for i, n in enumerate(sizes) if n]                        # an empty scene has no sequence
    n = sum(sizes)
    se = np.cumsum([0] + [sizes[i] for i in keep])
    rng = np.random.default_rng(0)
    np.savez_compressed(tmp_path / "d.npz", obs_traj=rng.normal(size=(n, 2, 20)).astype(np.float32),
                        pred_traj=rng.normal(size=(n, 2, 6)).astype(np.float32),
                        obs_traj_rel=rng.normal(size=(n, 2, 20)).astype(np.float32),
                        pred_traj_rel=rng.normal(size=(n, 2, 6)).astype(np.float32), fut_mask=np.ones((n, 6), bool),
                        agent_id=np.arange(n, dtype=np.int32), agent_type=np.asarray(["Car"] * n, dtype=object),
                        seq_start_end=np.stack([se[:-1], se[1:]], 1).astype(np.int64),
                        scene_id=np.asarray([scenes[i] for i in keep], dtype=object), t0=np.zeros(len(keep), np.int32))
    (tmp_path / "s.json").write_text(json.dumps({"splits": {p: [s for s, q in IDD.items() if q == p]
                                                            for p in ("train", "val", "test")}}))
    monkeypatch.setattr(sys, "argv", ["cg_ft_splits.py", "--idd", str(tmp_path / "d.npz"), "--idd-splits",
                                      str(tmp_path / "s.json"), "--ks", "3", "--draws", "2", "--out-dir", str(tmp_path / "ft")])
    assert ft.main() == 0
    for r in range(2):
        f = tmp_path / "ft" / f"idd_k03_r{r}_dev.json"
        meta = json.loads(f.read_text())
        assert meta["counts"]["train"] == 3 and meta["agents"]["train"] == 9
        assert meta["composition"]["val"]["agents"] == meta["agents"]["val"]
        for split in ("train", "val", "test"):
            ds = CanonicalSeqDataset(tmp_path / "d.npz", split, f)
            assert sum(ds[i][0].shape[0] for i in range(len(ds))) == meta["agents"][split]


def test_stop_set_is_held_out_train_and_never_drawn_and_test_is_all_val():
    out = ft.stop_set_splits(IDD, STATS, ks=[2, 5, 8], draws=3, n_stop=3)
    stops = {tuple(v["val"]) for v in out.values()}
    assert len(stops) == 1                                              # one S for every k and draw
    stop = set(next(iter(stops)))
    assert len(stop) == 3 and stop <= {f"tr{i}" for i in range(1, 12)}  # non-empty IDD train scenes only
    for (k, r), v in out.items():
        assert len(v["train"]) == k and not set(v["train"]) & stop
        assert v["test"] == [f"va{i}" for i in range(4)]                # all of IDD val; never the frozen test
        assert not any(s.startswith(("va", "te")) for s in v["train"] + v["val"])
    for r in range(3):
        a, b, c = (set(out[(k, r)]["train"]) for k in (2, 5, 8))
        assert a < b < c                                                # nested, as the dev draws
        full = ft.build(IDD, STATS, ks=[11], draws=3, final=False)[(11, r)]["train"]
        order = [s for s in ft.hash_sorted(full, f"{ft.SALT}-ft{r}") if s not in stop]
        assert sorted(order[:5]) == sorted(b)                           # the draw's own order with S removed
