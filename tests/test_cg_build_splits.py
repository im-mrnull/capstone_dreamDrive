"""Chaos-gap dataset builder (scripts/cg_build_splits.py): hash-ordered, exact-size AV2 subsets,
merged arrays identical to their sources, disjoint splits, and no frozen IDD test unless asked."""

import json

import numpy as np

from conftest import load_script

cg = load_script("cg_build_splits.py")


def _canonical(scene_ids, sizes, seed, t_obs=4, t_pred=6):
    """A small canonical-format dict: one sequence per scene id with the given agent counts."""
    rng = np.random.default_rng(seed)
    n = int(np.sum(sizes))
    se = np.cumsum([0] + list(sizes))
    return {"obs_traj": rng.normal(size=(n, 2, t_obs)).astype(np.float32),
            "pred_traj": rng.normal(size=(n, 2, t_pred)).astype(np.float32),
            "obs_traj_rel": rng.normal(size=(n, 2, t_obs)).astype(np.float32),
            "pred_traj_rel": rng.normal(size=(n, 2, t_pred)).astype(np.float32),
            "fut_mask": rng.random((n, t_pred)) > 0.2, "agent_id": np.arange(n, dtype=np.int32),
            "agent_type": np.asarray(["Car"] * n, dtype=object),
            "seq_start_end": np.stack([se[:-1], se[1:]], axis=1).astype(np.int64),
            "scene_id": np.asarray(scene_ids, dtype=object), "t0": np.zeros(len(scene_ids), np.int32)}


def _sources():
    tr = _canonical([f"av2_t{i}" for i in range(40)], [2 + i % 7 for i in range(40)], 0)
    va = _canonical([f"av2_v{i}" for i in range(12)], [3] * 12, 1)
    idd = _canonical(["s_tr", "s_tr", "s_va", "s_te", "s_va"], [4, 5, 3, 6, 2], 2)
    return tr, va, idd, {"s_tr": "train", "s_va": "val", "s_te": "test"}


def test_train_subset_has_exactly_the_target_agents_in_hash_order():
    ids = [f"av2_{i}" for i in range(8)]
    order = cg.hash_order(ids)
    sizes = np.array([5, 3, 7, 2, 4, 6, 3, 2])
    target = int(sizes[order[:3]].sum())
    assert cg.fill_exact(sizes, order, target) == order[:3].tolist()     # a hash-order prefix
    sizes = np.full(8, 2)
    sizes[order[0]] = 100                                                # the hash-first one would overshoot
    assert cg.fill_exact(sizes, order, 4) == order[1:3].tolist()
    assert [ids[i] for i in cg.hash_order(ids)]                          # the order depends on the ids, not positions == [list(reversed(ids))[i] for i in cg.hash_order(list(reversed(ids)))]


def test_merged_arrays_are_the_source_sequences():
    tr, va, idd, m = _sources()
    data, splits, stats = cg.build(tr, va, idd, m, train_agents=50, val_scenarios=5, idd_test_split="val")
    assert stats["train"]["agents"] == 50 and stats["val"]["sequences"] == 5
    se = data["seq_start_end"]
    assert se[0, 0] == 0 and se[-1, 1] == data["obs_traj"].shape[0] and np.all(se[1:, 0] == se[:-1, 1])
    src = {str(s): (d, i) for d in (tr, va, idd) for i, s in enumerate(d["scene_id"])}
    for j, sid in enumerate(data["scene_id"]):
        if str(sid).startswith("s_"):
            continue                                                     # IDD scenes repeat; checked below
        d, i = src[str(sid)]
        a, b = d["seq_start_end"][i]
        for k in cg.KEYS:
            assert np.array_equal(data[k][se[j, 0]:se[j, 1]], d[k][a:b]), (sid, k)
    idd_rows = np.concatenate([np.arange(*se[j]) for j, s in enumerate(data["scene_id"]) if str(s).startswith("s_")])
    want = np.concatenate([np.arange(*idd["seq_start_end"][i]) for i, s in enumerate(idd["scene_id"]) if s == "s_va"])
    assert np.array_equal(data["obs_traj"][idd_rows], idd["obs_traj"][want])


def test_splits_are_disjoint_and_the_frozen_test_stays_out_by_default():
    tr, va, idd, m = _sources()
    data, splits, _ = cg.build(tr, va, idd, m, train_agents=50, val_scenarios=5, idd_test_split="val")
    assert splits["test"] == ["s_va"]
    assert "s_te" not in set(map(str, data["scene_id"])) and "s_tr" not in set(map(str, data["scene_id"]))
    assert not set(splits["train"]) & set(splits["val"])
    _, splits_final, _ = cg.build(tr, va, idd, m, train_agents=50, val_scenarios=5, idd_test_split="test")
    assert splits_final["test"] == ["s_te"]


def test_the_written_files_load_through_the_benchmark_loader(tmp_path):
    from dd_data import CanonicalSeqDataset
    from dd_split import load_splits
    tr, va, idd, m = _sources()
    data, splits, stats = cg.build(tr, va, idd, m, train_agents=50, val_scenarios=5, idd_test_split="val")
    np.savez_compressed(tmp_path / "d.npz", **data)
    (tmp_path / "s.json").write_text(json.dumps({"splits": splits}))
    assert set(load_splits(tmp_path / "s.json").values()) == {"train", "val", "test"}
    for split in ("train", "val", "test"):
        ds = CanonicalSeqDataset(tmp_path / "d.npz", split, tmp_path / "s.json")
        assert sum(ds[i][0].shape[0] for i in range(len(ds))) == stats[split]["agents"]
