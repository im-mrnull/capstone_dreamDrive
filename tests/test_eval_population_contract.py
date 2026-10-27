"""Shared eval-population contract for benchmark v2.

Written BEFORE the new models exist: it pins the population every model must be
scored on — the agents of outputs/canonical_world_obs20_pred60.npz, partitioned
by the frozen scene split, with the same per-timestep fut_mask. Both new models
(the map-free transformer and upstream AgentFormer) must satisfy the reusable
gate `assert_eval_population` from conftest, exercised in their own test suites.

This module locks the two ends of that contract:
  1. the population is a clean partition of the canonical dataset (matches stats);
  2. the eval harness (CanonicalSeqDataset + collate_pad) surfaces exactly that
     population — the invariant every model inherits by evaluating through it.
"""

import json
from pathlib import Path

import pytest

from conftest import canonical_split_populations

REPO = Path(__file__).resolve().parent.parent
CANONICAL = REPO / "outputs" / "canonical_world_obs20_pred60.npz"
STATS = REPO / "outputs" / "canonical_world_obs20_pred60_stats.json"
SPLITS_FILE = REPO / "splits" / "scene_splits.json"

needs_canonical = pytest.mark.skipif(
    not CANONICAL.exists(), reason="canonical dataset not generated")


@needs_canonical
def test_population_is_clean_partition_of_canonical():
    pops = canonical_split_populations()
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    total_agents = sum(p["agents"] for p in pops.values())
    # The three splits partition every canonical agent, nothing dropped/duplicated.
    assert total_agents == stats["num_agents_total"]
    for split in ("train", "val", "test"):
        assert pops[split]["agents"] > 0, f"{split} split has no agents"
        assert pops[split]["valid_steps"] > 0, f"{split} split has no valid future steps"
    # Model selection happens on val, final numbers on test: both must be real.
    assert pops["val"]["agents"] >= 100
    assert pops["test"]["agents"] >= 100


@needs_canonical
def test_harness_surfaces_exact_population():
    """CanonicalSeqDataset + collate_pad (the harness every model evaluates
    through) must surface exactly the canonical val/test agents and their
    fut_mask — real agents only, padding excluded."""
    import torch
    from torch.utils.data import DataLoader

    from dd_data import CanonicalSeqDataset, collate_pad

    pops = canonical_split_populations()
    for split in ("val", "test"):
        ds = CanonicalSeqDataset(str(CANONICAL), split, str(SPLITS_FILE),
                                 rot_aug=False, seed=0)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                            collate_fn=collate_pad)
        n_agents, n_steps = 0, 0
        for b in loader:
            n_agents += int(b.agent_mask.sum())
            valid = b.fut_mask & b.agent_mask.unsqueeze(-1)
            n_steps += int(valid.sum())
        assert n_agents == pops[split]["agents"], (
            f"{split}: harness yielded {n_agents} agents, canonical has "
            f"{pops[split]['agents']}")
        assert n_steps == pops[split]["valid_steps"], (
            f"{split}: harness yielded {n_steps} valid steps, canonical has "
            f"{pops[split]['valid_steps']}")


@needs_canonical
def test_metrics_jsons_report_the_canonical_population():
    """Every ranked benchmark row is scored on exactly the canonical val/test agents.
    Only rows flagged as Trajectron++'s dense protocol may differ (never ranked)."""
    from run_benchmark_v2 import DENSE_PROTOCOL, OUT, VARIANTS, _dig

    pops = canonical_split_populations()
    checked = 0
    for label, pattern, subpath, _, _ in VARIANTS:
        f = OUT / pattern.format(seed=0)
        if label in DENSE_PROTOCOL or not f.exists():
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        for split in ("val", "test"):
            assert _dig(m[split], subpath)["n_agents"] == pops[split]["agents"], (label, split)
        checked += 1
    if not checked:
        pytest.skip("no benchmark metrics JSONs yet")


@needs_canonical
def test_prediction_dumps_are_row_aligned_on_the_canonical_population():
    """Dumps feeding the conformal + bootstrap analyses cover the same rows (split,
    scene, class, future mask, ground truth), so paired comparisons are like for like."""
    import numpy as np

    from conformal_wrapper import canonical_dumps

    dumps = canonical_dumps(REPO / "outputs" / "benchmark_v2")
    if not dumps:
        pytest.skip("no prediction dumps yet")
    pops = canonical_split_populations()
    ref = None
    for npz in dumps:
        p = np.load(npz, allow_pickle=True)
        sp = p["split"].astype(object)
        for split in ("val", "test"):
            assert int((sp == split).sum()) == pops[split]["agents"], (npz.name, split)
            assert int(p["fut_mask"][sp == split].sum()) == pops[split]["valid_steps"], (npz.name, split)
        cur = {k: p[k] for k in ("split", "scene_id", "agent_type", "fut_mask")}
        cur["gt"] = np.where(p["fut_mask"][..., None], p["gt"], 0.0)
        if ref is None:
            ref = (npz.name, cur)
            continue
        for k, v in cur.items():
            assert np.array_equal(v, ref[1][k]), f"{npz.name} vs {ref[0]}: {k} differs"


def test_trajectronpp_canonical_windows_cover_the_population():
    """The matched Trajectron++ re-scoring visits each canonical val/test window once."""
    if not CANONICAL.exists():
        pytest.skip("canonical dataset not generated")
    from conftest import load_script

    ev = load_script("eval_trajectronpp_canonical.py")
    pops = canonical_split_populations()
    for split in ("val", "test"):
        w = ev.windows(split)
        assert len(w) == pops[split]["agents"]
        assert len({r for _, _, r in w}) == len(w)
