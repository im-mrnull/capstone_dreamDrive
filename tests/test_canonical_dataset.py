"""Invariants of the committed canonical v2 dataset and its per-agent view."""

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
CANONICAL = REPO / "outputs" / "canonical_world_obs20_pred60.npz"
AGENT_WINDOWS = REPO / "outputs" / "agent_windows_canonical_obs20_pred60.npz"
STATS = REPO / "outputs" / "canonical_world_obs20_pred60_stats.json"

needs_canonical = pytest.mark.skipif(not CANONICAL.exists(), reason="canonical dataset not generated")
needs_windows = pytest.mark.skipif(not AGENT_WINDOWS.exists(), reason="agent windows not generated")


@needs_canonical
def test_canonical_invariants():
    d = np.load(CANONICAL, allow_pickle=True)
    n = d["obs_traj"].shape[0]
    assert d["pred_traj"].shape == (n, 2, 60)
    assert d["fut_mask"].shape == (n, 60)
    assert d["agent_type"].shape == (n,)
    # Full observation guaranteed; at least one valid future step per agent.
    assert np.isfinite(d["obs_traj"]).all()
    assert d["fut_mask"].any(axis=1).all()
    # Valid future steps carry finite positions; invalid steps are zero-filled.
    fm = d["fut_mask"][:, None, :]
    assert np.isfinite(d["pred_traj"]).all()
    assert (d["pred_traj"] * (~fm)).sum() == 0.0
    # Every agent has a non-empty class label.
    assert all(str(t) != "" for t in d["agent_type"])
    # seq_start_end contiguous and consistent.
    sse = d["seq_start_end"]
    assert sse[0, 0] == 0 and (sse[1:, 0] == sse[:-1, 1]).all() and sse[-1, 1] == n


@needs_canonical
def test_canonical_counts_match_stats():
    d = np.load(CANONICAL, allow_pickle=True)
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    assert stats["num_agents_total"] == d["obs_traj"].shape[0]
    assert stats["num_seqs"] == d["seq_start_end"].shape[0]
    assert stats["presence"] == "obs+1fut"
    # The survivor filter removal must actually have added agents vs. legacy 4,443.
    assert stats["num_agents_total"] > 4443


@needs_windows
@needs_canonical
def test_agent_windows_match_canonical():
    d = np.load(CANONICAL, allow_pickle=True)
    w = np.load(AGENT_WINDOWS, allow_pickle=True)
    assert w["obs"].shape[0] == d["obs_traj"].shape[0]
    assert (w["fut_mask"] == d["fut_mask"]).all()
    assert (w["agent_type"] == d["agent_type"]).all()
    # Centering invariant: last observed point is the origin.
    assert np.abs(w["obs"][:, -1, :]).max() < 1e-4
    # Heading normalization: net observed displacement points along +x
    # for clearly moving agents.
    disp = w["obs"][:, -1, :] - w["obs"][:, -6, :]  # last 5 deltas summed
    moving = np.linalg.norm(disp, axis=1) > 1.0
    assert moving.any()
    assert (disp[moving, 0] > 0).mean() > 0.99
