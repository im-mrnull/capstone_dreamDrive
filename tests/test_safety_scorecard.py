"""Safety scorecard building blocks (scripts/safety_scorecard.py)."""

from pathlib import Path

import numpy as np
import pytest

from conftest import load_script

REPO = Path(__file__).resolve().parent.parent
CANONICAL = REPO / "outputs" / "canonical_world_obs20_pred60.npz"
EGO = REPO / "outputs" / "ego_trajectories.csv"

sc = load_script("safety_scorecard.py")


def test_conflict_label_and_danger_score_on_a_crossing_agent():
    """An agent that crosses the ego's path is a conflict, one that stays 20 m away is not;
    a prediction on the crossing path gets the lower (more dangerous) score."""
    T = 60
    t = np.arange(T, dtype=float)
    ego = np.stack([t * 0.5, np.zeros(T)], -1)[None].repeat(2, 0)          # ego drives along +x
    crossing = np.stack([np.full(T, 15.0), 30.0 - t], -1)                     # passes (15, 0) at t=30
    parallel = np.stack([t * 0.5, np.full(T, 20.0)], -1)
    gt = np.stack([crossing, parallel])
    fm = np.ones((2, T), bool)
    assert sc.conflict_labels(gt, fm, ego).tolist() == [True, False]
    pred = gt[:, None]                                                        # perfect predictions
    D = sc.danger_scores(pred, np.ones((2, T)), fm, ego)
    assert D[0] < 0 < D[1]
    # a step that is not valid (no ground truth) can neither create a conflict nor a flag
    fm_masked = fm.copy(); fm_masked[0, 25:35] = False
    assert not sc.conflict_labels(gt, fm_masked, ego, radius=2.0)[0]


def test_weighted_auc_matches_mann_whitney_with_ties():
    rng = np.random.default_rng(0)
    s = np.round(rng.normal(size=300), 1)                                    # many ties
    y = rng.random(300) < 0.3
    pos, neg = s[y], s[~y]
    mw = ((pos[:, None] < neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert sc.weighted_auc(s, y, np.ones(300)) == pytest.approx(mw)
    # integer weights == repeating rows
    w = rng.integers(0, 3, 300).astype(float)
    rep = np.repeat(np.arange(300), w.astype(int))
    assert sc.weighted_auc(s, y, w) == pytest.approx(sc.weighted_auc(s[rep], y[rep], np.ones(len(rep))))
    # a batch of weight vectors gives one AUC per row
    out = sc.weighted_auc(s, y, np.stack([np.ones(300), w]))
    assert out.shape == (2,) and out[0] == pytest.approx(mw)


def test_union_of_discs_band_covers_on_exchangeable_data():
    """Calibrating on the min-over-samples error gives >= 1 - alpha coverage of the union."""
    rng = np.random.default_rng(1)
    n, K, T = 4000, 5, 3
    gt = rng.normal(size=(n, T, 2))
    pred = gt[:, None] + rng.normal(scale=1.5, size=(n, K, T, 2))
    fm = np.ones((n, T), bool)
    pop = {"split": np.array(["val"] * (n // 2) + ["test"] * (n // 2)), "gt": gt, "fm": fm,
           "cls": np.array(["A"] * n)}
    r = sc.conformal_radii(pred, pop, alpha=0.1, min_cal=30)
    test = pop["split"] == "test"
    covered = sc.min_sample_dist(pred[test], gt[test]) <= r
    assert covered.mean() >= 0.88


@pytest.mark.skipif(not (CANONICAL.exists() and EGO.exists()), reason="data not generated")
def test_ego_path_is_aligned_with_canonical_windows():
    """t0 indexes the scene frames, so the ego's future is frames t0+20 .. t0+79."""
    import pandas as pd

    c = np.load(CANONICAL, allow_pickle=True)
    ego = pd.read_csv(EGO, dtype={"frame_id": str})
    for q in range(0, len(c["seq_start_end"]), 97):
        a, _ = c["seq_start_end"][q]
        sid, t0 = str(c["scene_id"][q]), int(c["t0"][q])
        scene = np.load(REPO / "outputs" / "multiagent_world_v1" / "scenes" / f"{sid}.npz", allow_pickle=True)
        e = ego[ego.scene_id == sid].sort_values("frame_id")
        assert [int(f) for f in e.frame_id] == [int(f) for f in scene["frame_id"]]
        j = [str(x) for x in scene["agent_id"]].index(str(c["agent_id"][a]))
        assert np.allclose(scene["pos"][t0:t0 + 20, j], c["obs_traj"][a].T)


@pytest.mark.skipif(not (REPO / "outputs" / "benchmark_v2" / "mlp_best_seed0.npz").exists(),
                    reason="MLP weights not present")
def test_mlp_world_frame_reproduces_its_benchmark_ade():
    import json

    pop = sc.load_population(REPO / "splits" / "scene_splits.json")
    pred = sc.package_predictions("single", "mlp", 0, pop, 1)[:, 0]
    t = pop["split"] == "test"
    from dd_data import masked_ade_fde_np
    ade = masked_ade_fde_np(pred[t], pop["gt"][t], pop["fm"][t])[0].mean()
    ref = json.loads((REPO / "outputs" / "benchmark_v2" / "mlp_seed0.json").read_text(encoding="utf-8"))
    assert ade == pytest.approx(ref["test"]["ade"], abs=1e-3)
