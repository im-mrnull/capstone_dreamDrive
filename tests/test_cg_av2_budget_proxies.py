"""Size-matched AV2 proxies for the fine-tuning budget (scripts/cg_av2_budget_proxies.py): sliding
windows are the canonical exporter's stride-3 windows, the fill matches distinct tracks without
overshooting, and the written files hold AV2 only, with the frozen AV2 selection and readout sets."""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import REPO, load_script

bp = load_script("cg_av2_budget_proxies.py")
cg, exp5 = bp.cg, bp.cg.exp5
SPLITS = sorted((REPO / "splits" / "chaos_gap" / "budget").glob("av2_k*_r*.json"))
REPORT = REPO / "outputs" / "chaos_gap" / "dev" / "budget_proxy_match.json"


def _track(tid, otype, frames, x0=0.0):
    f = np.asarray(list(frames))
    return pd.DataFrame({"track_id": tid, "object_type": otype, "timestep": f, "position_x": x0 + 0.1 * f,
                         "position_y": 0.5 * f, "scenario_id": "s1", "city": "austin"})


def _arrays():
    return cg.scenario_arrays(pd.concat([
        _track("10", "vehicle", range(0, 110)), _track("11", "pedestrian", range(10, 70), 5.0),
        _track("12", "cyclist", range(20, 110), 7.0), _track("13", "bus", range(0, 45), 9.0)], ignore_index=True))


def test_t0s_are_stride_3_starts_ending_at_the_canonical_t0():
    assert bp.t0s(6) == [15, 18, 21, 24, 27, 30]
    assert bp.t0s(11)[0] == 0 and all(b - a == 3 for a, b in zip(bp.t0s(11), bp.t0s(11)[1:]))


def test_windows_at_every_start_equal_the_canonical_exporter_with_stride_3(tmp_path):
    pos, mask, aid, atype = _arrays()
    np.savez(tmp_path / "s1.npz", pos=pos, mask=mask, agent_id=aid, agent_type=atype)
    ref, meta = exp5.export_dataset(tmp_path, cg.OBS_LEN, cg.PRED_LEN, stride=bp.STRIDE, min_agents=cg.MIN_AGENTS,
                                    presence="obs+1fut")
    ours = bp.scenario_windows((pos, mask, aid, atype))
    assert sorted(ours) == sorted(m.t0 for m in meta if m.t0 <= cg.T0)  # the same windows exist
    for k, m in enumerate(meta):
        if m.t0 > cg.T0:
            continue
        lo, hi = ref["seq_start_end"][k]
        for key, v in ours[m.t0].items():
            assert v.dtype == ref[key].dtype and np.array_equal(v, ref[key][lo:hi]), (m.t0, key)
    assert cg.scenario_window(pos, mask, aid, atype).keys() == ours[cg.T0].keys()
    for key, v in cg.scenario_window(pos, mask, aid, atype).items():   # the default is still t0 = 30
        assert np.array_equal(v, ours[cg.T0][key])


def test_fill_matches_tracks_skips_overshoot_and_stops_at_the_target():
    def win(ids):
        return {"agent_id": np.asarray(ids)}
    windows = {"a": {30: win([1, 2, 3])}, "b": {30: win([1, 2, 3, 4, 5])}, "c": {27: win([1, 2]), 30: win([2, 9])},
               "d": {30: win([1, 2])}, "e": {30: win([7, 8])}}
    assert bp.fill(list("abcde"), windows, 2, 6) == ["a", "c"]          # b overshoots; c has 3 tracks over 2 windows
    assert bp.fill(list("abcde"), windows, 1, 6) == ["a", "c"]          # one window: c has 2; d, e overshoot (shortfall 1)
    assert bp.fill(list("abcde"), windows, 1, 7) == ["a", "c", "d"]
    assert bp.fill(list("ab"), windows, 2, 100) == ["a", "b"]          # candidates run out: the shortfall is reported


def test_structure_counts_windows_per_track_and_composition():
    obs = np.zeros((2, 2, 20))
    obs[0, 0] = np.arange(20)                                           # moving
    w = {"agent_id": np.array([1, 2]), "agent_type": np.array(["Car", "Pedestrian"], dtype=object), "obs_traj": obs}
    s = bp.structure([("s", w), ("s", w), ("t", w)])
    assert s["distinct_tracks"] == 4 and s["agent_windows"] == 6 and s["scenes"] == 2
    assert s["windows_per_track"]["histogram"] == {1: 2, 2: 2} and s["windows_per_track"]["mean"] == 1.5
    assert s["moving_share"] == 0.5 and s["class_groups"]["vehicle"] == 0.5


@pytest.mark.skipif(not SPLITS, reason="budget proxies not built")
def test_written_split_files_hold_av2_only_with_the_frozen_selection_and_readout_sets():
    matched = json.loads((REPO / "splits" / "chaos_gap" / "av2_matched_dev.json").read_text(encoding="utf-8"))["splits"]
    vals, tests = set(), set()
    for f in SPLITS:
        d = json.loads(f.read_text(encoding="utf-8"))
        sp = d["splits"]
        assert all(s.startswith("av2_") for v in sp.values() for s in v)                # no IDD scene
        assert sorted(sp["val"]) == sorted(matched["val"])                             # the only selection set
        assert len(sp["test"]) == bp.N_VAL and not set(sp["test"]) & set(sp["val"])
        assert not set(sp["train"]) & (set(matched["train"]) | set(sp["val"]) | set(sp["test"]))
        assert d["t0s"] == bp.t0s(d["n_w"]) and d["generator_git_blob"]
        vals.add(tuple(sp["val"]))
        tests.add(tuple(sp["test"]))
    assert len(vals) == 1 and len(tests) == 1                                          # frozen across draws


@pytest.mark.skipif(not REPORT.exists(), reason="budget proxies not built")
def test_every_proxy_passed_the_acceptance_gate():
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    assert rep["accepted"] and len(rep["draws"]) == len(bp.KS) * bp.DRAWS
    for name, d in rep["draws"].items():
        assert abs(d["ratio"]["agent_windows"] - 1) <= bp.TOL and abs(d["ratio"]["windows_per_track"] - 1) <= bp.TOL, name
        assert 0 <= d["shortfall_tracks"] <= 0.02 * d["idd"]["distinct_tracks"] + 1, name
