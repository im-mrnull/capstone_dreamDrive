"""Argoverse 2 -> canonical adapter (scripts/cg_av2_canonical.py): the window rule is the
canonical exporter's, the arrays are the canonical file's, and the kept agents are the
dynamic road users without the ego."""

import numpy as np
import pandas as pd
import pytest

from conftest import CANONICAL, REPO, load_script

cg = load_script("cg_av2_canonical.py")
exp5 = cg.exp5
AV2_VAL = REPO / "data_external" / "av2" / "val"


def _track(tid, otype, frames, x0=0.0, vx=1.0):
    f = np.asarray(list(frames))
    return pd.DataFrame({"track_id": tid, "object_type": otype, "timestep": f, "position_x": x0 + vx * 0.1 * f,
                         "position_y": 0.5 * f, "scenario_id": "s1", "city": "austin"})


def _scenario():
    full, obs_only = range(0, 110), range(30, 50)
    return pd.concat([
        _track("10", "vehicle", full),                     # kept
        _track("11", "pedestrian", range(25, 60), 5.0),    # kept: full observation, 10 future frames
        _track("12", "cyclist", range(31, 110), 7.0),      # dropped: observation misses frame 30
        _track("13", "motorcyclist", obs_only, 9.0),       # dropped: no future frame
        _track("AV", "vehicle", full, 2.0),                # dropped: the ego
        _track("14", "static", full, 3.0),                 # dropped: not a road user
        _track("15", "riderless_bicycle", full, 4.0),      # dropped: not a road user
        _track("16", "bus", full, 6.0)], ignore_index=True)  # kept


def test_kept_agents_are_road_users_with_the_canonical_presence_rule():
    w = cg.scenario_window(*cg.scenario_arrays(_scenario()))
    assert list(w["agent_id"]) == [10, 11, 16]
    assert list(w["agent_type"]) == ["Car", "Pedestrian", "Bus"]
    assert w["fut_mask"][1].sum() == 10 and w["fut_mask"][0].all() and w["fut_mask"][2].all()


def test_window_is_frames_30_to_109_of_the_parquet_positions():
    df = _scenario()
    w = cg.scenario_window(*cg.scenario_arrays(df))
    car = df[df.track_id == "10"].set_index("timestep")
    assert np.allclose(w["obs_traj"][0, 0], car.loc[30:49, "position_x"]) and np.allclose(w["obs_traj"][0, 1], car.loc[30:49, "position_y"])
    assert np.allclose(w["pred_traj"][0, 0], car.loc[50:109, "position_x"])
    assert np.all(w["pred_traj"][1][:, ~w["fut_mask"][1]] == 0.0)          # absent future steps are zero-filled


def test_the_window_equals_the_canonical_exporter_on_the_same_scene(tmp_path):
    pos, mask, aid, atype = cg.scenario_arrays(_scenario())
    np.savez(tmp_path / "s1.npz", pos=pos, mask=mask, agent_id=aid, agent_type=atype)
    ref, meta = exp5.export_dataset(tmp_path, cg.OBS_LEN, cg.PRED_LEN, stride=cg.T0, min_agents=cg.MIN_AGENTS,
                                    presence="obs+1fut")
    k = [m.t0 for m in meta].index(cg.T0)
    lo, hi = ref["seq_start_end"][k]
    w = cg.scenario_window(pos, mask, aid, atype)
    for key, v in w.items():
        assert v.dtype == ref[key].dtype and np.array_equal(v, ref[key][lo:hi]), key


def test_a_scenario_with_fewer_than_two_agents_is_skipped():
    assert cg.scenario_window(*cg.scenario_arrays(_track("10", "vehicle", range(110)))) is None


@pytest.mark.skipif(not CANONICAL.exists(), reason="canonical dataset not built")
@pytest.mark.skipif(not AV2_VAL.exists(), reason="no local Argoverse 2 sample")
def test_converted_sample_has_the_canonical_keys_shapes_and_dtypes():
    ds, stats = cg.convert(AV2_VAL, limit=20)
    ref = np.load(CANONICAL, allow_pickle=True)
    for key in ref.files:
        assert ds[key].dtype == ref[key].dtype and ds[key].shape[1:] == ref[key].shape[1:], key
    assert set(ds["agent_type"]) <= set(cg.AV2_CLASS.values())
    assert ds["seq_start_end"][-1, 1] == ds["obs_traj"].shape[0] == stats["agents"]
    assert all(s.startswith("av2_") for s in ds["scene_id"]) and (ds["t0"] == cg.T0).all()
    assert np.isfinite(ds["obs_traj"]).all() and np.isfinite(ds["pred_traj"]).all()
