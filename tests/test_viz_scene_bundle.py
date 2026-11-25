"""Scene-viewer bundles (scripts/viz_scene_bundle.py): coordinate, time and identity alignment with
the recorded tracks, predictions equal to their saved sources (world frame), conformal bands from the
other half of IDD val only, the safety scorecard's definitions, and no frozen-test scene. Every check
recomputes its quantity from the raw files with numpy, not with the builder's helpers."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import load_script

REPO = Path(__file__).resolve().parent.parent
SCENE = "20220118105838_seq_8"
CANON = REPO / "outputs" / "canonical_world_obs20_pred60.npz"
vb = load_script("viz_scene_bundle.py")


def _sources_present() -> bool:
    """Every prediction dump and weight file the bundle reads (gitignored, so absent on CI)."""
    return all((REPO / src[k]).exists() for *_, src in vb.MODELS for k in ("dump", "weights") if k in src)


needs_data = pytest.mark.skipif(
    not (CANON.exists() and (REPO / "outputs" / "parsed_positions_world" / f"{SCENE}.csv").exists()
         and _sources_present()),
    reason="canonical data, parsed tracks or the prediction dumps not on this machine")
TOL = 0.0051                                                            # int16 centimetres


@lru_cache(maxsize=None)
def _bundle():
    return vb.build_bundle(vb.build_idd(SCENE))


def _splits(name):
    return json.loads((REPO / "splits" / name).read_text(encoding="utf-8"))["splits"]


def _windows(b):
    """Per window: decoded history, future, mask, last point, and agent ids."""
    out = []
    for w in b["windows"]:
        hist = vb.unpack(w["hist"])
        last = hist[:, -1, :]
        out.append({"w": w, "hist": hist, "last": last, "gt": vb.unpack(w["gt"], last[:, None, :]),
                    "fm": vb.unbits(w["fm"]), "ids": [b["agents"]["ids"][a] for a in w["agents"]]})
    return out


def test_encoding_round_trips_to_half_a_centimetre():
    rng = np.random.default_rng(0)
    a = rng.uniform(-300, 300, (7, 60, 2))
    a[0, 5] = np.nan
    back = vb.unpack(vb.pack(a, 1.5), 1.5)
    assert np.array_equal(np.isnan(back), np.isnan(a)) and np.nanmax(np.abs(back - a)) <= 0.005
    far = vb.unpack(vb.pack(np.array([[4000.0, -1.0]])))                 # out of int16 range: float32
    assert np.allclose(far, [[4000.0, -1.0]], atol=1e-3)
    m = rng.random((5, 60)) > 0.3
    assert np.array_equal(vb.unbits(vb.bits(m)), m)


@needs_data
def test_windows_align_with_the_recorded_tracks_in_time_and_identity():
    b = _bundle()
    d = pd.read_csv(REPO / "outputs" / "parsed_positions_world" / f"{SCENE}.csv", dtype={"frame_id": str, "obj_id": str})
    frames = sorted(d.frame_id.unique())
    track = vb.unpack(b["agents"]["track"])
    assert track.shape == (len(frames), len(b["agents"]["ids"]), 2)
    for aid, a in zip(b["agents"]["ids"], range(track.shape[1])):    # the track is the CSV, frame by frame
        rows = d[d.obj_id == str(aid)]
        f = rows.frame_id.map({fr: i for i, fr in enumerate(frames)}).values
        assert np.abs(track[f, a] - rows[["x", "y"]].values).max() <= TOL
    c = np.load(CANON, allow_pickle=True)
    q = {int(c["t0"][i]): i for i in np.flatnonzero(np.asarray(c["scene_id"], dtype=object) == SCENE)}
    assert sorted(q) == [w["t0"] for w in b["windows"]] == list(range(0, 21, 3))
    for x in _windows(b):
        w = x["w"]
        t0 = w["t0"]
        assert w["issue_frame"] == t0 + 19
        assert x["ids"] == [int(i) for i in c["agent_id"][slice(*c["seq_start_end"][q[t0]])]]
        for n, a in enumerate(w["agents"]):
            assert np.abs(x["hist"][n] - track[t0:t0 + 20, a]).max() <= TOL         # history = frames t0..t0+19
            v = x["fm"][n]
            assert np.abs(x["gt"][n][v] - track[t0 + 20:t0 + 80, a][v]).max() <= 2 * TOL
            assert np.isnan(x["gt"][n][~v]).all()                                    # masked steps never drawn


def _dump_rows(path, data, splits):
    """{(scene, t0, agent_id): row} of a dump, from its own data and split files."""
    c = np.load(data, allow_pickle=True)
    m = {s: k for k, v in json.loads(Path(splits).read_text(encoding="utf-8"))["splits"].items() for s in v}
    keys = []
    for want in ("val", "test"):
        for q, (a, z) in enumerate(c["seq_start_end"]):
            if m.get(str(c["scene_id"][q])) == want:
                keys += [(str(c["scene_id"][q]), int(c["t0"][q]), int(c["agent_id"][i]), i) for i in range(a, z)]
    return {k[:3]: (j, k[3]) for j, k in enumerate(keys)}, c


@needs_data
def test_predictions_equal_their_saved_sources_in_the_world_frame():
    b = _bundle()
    checked = []
    for m in b["models"]:
        if not m["available"] or m["key"] in ("cv", "mlp_idd"):
            continue
        spec = next(s for k, *_, s in vb.MODELS if k == m["key"])
        path = REPO / m["source"]
        cg = vb.CG_DEV in path.parents
        rows, c = _dump_rows(path, spec.get("data", vb.CG_DATA if cg else CANON),
                             spec.get("splits", vb.CG_SPLITS if cg else REPO / "splits" / "scene_splits.json"))
        p = np.load(path, allow_pickle=True)
        for x in _windows(b):
            ml = vb.unpack(x["w"]["preds"][m["key"]]["ml"], x["last"][:, None, :])
            for n, aid in enumerate(x["ids"]):
                j, r = rows[(SCENE, x["w"]["t0"], aid)]
                src = p["pred"][j].astype(np.float64)
                if spec.get("windows"):                                 # independent inverse: Procrustes fit
                    obs = c["obs_traj"][r].T.astype(np.float64)
                    loc = np.load(vb.CG / "av2_matched_dev_agents.npz", allow_pickle=True)["obs"][r].astype(np.float64)
                    u, _, vt = np.linalg.svd(loc.T @ (obs - obs[-1]))
                    R = (u @ vt).T                                      # world = R @ local
                    assert np.abs(loc @ R.T + obs[-1] - obs).max() < 1e-3
                    src = src @ R.T + obs[-1]
                assert np.abs(ml[n] - src).max() <= 2 * TOL, (m["key"], x["w"]["t0"], aid)
        checked.append(m["key"])
    assert {"stgat_idd", "transformer_idd", "mfp_idd", "agentformer_idd", "led_idd", "tpp_idd", "mlp_av2",
            "mfp_av2"} <= set(checked)


@needs_data
def test_cv_is_reproduced_independently():
    c = np.load(CANON, allow_pickle=True)                               # full-precision observations
    q = {int(c["t0"][i]): i for i in np.flatnonzero(np.asarray(c["scene_id"], dtype=object) == SCENE)}
    for x in _windows(_bundle()):
        obs = c["obs_traj"][slice(*c["seq_start_end"][q[x["w"]["t0"]]])].transpose(0, 2, 1).astype(np.float64)
        ml = vb.unpack(x["w"]["preds"]["cv"]["ml"], x["last"][:, None, :])
        v = (obs[:, -1] - obs[:, -6]) / 5                               # mean of the last 5 steps
        want = obs[:, -1][:, None] + v[:, None] * np.arange(1, 61)[None, :, None]
        assert np.abs(ml - want).max() <= 2 * TOL


@needs_data
def test_sets_probabilities_and_the_oracle():
    b = _bundle()
    for x in _windows(b):
        for key, e in x["w"]["preds"].items():
            m = next(mm for mm in b["models"] if mm["key"] == key)
            if "set" not in e:
                assert m["kind"] == "point"
                continue
            st = vb.unpack(e["set"], x["last"][:, None, None, :])
            assert st.shape[1] == m["K"]
            err = np.linalg.norm(st - x["gt"][:, None], axis=-1)
            for n in range(len(st)):
                v = x["fm"][n]
                if v.any():
                    ade = err[n][:, v].mean(1)
                    # both the set and the truth are decoded from centimetres: up to ~1 cm apart per step
                    assert e["min_ade"][n] == pytest.approx(ade.min(), abs=0.01)
                    assert ade[e["oracle_index"][n]] <= ade.min() + 0.01
            if "probs" in e:
                pr = vb.unpack(e["probs"])
                assert np.allclose(pr.sum(1), 1, atol=1e-4)
                if m["family"] == "MFP":                                # its top-1 is the best-scored mode
                    ml = vb.unpack(e["ml"], x["last"][:, None, :])
                    top = st[np.arange(len(st)), pr.argmax(1)]
                    assert np.abs(top - ml).max() <= 2 * TOL


def _radius(scores, alpha=0.1):
    s = np.sort(scores)
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return s[min(k, len(s)) - 1] if len(s) else np.inf


@needs_data
def test_conformal_bands_come_from_the_other_half_of_idd_val_only():
    b = _bundle()
    halves = _splits("chaos_gap/ft/idd_k72_r0_dev.json")
    cal = set(b["calibration"]["scenes"])
    assert cal == set(halves["val"]) and SCENE in halves["test"] and SCENE not in cal
    c = np.load(CANON, allow_pickle=True)
    sid = np.repeat(np.asarray(c["scene_id"], dtype=object), np.diff(c["seq_start_end"], axis=1)[:, 0])
    rows = np.flatnonzero(np.isin(sid, list(cal)))
    gt = c["pred_traj"][rows].transpose(0, 2, 1)
    fm = c["fut_mask"][rows].astype(bool)
    cls = c["agent_type"][rows].astype(str)
    obs = c["obs_traj"][rows].transpose(0, 2, 1)
    cv = obs[:, -1:] + (obs[:, -1] - obs[:, -6])[:, None] / 5 * np.arange(1, 61)[None, :, None]
    s = np.linalg.norm(cv - gt, axis=-1)                                # CV: scores recomputed independently
    x = _windows(b)[3]
    r = vb.unpack(x["w"]["preds"]["cv"]["radii_ml"])
    for n, k in enumerate(x["w"]["cls"]):
        for t in (0, 9, 29, 59):
            mine = fm[:, t] & (cls == k)
            want = _radius(s[mine, t]) if mine.sum() >= 30 else _radius(s[fm[:, t], t])
            assert r[n, t] == pytest.approx(want, abs=1e-4), (k, t)


@needs_data
def test_safety_quantities_follow_the_scorecard_definitions():
    b = _bundle()
    ego = vb.unpack(b["ego"]["track"])
    for x in _windows(b):
        w = x["w"]
        ef = ego[w["issue_frame"] + 1:w["issue_frame"] + 61]
        d = np.linalg.norm(x["gt"] - ef, axis=-1)
        conflict = [bool(np.any(d[n][x["fm"][n]] <= 5.0)) for n in range(len(d))]
        assert w["conflict"] == conflict
        for key, e in w["preds"].items():
            ml = vb.unpack(e["ml"], x["last"][:, None, :])
            r = vb.unpack(e["radii_ml"])
            z = (np.linalg.norm(ml - ef, axis=-1) - 5.0) / r
            for n in range(len(ml)):
                v = x["fm"][n]
                want = z[n][v].min() if v.any() else None
                got = e["danger_ml"][n]
                if got is None:
                    assert want is None or not np.isfinite(want)
                    continue
                # positions are decoded from centimetres (prediction and ego each within 0.5 cm, per axis),
                # so D can move by up to ~1.5 cm / r_t; the definition, not the rounding, is under test
                assert got == pytest.approx(want, abs=0.015 / r[n][v].min() + 1e-3), (key, n)


AV2_UUID = "5a0868d6-958c-42b4-aa77-a238010a5da7"


@pytest.mark.skipif(not (vb.vp.AV2_DIR / AV2_UUID).exists(), reason="the Argoverse 2 scenario is not extracted")
def test_av2_bundle_aligns_with_the_parquet_and_is_labelled_in_domain():
    b = vb.build_bundle(vb.build_av2(AV2_UUID))
    origin = np.array(b["scene"]["origin"])
    track = vb.unpack(b["agents"]["track"], origin)
    raw = vb.vp.load_av2(AV2_UUID)
    assert np.nanmax(np.abs(track - raw["pos"])) <= TOL                 # the parquet, frame by frame
    (w,) = b["windows"]
    assert w["t0"] == 30 and w["issue_frame"] == 49
    hist = vb.unpack(w["hist"], origin)
    for n, a in enumerate(w["agents"]):
        assert np.abs(hist[n] - raw["pos"][30:50, a]).max() <= 2 * TOL   # identity: canonical id = track id
    assert b["map"] and len(b["map"]["lanes"]) == len(raw["lanes"])
    avail = [m for m in b["models"] if m["available"]]
    assert {m["key"] for m in avail} >= {"cv", "led_av2", "agentformer_av2", "mfp_av2"}
    assert all("in-domain on AV2" in m["domain"] for m in avail if m["key"] != "cv")
    assert not any(k.endswith("_idd") for k in (m["key"] for m in b["models"]))
    assert b["scene"]["id"] not in b["calibration"]["scenes"]


@needs_data
def test_no_frozen_test_scene_and_trajectronpp_fallbacks_match_the_benchmark():
    b = _bundle()
    test = set(_splits("scene_splits.json")["test"])
    assert b["scene"]["id"] not in test and not (set(b["calibration"]["scenes"]) & test)
    # the CV stand-in rows over all of IDD val equal the benchmark's own count (24)
    val = set(_splits("scene_splits.json")["val"])
    rows = vb.idd_rows(val)
    spec = next(s for k, *_, s in vb.MODELS if k == "tpp_idd")
    got, _ = vb.read_dump(spec, "samples", "Trajectron++", set(rows))
    bench = json.loads((REPO / "outputs" / "benchmark_v2" / "trajectronpp_canonical_seed0.json").read_text(encoding="utf-8"))
    assert sum(v["fallback"] for v in got.values()) == bench["val"]["n_cv_fallback"]
