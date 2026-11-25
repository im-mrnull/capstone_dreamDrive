"""Chaos-gap evaluation contract: the formal gate of the transfer study (plan, section 1.6).

Every chaos-gap metrics JSON must be scored on exactly the canonical agents of its own data and
split files. Every prediction dump must hold those agents, val then test, in canonical row order,
with 60 predicted frames, the canonical ground truth and fut_mask, and must reproduce its JSON's
most-likely ADE/FDE under an implementation independent of dd_data (ADE over the valid steps, FDE
at the last valid step). Trajectron++ may fall back to CV only where it has no node: never on AV2,
and on IDD val exactly as often as the IDD-trained rows. A JSON or dump that fails is not a result.

Excluded by design: the upstream dense Trajectron++ protocol (script 7's own JSON, scored at every
node-timestep), which is reported separately; only its canonical re-scoring counts."""

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "outputs" / "chaos_gap"
TF = 60


def _path(p) -> Path:
    p = Path(str(p).replace("\\", "/"))
    return p if p.is_absolute() else REPO / p


def _load(f: Path):
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _metrics_files() -> list:
    out = []
    for f in sorted(ROOT.glob("**/*.json")):
        d = _load(f)
        if not (isinstance(d, dict) and all(isinstance(d.get(s), dict) for s in ("val", "test"))):
            continue
        src = d.get("data") or d.get("input")
        if src is None or "splits_file" not in d or ".pkl" in str(src):
            continue                                                    # *.stats.json, the dense T++ JSON
        out.append(f)
    return out


def _dumps() -> list:
    """(dump, its metrics JSON): <stem>_preds.npz beside <stem>.json, and the canonical Trajectron++
    dump beside trajectronpp_canonical_seed0.json."""
    pairs = [(p, p.with_name(p.name[:-len("_preds.npz")] + ".json")) for p in ROOT.glob("**/*_preds.npz")]
    pairs += [(p, p.with_name("trajectronpp_canonical_seed0.json"))
              for p in ROOT.glob("**/preds_trajectronpp_canonical.npz")]
    return sorted((p, j) for p, j in pairs if j.exists())


@lru_cache(maxsize=None)
def _population(data: str, splits: str) -> dict:
    """{split: canonical row indices (val, test), scene: scene id per row} of a data file under a
    split file. Rows follow file order: sequences then their agents (canonical format), or the
    per-agent windows of scripts/9 (CV, MLP)."""
    sys.path.insert(0, str(REPO / "scripts"))
    from dd_split import load_splits
    m, d = load_splits(splits), np.load(data, allow_pickle=True)
    if "seq_start_end" in d.files:
        se, sid = d["seq_start_end"], d["scene_id"]
        scene = np.repeat(np.asarray(sid, dtype=object), se[:, 1] - se[:, 0])
    else:
        scene = np.asarray(d["scene_id"], dtype=object)
    out = {s: np.flatnonzero([m.get(str(x)) == s for x in scene]) for s in ("val", "test")}
    out["scene"] = scene
    return out


def _n_agents(summary) -> list:
    """Every n_agents in a split's summary: flat, most_likely / best_of_k, or top1 / oracle_best_of_k."""
    out = [summary["n_agents"]] if "n_agents" in summary else []
    for k, v in summary.items():
        if k != "per_class" and isinstance(v, dict):
            out += _n_agents(v)
    return out


def _most_likely(summary) -> dict:
    return next((summary[k] for k in ("most_likely", "top1") if k in summary), summary)


def _sources(d):
    data, splits = _path(d.get("data") or d["input"]), _path(d["splits_file"])
    if not (data.exists() and splits.exists()):
        pytest.skip("its data or split file is not on this machine")
    return data, splits


def _ade_fde(pred, gt, fm) -> tuple:
    """Independent of dd_data: the mean error over valid steps, and the error at the last valid step."""
    err = np.sqrt(((np.asarray(pred, np.float64) - np.asarray(gt, np.float64)) ** 2).sum(-1))
    ade = np.array([e[mask].mean() for e, mask in zip(err, fm)])
    last = np.array([np.flatnonzero(mask)[-1] for mask in fm])
    return float(ade.mean()), float(err[np.arange(len(err)), last].mean())


METRICS, DUMPS = _metrics_files(), _dumps()
TPP = [f for f in METRICS if f.name.startswith("trajectronpp_canonical_seed")]
rel = lambda f: f.relative_to(ROOT).as_posix()                         # noqa: E731


@pytest.mark.skipif(not METRICS, reason="no chaos-gap metrics generated")
@pytest.mark.parametrize("f", METRICS, ids=rel)
def test_metrics_cover_exactly_the_canonical_agents_of_their_split_file(f):
    d = _load(f)
    pop = _population(*map(str, _sources(d)))
    for split in ("val", "test"):
        ns = _n_agents(d[split])
        assert ns and all(n == len(pop[split]) for n in ns), (split, ns, len(pop[split]))


@pytest.mark.skipif(not DUMPS, reason="no chaos-gap prediction dumps generated")
@pytest.mark.parametrize("dump,metrics", DUMPS, ids=[rel(p) for p, _ in DUMPS])
def test_dumps_are_the_canonical_rows_and_reproduce_their_metrics(dump, metrics):
    d = _load(metrics)
    data, splits = _sources(d)
    pop = _population(str(data), str(splits))
    n_val, n_test = len(pop["val"]), len(pop["test"])
    rows = np.concatenate([pop["val"], pop["test"]])
    p, c = np.load(dump, allow_pickle=True), np.load(data, allow_pickle=True)
    assert p["pred"].shape == (n_val + n_test, TF, 2)
    assert list(p["split"]) == ["val"] * n_val + ["test"] * n_test
    assert list(map(str, p["scene_id"])) == list(map(str, pop["scene"][rows]))
    assert np.array_equal(p["fut_mask"].astype(bool), c["fut_mask"][rows].astype(bool))
    gt = c["pred_traj"][rows].transpose(0, 2, 1) if "pred_traj" in c.files else c["fut"][rows]
    assert np.allclose(p["gt"], gt, atol=1e-4)                          # the canonical future, never a node's
    for split, sel in (("val", slice(0, n_val)), ("test", slice(n_val, None))):
        ade, fde = _ade_fde(p["pred"][sel], p["gt"][sel], p["fut_mask"][sel].astype(bool))
        ml = _most_likely(d[split])
        assert ade == pytest.approx(ml["ade"], abs=1e-3) and fde == pytest.approx(ml["fde"], abs=1e-3), split


@pytest.mark.skipif(not TPP, reason="no canonical Trajectron++ re-scoring generated")
@pytest.mark.parametrize("f", TPP, ids=rel)
def test_trajectronpp_falls_back_to_cv_only_where_it_has_no_node(f):
    d = _load(f)
    _, splits = _sources(d)
    sp = json.loads(splits.read_text(encoding="utf-8"))["splits"]
    bench = _load(REPO / "outputs" / "benchmark_v2" / "trajectronpp_canonical_seed0.json")
    idd_val = set(json.loads((REPO / "splits" / "scene_splits.json").read_text(encoding="utf-8"))["splits"]["val"])
    for split in ("val", "test"):
        if all(str(s).startswith("av2_") for s in sp[split]):
            assert d[split]["n_cv_fallback"] == 0, split               # every AV2 window has its node
        elif set(sp[split]) == idd_val and bench:
            assert d[split]["n_cv_fallback"] == bench["val"]["n_cv_fallback"], split
