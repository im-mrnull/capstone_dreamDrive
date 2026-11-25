"""Post-hoc sensitivity (development): G0 = zero-shot - from-scratch reference (largest k) per half, with the same
paired scene bootstrap as cg_ft_diagnose, recomputed (1) on all scenes, to reproduce ft_compare.json, and (2) on
half B without 20220118103308_seq_27 (the scene with the strongest shared-velocity-jump artefact). Not a result."""
import json, sys
from pathlib import Path
import numpy as np
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(Path.cwd()))
import cg_autobot_fast_scene as f
fd = f.fd
idd = dict(np.load(f.IDD, allow_pickle=True)); ft_map = f.load_splits(fd.SPLIT)
_, order = fd.vb.canonical_order(f.IDD, fd.SPLIT)
pos = {(s, t, a): i for i, (s, t, a, _) in enumerate(order)}
n = len(order)
scene = np.array([s for s, *_ in order]); hb = np.array([ft_map[s] == "test" for s in scene])
def ade(path, split):
    p, g, m = f.load_aligned(path, split, pos, n)          # ADE from the dump's own gt: frame-free
    return f.metrics(p, g, m, g[:, 0])["ade"]
def runs(m, grid, k, arm):
    if grid is None:
        return [(fd.DEV / x, None) for x in fd.ZS[m]]
    out = []
    for r in range(3):
        t = f"{m}_k{k}_r{r}_{arm}"
        meta = json.loads((fd.DEV / grid / f"{t}.json").read_text(encoding="utf-8"))
        dump = fd.DEV / grid / (f"{t}_eval_preds.npz" if m in ("stgat", "transformer") else f"{t}_preds.npz")
        out.append((dump, str(fd.REPO / meta["splits_file"])))
    return out
cmp_ = json.loads((fd.DEV / "ft_compare.json").read_text(encoding="utf-8"))
res = {}
print(f"{'model':12s} {'grid':9s} {'half B G0 [95% CI] (reproduced)':40s} {'half B without seq_27':34s} half A")
for m in fd.MODELS:
    z = np.mean([ade(*r) for r in runs(m, None, None, None)], axis=0)
    for grid, kmax in (("ft_budget", "72"), ("ft_stop", "64")):
        ref = np.mean([ade(*r) for r in runs(m, grid, kmax, "scr")], axis=0)
        row = {}
        for name, sel in (("half A", ~hb), ("half B", hb), ("half B without seq_27", hb & (scene != f.FAST))):
            stat = fd.half_stat(scene[sel]); g, gr = stat((z - ref)[sel]); ci = fd.bs.ci(gr)
            row[name] = {"G0": g, "ci95": ci, "zero_shot": float(z[sel].mean()), "reference": float(ref[sel].mean())}
        want = cmp_[f"half B {m} {grid}"]["G0"]
        assert abs(want["estimate"] - row["half B"]["G0"]) < 1e-6, (m, grid, want, row["half B"])
        res[f"{m} {grid}"] = row
        fmt = lambda d: f"{d['G0']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
        print(f"{m:12s} {grid:9s} {fmt(row['half B']):40s} {fmt(row['half B without seq_27']):34s} {fmt(row['half A'])}")
json.dump(res, open(sys.argv[0].replace(".py", ".json"), "w"), indent=1)
