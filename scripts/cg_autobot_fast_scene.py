"""AutoBot fast-scene check (chaos gap, development only).

Open question from Phase 4b (docs/wiki/chaos-gap.md, dump check of 2026-11-19): fine-tuned AutoBot-Joint is worse
than zero-shot on IDD val half B at every k in both schedules, and one scene drives it (20220118103308_seq_27:
62 agent-windows in 7 windows, all moving, about 8 m/s). Two candidate explanations, written down before this
script was first run:

  H1 coverage. IDD train has few fast agents, so fine-tuning on it erodes the fast extrapolation learned on AV2.
     Expected if H1 holds: fast agents are much rarer in IDD train than in AV2-matched train; the fine-tuned to
     zero-shot error ratio grows with speed across IDD val, not only on seq_27; and models fine-tuned on the same
     data (MFP, AgentFormer, LED) show some of the same speed dependence.
  H2 frame. The AutoBot adapter (scripts/model9_autobot_train_eval.py, scene_inputs) puts each window in one scene
     frame, centred on the agents' mean last-observed position, with absolute rather than agent-relative
     coordinates. A fast, spread-out window then feeds and asks for coordinates beyond what IDD training shows.
     Expected if H2 holds: seq_27's frame coordinates lie beyond most of IDD train's; at a given speed the error
     grows with distance from the frame origin; and re-centring the frame on each agent (part C) lowers the
     fine-tuned runs' seq_27 error clearly more than the zero-shot runs'.
  Both can hold, or neither.

Parts
  A. Data. Per agent-window: speed over the last observed second; displacement at the last valid future step;
     frame radius (distance from the window's frame origin at the last observed step, and the largest over the
     observed and valid future steps). For IDD train (72 scenes, the k = 72 set), every option-1 fine-tuning
     draw, AV2-matched train (the zero-shot models' training set) and IDD val.
  B. Errors by speed over all of IDD val (both halves), from the saved dumps: AutoBot, MFP, AgentFormer, LED and
     CV; zero-shot (3 AV2-matched seeds) and fine-tuned (option 1 k05 / k72, option 2 k05 / k64, both arms,
     3 draws). Most-likely ADE (MFP top-1), and for agents whose true displacement exceeds 5 m: travel (predicted
     / true displacement at the last valid step) and heading error (angle between the two).
  C. Counterfactual, AutoBot only, on the CPU: every IDD val window predicted again
     (1) as trained: origin = mean last-observed position (must reproduce the dumps);
     (2) agent-centred: once per agent, origin at that agent's last observed position; that agent's row of the
         scene's most probable mode is kept;
     (3) rotated: the window rotated by 90 / 180 / 270 degrees about its origin, the prediction rotated back
         (an orientation check; training used random rotations).

Bins fixed in advance: speed [0, 0.5), [0.5, 3), [3, 6), [6, inf) m/s; radius [0, 10), [10, 20), [20, inf) m.
Only IDD val (halves A and B), IDD train and AV2 rows are read; the frozen IDD test is never read (asserted).
Descriptive only: no CIs and nothing selected on these numbers.

Usage:  python scripts/cg_autobot_fast_scene.py [--skip-c]
Output: outputs/chaos_gap/dev/autobot_fast_scene.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cg_ft_diagnose as fd                                               # noqa: E402
import model9_autobot_train_eval as m9                                    # noqa: E402
from dd_split import load_splits                                          # noqa: E402

REPO = fd.REPO
DEV = fd.DEV
IDD = REPO / "outputs" / "canonical_world_obs20_pred60.npz"
AV2M = REPO / "outputs" / "chaos_gap" / "av2_matched_dev.npz"
AV2M_SPLITS = REPO / "splits" / "chaos_gap" / "av2_matched_dev.json"
FT_SPLITS = REPO / "splits" / "chaos_gap" / "ft"
FAST = "20220118103308_seq_27"
SPEED_BINS = (0.5, 3.0, 6.0)                       # -> [0, .5), [.5, 3), [3, 6), [6, inf)
SPEED_LABELS = ("0-0.5", "0.5-3", "3-6", "6+")
RADIUS_BINS = (10.0, 20.0)
RADIUS_LABELS = ("0-10", "10-20", "20+")
MOVE_M = 5.0                                       # travel / heading only for true displacement above this
GROUPS = [("zero-shot", None, None, None),
          ("opt1 k05 pre", "ft_budget", "05", "pre"), ("opt1 k05 scr", "ft_budget", "05", "scr"),
          ("opt1 k72 pre", "ft_budget", "72", "pre"), ("opt1 k72 scr", "ft_budget", "72", "scr"),
          ("opt2 k05 pre", "ft_stop", "05", "pre"), ("opt2 k05 scr", "ft_stop", "05", "scr"),
          ("opt2 k64 pre", "ft_stop", "64", "pre"), ("opt2 k64 scr", "ft_stop", "64", "scr")]
MODELS = ("autobot", "mfp", "agentformer", "led")
FROZEN = {s for s, v in load_splits(REPO / "splits" / "scene_splits.json").items() if v == "test"}


# ---------------------------------------------------------------- per-row geometry
def last_valid(fm: np.ndarray) -> np.ndarray:
    return np.where(fm, np.arange(fm.shape[1]), -1).max(1)


def window_rows(npz: dict, mapping: dict, wanted: set) -> dict:
    """Per agent-window geometry for the windows whose scene's split is in `wanted`."""
    out = {k: [] for k in ("scene", "t0", "aid", "row", "speed", "disp", "r_last", "r_max")}
    for q, (a, b) in enumerate(npz["seq_start_end"]):
        sid = str(npz["scene_id"][q])
        if mapping.get(sid) not in wanted:
            continue
        assert sid not in FROZEN, f"{sid} is a frozen-test scene"
        obs = npz["obs_traj"][a:b].astype(float)                          # (M, 2, 20)
        pred = npz["pred_traj"][a:b].astype(float)                        # (M, 2, 60)
        fm = npz["fut_mask"][a:b].astype(bool)
        last = obs[:, :, -1]
        origin = last.mean(0)                                             # every row of a window is a real agent
        L = last_valid(fm)
        r_obs = np.linalg.norm(obs - origin[None, :, None], axis=1).max(1)
        r_fut = np.where(fm, np.linalg.norm(pred - origin[None, :, None], axis=1), 0.0).max(1)
        out["scene"] += [sid] * (b - a)
        out["t0"] += [int(npz["t0"][q])] * (b - a)
        out["aid"] += [int(x) for x in npz["agent_id"][a:b]]
        out["row"] += list(range(a, b))
        out["speed"] += list(np.linalg.norm(obs[:, :, -1] - obs[:, :, -11], axis=1))
        out["disp"] += list(np.linalg.norm(pred[np.arange(b - a), :, L] - last, axis=1))
        out["r_last"] += list(np.linalg.norm(last - origin, axis=1))
        out["r_max"] += list(np.maximum(r_obs, r_fut))
    return {k: np.asarray(v) for k, v in out.items()}


def dist(v: np.ndarray) -> dict:
    return {"p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max())}


def part_a(idd: dict, ft_map: dict, val: dict) -> dict:
    av2 = dict(np.load(AV2M, allow_pickle=True))
    sets = {"IDD train (k72)": window_rows(idd, ft_map, {"train"}),
            "AV2-matched train": window_rows(av2, load_splits(AV2M_SPLITS), {"train"}),
            "IDD val (both halves)": val}
    fast = val["scene"] == FAST
    ref = {"speed": float(np.median(val["speed"][fast])), "r_max": float(np.median(val["r_max"][fast])),
           "disp": float(np.median(val["disp"][fast]))}
    out = {"seq_27": {"n": int(fast.sum()), "median": ref,
                      "speed": dist(val["speed"][fast]), "r_max": dist(val["r_max"][fast]),
                      "disp": dist(val["disp"][fast])}, "sets": {}, "ft_draws": {}}
    for name, s in sets.items():
        out["sets"][name] = {
            "n": int(len(s["speed"])), "n_scenes": int(len(set(s["scene"]))),
            "speed": dist(s["speed"]), "disp": dist(s["disp"]), "r_last": dist(s["r_last"]), "r_max": dist(s["r_max"]),
            "share_speed_ge_3": float((s["speed"] >= 3).mean()), "share_speed_ge_6": float((s["speed"] >= 6).mean()),
            "share_speed_ge_seq27_median": float((s["speed"] >= ref["speed"]).mean()),
            "share_disp_ge_seq27_median": float((s["disp"] >= ref["disp"]).mean()),
            "share_r_max_ge_seq27_median": float((s["r_max"] >= ref["r_max"]).mean()),
        }
    for k in fd.KS:
        for r in range(3):
            s = window_rows(idd, load_splits(FT_SPLITS / f"idd_k{k}_r{r}_dev.json"), {"train"})
            out["ft_draws"][f"k{k}_r{r}"] = {"n": int(len(s["speed"])), "n_speed_ge_6": int((s["speed"] >= 6).sum()),
                                             "n_speed_ge_seq27_median": int((s["speed"] >= ref["speed"]).sum()),
                                             "n_r_max_ge_seq27_median": int((s["r_max"] >= ref["r_max"]).sum())}
    return out


# ---------------------------------------------------------------- dumps aligned to IDD val
def metrics(pred: np.ndarray, gt: np.ndarray, fm: np.ndarray, last_obs: np.ndarray) -> dict:
    err = np.where(fm, np.linalg.norm(pred - gt, axis=-1), 0.0)          # invalid future steps never count
    ade = err.sum(1) / fm.sum(1)
    L, n = last_valid(fm), np.arange(len(fm))
    D, d = gt[n, L] - last_obs, pred[n, L] - last_obs
    nD, nd = np.linalg.norm(D, axis=1), np.linalg.norm(d, axis=1)
    cos = (D * d).sum(1) / np.maximum(nD * nd, 1e-12)
    return {"ade": ade, "travel": nd / np.maximum(nD, 1e-12), "heading": np.degrees(np.arccos(np.clip(cos, -1, 1))),
            "nD": nD}


def load_aligned(path: Path, split: str | None, pos: dict, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A flat dump's pred, gt and fut_mask in IDD-val order (rows outside IDD val dropped)."""
    p = np.load(path, allow_pickle=True)
    order = fd.dump_order(split)
    assert len(order) == len(p["pred"]), path.name
    pred, gt, fm = np.full((n, 60, 2), np.nan), np.full((n, 60, 2), np.nan), np.zeros((n, 60), bool)
    for j, (s, t0, aid, _) in enumerate(order):
        i = pos.get((s, t0, aid))
        if i is not None:
            pred[i], gt[i], fm[i] = p["pred"][j], p["gt"][j], p["fut_mask"][j]
    assert not np.isnan(pred).any(), f"{path.name}: IDD val rows missing"
    return pred, gt, fm


def runs_of(model: str, grid: str | None, k: str | None, arm: str | None) -> list[tuple[Path, str | None]]:
    if grid is None:
        return [(DEV / f, None) for f in fd.ZS[model]]
    out = []
    for r in range(3):
        t = f"{model}_k{k}_r{r}_{arm}"
        meta = json.loads((DEV / grid / f"{t}.json").read_text(encoding="utf-8"))
        out.append((DEV / grid / f"{t}_preds.npz", str(REPO / meta["splits_file"])))
    return out


def summarize(per_run: list[dict], sel: np.ndarray) -> dict:
    mv = sel & (per_run[0]["nD"] > MOVE_M)
    if np.isnan(per_run[0]["travel"]).all():                              # ADE only (a dump in another frame)
        mv = np.zeros_like(mv)
    return {"n": int(sel.sum()), "n_moving_5m": int(mv.sum()),
            "ade": float(np.mean([r["ade"][sel].mean() for r in per_run])) if sel.any() else None,
            "travel_median": float(np.mean([np.median(r["travel"][mv]) for r in per_run])) if mv.any() else None,
            "heading_median_deg": float(np.mean([np.median(r["heading"][mv]) for r in per_run])) if mv.any() else None,
            "share_heading_gt_45": float(np.mean([(r["heading"][mv] > 45).mean() for r in per_run])) if mv.any() else None}


def selections(val: dict, half_b: np.ndarray) -> dict:
    sb = np.digitize(val["speed"], SPEED_BINS)
    rb = np.digitize(val["r_last"], RADIUS_BINS)
    fast = val["scene"] == FAST
    sel = {"all": np.ones(len(sb), bool), "half A": ~half_b, "half B": half_b, "seq_27": fast,
           "half B without seq_27": half_b & ~fast}
    for i, lab in enumerate(SPEED_LABELS):
        sel[f"speed {lab}"] = sb == i
        sel[f"speed {lab} without seq_27"] = (sb == i) & ~fast
    for slow, cond in (("speed < 3", val["speed"] < 3), ("speed >= 3", val["speed"] >= 3)):
        for i, lab in enumerate(RADIUS_LABELS):
            sel[f"{slow}, radius {lab}"] = cond & (rb == i)
            sel[f"{slow}, radius {lab} without seq_27"] = cond & (rb == i) & ~fast
    return sel


def part_b(val: dict, half_b: np.ndarray, pos: dict, gt: np.ndarray, fm: np.ndarray, last_obs: np.ndarray) -> dict:
    n = len(val["speed"])
    sel = selections(val, half_b)
    fast_scenes = {}
    for s in sorted(set(val["scene"][val["speed"] >= 6])):
        fast_scenes[s] = int(((val["scene"] == s) & (val["speed"] >= 6)).sum())
    out = {"bins": {k: int(v.sum()) for k, v in sel.items()}, "speed_6plus_by_scene": fast_scenes, "models": {}}
    pred, gt_cv, fm_cv = load_aligned(DEV / "cv_matched_preds.npz", None, pos, n)   # stored in another frame:
    assert (fm_cv == fm).all()                                                       # ADE only, from its own gt
    cv = [metrics(pred, gt_cv, fm, gt_cv[:, 0])]
    cv[0]["travel"][:] = np.nan
    cv[0]["heading"][:] = np.nan
    cv[0]["nD"] = metrics(gt, gt, fm, last_obs)["nD"]
    out["models"]["cv"] = {"CV": {k: summarize(cv, v) for k, v in sel.items()}}
    for m in MODELS:
        out["models"][m] = {}
        for name, grid, k, arm in GROUPS:
            per_run = []
            for path, split in runs_of(m, grid, k, arm):
                pred, gt2, fm2 = load_aligned(path, split, pos, n)
                assert (fm2 == fm).all() and np.abs(np.where(fm[..., None], gt2 - gt, 0)).max() < 1e-3, path.name
                per_run.append(metrics(pred, gt, fm, last_obs))
            out["models"][m][name] = {k2: summarize(per_run, v) for k2, v in sel.items()}
    return out


# ---------------------------------------------------------------- part C: AutoBot re-predicted on the CPU
def load_autobot(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    AB = m9.load_autobot_joint()
    model = AB(k_attr=2, d_k=a["hidden_size"], _M=1, c=a["num_modes"], T=60, L_enc=a["num_encoder_layers"],
               dropout=a["dropout"], num_heads=a["tx_num_heads"], L_dec=a["num_decoder_layers"],
               tx_hidden_size=a["tx_hidden_size"], use_map_lanes=False, map_attr=0, num_agent_types=1,
               predict_yaw=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, float(a["pos_scale"])


@torch.no_grad()
def predict(model, obs_abs: torch.Tensor, origin: torch.Tensor, scale: float) -> torch.Tensor:
    """obs_abs (B, M, 2, T_obs), origin (B, 2) -> the scene's most probable mode in world coordinates (B, M, 60, 2).
    The same inputs as m9.scene_inputs (every slot a real agent), with the origin given."""
    B, M = obs_abs.shape[:2]
    obs = (obs_abs.permute(0, 3, 1, 2) - origin[:, None, None]) / scale            # (B, T, M, 2)
    x = torch.cat([obs, torch.ones(*obs.shape[:3], 1)], dim=-1)
    model._M = M - 1
    pred, mode_probs = model(x[:, :, 0], x[:, :, 1:], torch.ones((B, 1, 1)), torch.ones((B, M, 1)))
    modes = m9.world_modes(pred, origin, scale)                                    # (B, M, K, 2, T)
    ml = modes[torch.arange(B), :, mode_probs.argmax(1)]                           # (B, M, 2, T)
    return ml.permute(0, 1, 3, 2)


def rot(theta: float) -> torch.Tensor:
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)


def part_c(idd: dict, val: dict, half_b: np.ndarray, gt: np.ndarray, fm: np.ndarray, last_obs: np.ndarray,
           pos: dict) -> dict:
    n = len(val["speed"])
    windows = {}                                                          # (scene, t0) -> canonical rows
    for i in range(n):
        windows.setdefault((val["scene"][i], int(val["t0"][i])), []).append(i)
    sel = selections(val, half_b)
    keep = ["all", "half A", "half B", "seq_27", "half B without seq_27"] + [f"speed {lab}" for lab in SPEED_LABELS] \
        + [f"speed {lab} without seq_27" for lab in SPEED_LABELS]
    out = {"groups": {}, "reproduction_max_abs_m": {}}
    for name, grid, k, arm in GROUPS:
        res = {v: [] for v in ("as trained", "agent-centred", "rot90", "rot180", "rot270")}
        for path, split in runs_of("autobot", grid, k, arm):
            ckpt = path.with_name(path.name.replace("_eval_preds.npz", ".pt").replace("_preds.npz", ".pt"))
            model, scale = load_autobot(ckpt)
            preds = {v: np.full((n, 60, 2), np.nan) for v in res}
            for (s, t0), rows_i in windows.items():
                crow = val["row"][rows_i]
                obs = torch.from_numpy(idd["obs_traj"][crow].astype(np.float32))[None]       # (1, M, 2, 20)
                M = obs.shape[1]
                origin = obs[:, :, :, -1].mean(1)                                            # (1, 2)
                preds["as trained"][rows_i] = predict(model, obs, origin, scale)[0].numpy()
                ac = predict(model, obs.expand(M, -1, -1, -1).contiguous(), obs[0, :, :, -1], scale)
                preds["agent-centred"][rows_i] = ac[torch.arange(M), torch.arange(M)].numpy()
                for deg in (90, 180, 270):
                    R = rot(math.radians(deg))
                    o = origin[0][None, :, None]
                    obs_r = (torch.einsum("ij,bmjt->bmit", R, obs - o[None]) + o[None])
                    pr = predict(model, obs_r, origin, scale)[0]                             # (M, 60, 2)
                    back = torch.einsum("ji,mtj->mti", R, pr - origin[0]) + origin[0]
                    preds[f"rot{deg}"][rows_i] = back.numpy()
            ref, _, _ = load_aligned(path, split, pos, n)
            out["reproduction_max_abs_m"][f"{name} | {path.name}"] = float(np.abs(preds["as trained"] - ref).max())
            for v in res:
                res[v].append(metrics(preds[v], gt, fm, last_obs))
        out["groups"][name] = {v: {kk: summarize(r, sel[kk]) for kk in keep} for v, r in res.items()}
        print(f"  C {name}: seq_27 ADE as trained {out['groups'][name]['as trained']['seq_27']['ade']:.2f} | "
              f"agent-centred {out['groups'][name]['agent-centred']['seq_27']['ade']:.2f} | rot90/180/270 "
              + " / ".join(f"{out['groups'][name][f'rot{d}']['seq_27']['ade']:.2f}" for d in (90, 180, 270)),
              flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-c", action="store_true")
    ap.add_argument("--out", default=str(DEV / "autobot_fast_scene.json"))
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    idd = dict(np.load(IDD, allow_pickle=True))
    ft_map = load_splits(fd.SPLIT)                                        # train = IDD train, val = half A, test = half B
    _, order = fd.vb.canonical_order(IDD, fd.SPLIT)                       # half A rows then half B rows
    val = window_rows(idd, ft_map, {"val", "test"})
    # put val rows in the diagnose order (half A then half B)
    idx = {(s, t, a): i for i, (s, t, a) in enumerate(zip(val["scene"], val["t0"], val["aid"]))}
    perm = np.array([idx[(s, t, a)] for s, t, a, _ in order])
    val = {k: v[perm] for k, v in val.items()}
    half_b = np.array([ft_map[s] == "test" for s in val["scene"]])
    pos = {(s, int(t), int(a)): i for i, (s, t, a) in enumerate(zip(val["scene"], val["t0"], val["aid"]))}
    gt = idd["pred_traj"][val["row"]].transpose(0, 2, 1).astype(float)
    fm = idd["fut_mask"][val["row"]].astype(bool)
    last_obs = idd["obs_traj"][val["row"]][:, :, -1].astype(float)

    report = {"note": "development only (IDD val halves A and B); descriptive; see the docstring for the hypotheses",
              "fast_scene": FAST, "speed_bins": SPEED_LABELS, "radius_bins": RADIUS_LABELS}
    print("A. data ...", flush=True)
    report["A"] = part_a(idd, ft_map, val)
    print("B. dumps ...", flush=True)
    report["B"] = part_b(val, half_b, pos, gt, fm, last_obs)
    if not args.skip_c:
        print("C. counterfactual (CPU) ...", flush=True)
        report["C"] = part_c(idd, val, half_b, gt, fm, last_obs, pos)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
