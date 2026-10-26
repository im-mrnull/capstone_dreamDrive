"""Adapter: feed canonical IDD-3D sequences straight into upstream AgentFormer.

Benchmark v2 requires every model to be scored on the SAME population (the agents
of outputs/canonical_world_obs20_pred60.npz, partitioned by the frozen scene
split, with per-timestep fut_mask). AgentFormer's own data_generator slides a
window over every frame of every scene and filters agents by min_past/min_future
frames — a completely different population. So instead of round-tripping through
its loader, we build AgentFormer's `data` dict directly from one canonical
sequence (a seq_start_end group of co-observed agents), which makes the eval
population identical to the canonical one *by construction*.

This is the map-free path: no map/lane channel, no per-agent heading (IDD-3D is
lane-free and this repo's parsed data has no yaw). The base CVAE is configured
with input_type=['scene_norm','vel'], use_map=False, learn_prior=False, so the
prior is N(0,I): the deployable "most-likely" output is the z=0 rollout and the
oracle best-of-K is the min over K prior samples — matching the scope confirmed
for this integration.

Coordinates: AgentFormer works in scene-normalized units of world/traj_scale and
returns predictions in the same units, so metrics multiply the decoder output by
traj_scale to get meters. Reuses dd_data's masked ADE/FDE so AgentFormer is
scored by the identical metric code as every other benchmark-v2 model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
AF_REPO = REPO / "third_party" / "agentformer"


def add_af_to_path(af_repo: str | Path = AF_REPO) -> Path:
    """Put the AgentFormer repo on sys.path so `import model...`/`utils...`
    resolve (its modules use repo-root-relative absolute imports)."""
    p = Path(af_repo)
    if not (p / "model" / "agentformer.py").exists():
        raise FileNotFoundError(
            f"AgentFormer repo not found/patched at {p}. Clone it and run "
            f"scripts/patch_agentformer.py first.")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


def build_map_free_cfg(
    obs_len: int,
    pred_len: int,
    *,
    nz: int = 32,
    sample_k: int = 20,
    tf_model_dim: int = 256,
    tf_ff_dim: int = 512,
    tf_nhead: int = 8,
    tf_dropout: float = 0.1,
    nlayer: int = 2,
    out_mlp_dim=(512, 256),
    rand_rot_scene: bool = True,
    max_train_agent: int = 100,
    traj_scale: float = 10.0,
):
    """A minimal map-free AgentFormer (base CVAE) config as an EasyDict — no yaml,
    no results dirs, no dataset machinery. Standard-normal prior (learn_prior=
    False) so z=0 is the deployable most-likely mode. out_mlp_dim=None disables
    the output MLP (used by the tiny CPU smoke config)."""
    from easydict import EasyDict

    om = None if out_mlp_dim is None else list(out_mlp_dim)
    return EasyDict(dict(
        # feature extractor
        past_frames=int(obs_len), future_frames=int(pred_len),
        min_past_frames=int(obs_len), min_future_frames=1,
        motion_dim=2, forecast_dim=2, traj_scale=float(traj_scale),
        # model
        model_id="agentformer", tf_model_dim=int(tf_model_dim),
        tf_ff_dim=int(tf_ff_dim), tf_nhead=int(tf_nhead), tf_dropout=float(tf_dropout),
        input_type=["scene_norm", "vel"], fut_input_type=["scene_norm", "vel"],
        dec_input_type=[], pred_type="scene_norm", sn_out_type="norm",
        pos_concat=True, rand_rot_scene=bool(rand_rot_scene), discrete_rot=False,
        use_map=False, vel_heading=False, sn_out_heading=False,
        ar_detach=True, ar_train=True, max_agent_len=128,
        use_agent_enc=False, agent_enc_learn=False, agent_enc_shuffle=False,
        max_train_agent=int(max_train_agent), conn_dist=100000.0,
        context_encoder=EasyDict({"nlayer": int(nlayer)}),
        future_encoder=EasyDict({"nlayer": int(nlayer), "out_mlp_dim": om}),
        future_decoder=EasyDict({"nlayer": int(nlayer), "out_mlp_dim": om}),
        # VAE
        nz=int(nz), z_type="gaussian", learn_prior=False, sample_k=int(sample_k),
        # loss
        loss_cfg=EasyDict({
            "mse": {"weight": 1.0},
            "kld": {"weight": 1.0, "min_clip": 2.0},
            "sample": {"weight": 1.0, "k": int(sample_k)},
        }),
    ))


def _fill_copyforward(fut: np.ndarray, fm: np.ndarray, last_obs: np.ndarray) -> np.ndarray:
    """Fill masked-out future steps by copying forward (AgentFormer's own
    preprocessor convention). These steps are masked out of both the training
    loss and the eval metric, so the fill only keeps the arrays finite/sane.
    fut (M,T,2), fm (M,T) bool, last_obs (M,2)."""
    out = fut.copy()
    M, T, _ = out.shape
    for i in range(M):
        prev = last_obs[i]
        for t in range(T):
            if fm[i, t]:
                prev = out[i, t]
            else:
                out[i, t] = prev
    return out


def seq_to_af_data(obs_abs: np.ndarray, pred_abs: np.ndarray, fut_mask: np.ndarray,
                   traj_scale: float) -> dict:
    """One canonical sequence -> an AgentFormer `data` dict.

    obs_abs (M,2,obs_len), pred_abs (M,2,pred_len) are world-frame meters;
    fut_mask (M,pred_len) is the per-timestep validity mask. Positions are scaled
    into world/traj_scale units (AgentFormer's internal convention)."""
    obs = np.transpose(obs_abs, (0, 2, 1)).astype(np.float32) / traj_scale   # (M,obs,2)
    fut = np.transpose(pred_abs, (0, 2, 1)).astype(np.float32) / traj_scale  # (M,pred,2)
    fm = fut_mask.astype(bool)
    fut = _fill_copyforward(fut, fm, obs[:, -1])
    M, obs_len, _ = obs.shape
    return {
        "pre_motion_3D": [torch.from_numpy(obs[i]).float() for i in range(M)],
        "fut_motion_3D": [torch.from_numpy(fut[i]).float() for i in range(M)],
        "fut_motion_mask": [torch.from_numpy(fm[i].astype(np.float32)) for i in range(M)],
        "pre_motion_mask": [torch.ones(obs_len) for _ in range(M)],
        "heading": None,
        "traj_scale": float(traj_scale),
        "scene_map": None,
    }


@torch.no_grad()
def predict_seq(model, data: dict, nz: int, sample_k: int, traj_scale: float):
    """Run one scene through AgentFormer and return predictions in METERS:
      most_likely (M, T, 2)  — deployable z=0 rollout;
      samples     (M, K, T, 2) — K prior draws (oracle best-of-K).
    Shares the encoded context between the two decodes."""
    device = model.device
    model.set_data(data)
    model.context_encoder(model.data)  # populate context_enc + agent_context once
    M = model.data["agent_num"]

    z0 = torch.zeros(M, nz, device=device)
    model.future_decoder(model.data, mode="infer", sample_num=1, z=z0)
    ml = model.data["infer_dec_motion"][:, 0] * traj_scale            # (M,T,2)

    samp, _ = model.inference(mode="infer", sample_num=sample_k)      # reuses context_enc
    samp = samp * traj_scale                                          # (M,K,T,2)
    return ml, samp
