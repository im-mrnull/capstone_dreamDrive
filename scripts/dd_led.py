"""Adapter: run upstream LED on canonical IDD-3D sequences.

LED -- Leapfrog Diffusion Model for Stochastic Trajectory Prediction (Mao et al.,
CVPR 2023; https://github.com/MediaBrain-SJTU/LED). Two networks:

  * core denoiser -- a DDPM noise-prediction transformer over the future,
    conditioned on a social encoding of the past (stage 1, standard DDPM loss);
  * leapfrog initializer -- predicts a mean, a scale and K normalized samples
    that land the reverse process at step tau=NUM_TAU, so only the last few
    denoising steps are run (stage 2, core frozen).

The upstream repo is NBA-specific and ships only stage 2 (it loads a pretrained
NBA core, and its NBA data loader is not in the repo). This adapter reuses the
upstream modules unchanged except for the shape constants that are hard-coded to
NBA (11 agents, 10 past / 20 future frames, K split 10+10), and builds the batch
directly from canonical sequences so the eval population is the shared one, as
for AgentFormer (dd_agentformer.py). Runs at the full 10 Hz (20 past / 60 future):
LED attends over agents with one token each and over time per agent, so unlike
AgentFormer it has no agent x time token blow-up.

Input features follow upstream (abs, rel, vel per past frame -> 6 dims), except
"abs" is centered on the scene's agent centroid at the last observed frame
instead of the NBA court center (IDD-3D world frames have arbitrary origins).
Deployable most-likely = the initializer's mean estimate denoised with zero
noise (upstream p_sample_loop_mean, deterministic); oracle best-of-K = min over
the K leapfrog samples.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dd_agentformer import _fill_copyforward

REPO = Path(__file__).resolve().parent.parent
LED_REPO = REPO / "third_party" / "led"


def add_led_to_path(led_repo: str | Path = LED_REPO) -> Path:
    """Put the LED repo on sys.path so `import models...` resolves."""
    p = Path(led_repo)
    if not (p / "models" / "model_diffusion.py").exists():
        raise FileNotFoundError(
            f"LED repo not found at {p}. Clone it: git clone --depth 1 "
            f"https://github.com/MediaBrain-SJTU/LED.git third_party/led")
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


def build_models(past_len: int, fut_len: int, k: int):
    """(core denoiser, leapfrog initializer) with IDD-3D shapes."""
    add_led_to_path()
    from models.layers import PositionalEncoding
    from models.model_diffusion import TransformerDenoisingModel
    from models.model_led_initializer import LEDInitializer

    class CoreDenoiser(TransformerDenoisingModel):
        """Upstream core with the NBA constants made parametric: the social
        encoder's input is past_len*6 (upstream hard-codes 60 = 10 frames x 6),
        the positional table covers fut_len (upstream 24), and the batched
        generate path takes any K and T (upstream hard-codes 10 x 20)."""

        def __init__(self, context_dim: int = 256, tf_layer: int = 2):
            super().__init__(context_dim, tf_layer)
            self.encoder_context.encode_past = nn.Linear(past_len * 6, 256, bias=False)
            self.pos_emb = PositionalEncoding(d_model=2 * context_dim, dropout=0.1,
                                              max_len=max(24, fut_len))
            self.d_model = 2 * context_dim

        def generate_accelerate(self, x, beta, context, mask):
            B, K, T, _ = x.shape
            beta = beta.view(beta.size(0), 1, 1)
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
            context = self.encoder_context(context, mask)
            time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
            ctx_emb = torch.cat([time_emb, context], dim=-1).repeat(1, K, 1).unsqueeze(2)
            h = self.concat1.batch_generate(ctx_emb, x).contiguous().view(-1, T, self.d_model)
            h = self.pos_emb(h.permute(1, 0, 2))
            trans = self.transformer_encoder(h).permute(1, 0, 2).contiguous().view(B, K, T, self.d_model)
            trans = self.concat3.batch_generate(ctx_emb, trans)
            trans = self.concat4.batch_generate(ctx_emb, trans)
            return self.linear.batch_generate(ctx_emb, trans)

    core = CoreDenoiser()
    init = LEDInitializer(t_h=past_len, d_h=6, t_f=fut_len, d_f=2, k_pred=k)
    return core, init


class Diffusion:
    """Upstream LED's linear-beta DDPM schedule and sampling steps, made
    mask-aware (IDD-3D futures have per-step validity) and K-generic."""

    def __init__(self, device, steps: int = 100, beta_start: float = 1e-4,
                 beta_end: float = 5e-2):
        self.n_steps = steps
        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alphas = 1 - self.betas
        self.alphas_bar_sqrt = torch.sqrt(torch.cumprod(self.alphas, 0))
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - torch.cumprod(self.alphas, 0))

    @staticmethod
    def _extract(v, t, like):
        return v.gather(0, t).reshape([t.shape[0]] + [1] * (like.dim() - 1))

    def noise_estimation_loss(self, core, past, fut, fut_mask, mask):
        """Stage 1 (DDPM epsilon loss, upstream's antithetic t pairing), averaged
        over valid future steps only."""
        n = past.shape[0]
        t = torch.randint(0, self.n_steps, (n // 2 + 1,), device=past.device)
        t = torch.cat([t, self.n_steps - t - 1])[:n]
        a = self._extract(self.alphas_bar_sqrt, t, fut)
        am1 = self._extract(self.one_minus_alphas_bar_sqrt, t, fut)
        beta = self._extract(self.betas, t, fut)
        e = torch.randn_like(fut)
        out = core(fut * a + e * am1, beta, past, mask)
        w = fut_mask.unsqueeze(-1).float()
        return ((e - out).square() * w).sum() / (w.sum() * 2).clamp_min(1.0)

    def p_sample_accelerate(self, core, past, mask, cur_y, t: int, noise_scale: float):
        """One reverse step on (N,K,T,2) samples (upstream uses noise_scale=1e-5)."""
        tt = torch.tensor([t], device=past.device)
        eps_factor = (1 - self._extract(self.alphas, tt, cur_y)) / \
            self._extract(self.one_minus_alphas_bar_sqrt, tt, cur_y)
        beta = self._extract(self.betas, tt.repeat(past.shape[0]), cur_y[:, 0])
        eps = core.generate_accelerate(cur_y, beta, past, mask)
        mean = (cur_y - eps_factor * eps) / self._extract(self.alphas, tt, cur_y).sqrt()
        sigma = self._extract(self.betas, tt, cur_y).sqrt()
        return mean + sigma * torch.randn_like(cur_y) * noise_scale

    def denoise(self, core, past, mask, loc, n_tau: int = 5, noise_scale: float = 1e-5,
                checkpoint: bool = False):
        """Leapfrog: run only the last n_tau reverse steps from loc (N,K,T,2).

        checkpoint=True (stage-2 training) recomputes each step's activations in
        the backward pass instead of storing all n_tau of them: at 10 Hz with
        K=20 those activations are ~45 GB for an 8-sequence batch."""
        y = loc
        for t in reversed(range(n_tau)):
            if checkpoint:
                y = torch.utils.checkpoint.checkpoint(
                    self.p_sample_accelerate, core, past, mask, y, t, noise_scale,
                    use_reentrant=False)
            else:
                y = self.p_sample_accelerate(core, past, mask, y, t, noise_scale)
        return y


def leapfrog_loc(init, past, mask):
    """Initializer output -> starting samples (N,K,T,2), mean (N,T,2), log-var (N,1)
    (upstream's sample normalization, verbatim)."""
    samp, mean, logvar = init(past, mask)
    samp = torch.exp(logvar / 2)[..., None, None] * samp / \
        samp.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
    return samp + mean[:, None], mean, logvar


def initializer_loss(gen, fut, fut_mask, logvar):
    """Stage 2: best-of-K distance (upstream temporal reweight, early steps
    weighted more) + the uncertainty term, both over valid steps only.

    Deviation from upstream: upstream multiplies exp(-logvar) of shape (N,1) by a
    per-agent distance of shape (N,), which broadcasts to an N x N outer product.
    Here each agent's uncertainty pairs with its own distance, as in the paper."""
    T = fut.shape[-2]
    w_t = (T + 1 - torch.arange(1, T + 1, device=fut.device, dtype=fut.dtype)) / (T / 2)
    fm = fut_mask.float()[:, None, :]                                  # (N,1,T)
    nv = fm.sum(-1).clamp_min(1.0)                                     # (N,1)
    d = (gen - fut[:, None]).norm(dim=-1)                              # (N,K,T)
    has = fut_mask.any(dim=-1)
    loss_dist = ((d * w_t * fm).sum(-1) / nv).min(dim=1).values[has].mean()
    mean_d = ((d * fm).sum(-1) / nv).mean(dim=1)                       # (N,)
    lv = logvar.squeeze(-1)
    loss_unc = (torch.exp(-lv) * mean_d + lv)[has].mean()
    return loss_dist * 50 + loss_unc, loss_dist, loss_unc


def build_batch(items, traj_scale: float, device) -> dict:
    """List of canonical sequences (obs (M,2,To), pred (M,2,Tf), fut_mask (M,Tf),
    agent_type (M,)) -> one LED batch over all their agents.

    past (N,To,6) = [abs (scene-centered), rel (to own last obs), vel] / traj_scale;
    fut (N,Tf,2) relative to own last obs / traj_scale (masked steps copy-forward
    filled, never scored); mask (N,N) block-diagonal: agents only see their scene."""
    pasts, futs, fms, lasts, types, sizes = [], [], [], [], [], []
    for obs, pred, fm, at in items:
        o = np.transpose(obs, (0, 2, 1)).astype(np.float32)               # (M,To,2)
        p = np.transpose(pred, (0, 2, 1)).astype(np.float32)              # (M,Tf,2)
        last = o[:, -1:]                                                  # (M,1,2)
        p = _fill_copyforward(p, fm.astype(bool), last[:, 0])
        center = last[:, 0].mean(axis=0)
        rel = (o - last) / traj_scale
        vel = np.concatenate([rel[:, 1:] - rel[:, :-1], np.zeros_like(rel[:, -1:])], axis=1)
        pasts.append(np.concatenate([(o - center) / traj_scale, rel, vel], axis=-1))
        futs.append((p - last) / traj_scale)
        fms.append(fm.astype(bool)); lasts.append(last[:, 0])
        types.append(np.asarray(at, dtype=object)); sizes.append(o.shape[0])
    n = int(sum(sizes))
    mask = np.zeros((n, n), dtype=np.float32)
    s = 0
    for m in sizes:
        mask[s:s + m, s:s + m] = 1.0
        s += m
    t = lambda a: torch.from_numpy(np.concatenate(a)).to(device)
    return {"past": t(pasts), "fut": t(futs), "fut_mask": t(fms), "last": t(lasts),
            "mask": torch.from_numpy(mask).to(device),
            "agent_type": np.concatenate(types), "sizes": sizes}


@torch.no_grad()
def predict_batch(core, init, diff: Diffusion, batch: dict, traj_scale: float, n_tau: int = 5):
    """Meters, world frame: most_likely (N,T,2) and samples (N,K,T,2)."""
    loc, mean, _ = leapfrog_loc(init, batch["past"], batch["mask"])
    samp = diff.denoise(core, batch["past"], batch["mask"], loc, n_tau)
    ml = diff.denoise(core, batch["past"], batch["mask"], mean[:, None], n_tau, noise_scale=0.0)[:, 0]
    last = batch["last"]
    return ml * traj_scale + last[:, None], samp * traj_scale + last[:, None, None]
