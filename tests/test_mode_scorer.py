"""The content-aware mode scorer must actually read the decoded trajectories.

The default (context-only) scorer ranks modes before decoding, so its logits
cannot depend on the trajectories. The content scorer scores each mode from its
own rollout, so perturbing one mode's trajectory must move that mode's logit."""

import torch

from conftest import load_script

m4 = load_script("model4_mfp_train_eval.py")


def _model(content_scorer, k=6, obs=20, pred=60):
    torch.manual_seed(0)
    net = m4.MFPModel(obs_len=obs, pred_len=pred, k=k, content_scorer=content_scorer)
    net.eval()
    return net


def test_forward_shapes_both_modes():
    B, M, K, T, obs = 2, 3, 6, 60, 20
    obs_rel = torch.randn(B, M, 2, obs)
    am = torch.ones(B, M, dtype=torch.bool)
    for cs in (False, True):
        pred_k, logits = _model(cs, k=K)(obs_rel, am)
        assert pred_k.shape == (B, M, K, 2, T)
        assert logits.shape == (B, M, K)


def test_traj_features_depend_on_trajectory():
    """_traj_features is a pure function of the decoded displacements."""
    net = _model(True, k=4)
    B, M, K, T = 1, 1, 4, 60
    rel = torch.zeros(B, M, K, 2, T)
    rel[0, 0, 1] = 0.3          # mode 1 moves; others are static
    feats = net._traj_features(rel)
    assert feats.shape[:3] == (B, M, K)
    moving = feats[0, 0, 1]
    static = feats[0, 0, 0]
    assert not torch.allclose(moving, static)
    # the three static modes share identical features
    assert torch.allclose(feats[0, 0, 0], feats[0, 0, 2])
    assert torch.allclose(feats[0, 0, 0], feats[0, 0, 3])


def test_content_scorer_logit_tracks_its_own_mode():
    """Perturbing ONE mode's trajectory changes ONLY that mode's logit.

    We drive the scorer directly (bypassing the LSTM decoder) by exercising the
    exact path forward() uses: score_in = cat(h0, traj_enc(_traj_features))."""
    net = _model(True, k=5)
    B, M, K = 1, 1, 5
    torch.manual_seed(1)
    h0 = torch.randn(B, M, K, net.mode_score[0].in_features - net.traj_enc[-1].out_features)
    pred_k = torch.randn(B, M, K, 2, 60) * 0.1

    def logits_for(traj):
        feats = net.traj_enc(net._traj_features(traj))
        return net.mode_score(torch.cat([h0, feats], dim=-1)).squeeze(-1)

    base = logits_for(pred_k)
    bumped = pred_k.clone()
    bumped[0, 0, 2] += 1.0                     # perturb only mode 2's trajectory
    after = logits_for(bumped)

    delta = (after - base).abs()[0, 0]
    others = torch.cat([delta[:2], delta[3:]])
    assert delta[2] > 1e-4                      # mode 2's logit moved
    assert torch.allclose(others, torch.zeros_like(others), atol=1e-6)  # others fixed


def test_context_only_scorer_ignores_trajectory():
    """The default scorer must NOT depend on the decoded trajectory at all."""
    net = _model(False, k=5)
    B, M, K = 1, 1, 5
    torch.manual_seed(2)
    h0 = torch.randn(B, M, K, net.mode_score[0].in_features)
    base = net.mode_score(h0).squeeze(-1)
    # context-only scorer takes h0 alone; there is no trajectory input to change.
    assert base.shape == (B, M, K)
    assert net.mode_score[0].in_features == net.init_h.out_features // net.k
