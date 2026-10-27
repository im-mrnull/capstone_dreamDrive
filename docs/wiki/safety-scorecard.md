# Safety Scorecard — which predictor best supports safe navigation?

## Status: 📌 Rule pre-registered (2026-10-27) — results pending

Mean ADE ranks forecasters, not safety. This scorecard asks a planner's question instead: does the
predictor foresee the agents that actually come close to our car, without flagging everyone,
especially pedestrians and two-wheelers, fast enough for 10 Hz? The rule below was committed
**before** any scorecard number was computed (the result JSON records this commit as
`rule_commit`), so the pick cannot be tuned to the outcome.

## Candidates — 12 packages, seeds 0–2, same 993 val / 2,177 test canonical windows

- **Single trajectory + band (8):** CV, MLP, ST-GAT, map-free transformer, K-mode top-1,
  AgentFormer most-likely, LED mean-denoised, Trajectron++ most-likely.
- **Multi-future set (4):** LED, AgentFormer, Trajectron++ (first 10 of their 20 samples) and the
  K-mode decoder (its 10 modes); K = 20 is a reported sensitivity check, not used for the pick.
- **Band:** per-step split-conformal radius r_t at 90% (α = 0.1), per class (Mondrian) for classes
  with ≥ 30 val agents, marginal otherwise, **calibrated on val**, applied on test. A set's region
  is the union of discs of radius r_t around its samples; its score is the min-over-samples error.

## Definitions

- Ego path: `outputs/ego_trajectories.csv`; for a window starting at frame t0, future step k is
  frame t0 + 20 + k (k = 0…59).
- **Conflict:** the agent's true position comes within **5 m** of the ego's actual position at the
  same valid future step.
- **Danger score** D = min over valid steps of (d_t − 5 m) / r_t, where d_t is the distance from
  the ego to the nearest predicted position (sample) at step t. The agent is flagged at band scale
  λ when D ≤ λ (λ = 1: the calibrated band; λ = 0: the bare prediction).

## Decision rule

1. **Primary P:** conflict-detection ROC-AUC on test (P(D_conflict < D_no-conflict), ties ½),
   3-seed mean, with a seed-pooled scene bootstrap (2,000 replicates: test scenes resampled and
   shared by all packages, each package's seeds resampled with replacement).
2. **Leader** = highest P. **Tied set** = packages whose paired ΔP vs the leader has a 95% CI
   that includes 0.
3. **Tie-break 1 (S1):** within the tied set, keep the packages tied (same paired test) with the
   best on S1 = the same AUC over vulnerable road users only (Pedestrian, Child, MotorcyleRider,
   ScooterRider, BicycleRider, BicycleGroup).
4. **Tie-break 2 (S2):** among those, the lowest median GPU latency to predict one test sequence
   (all its agents, all samples), RTX 4500 Ada. A pick above 100 ms is flagged as not real-time
   at 10 Hz.

**Reported, not used for the pick:** recall / false-alarm rate at λ = 1; 0–3 s horizon; 3 m and
10 m conflict radii; near-ego (≤ 10 m) ADE/FDE; p95 FDE; miss rate (FDE > 2 m); mean band area;
val→test coverage drop.

**Caveat:** open loop. The ego path is what the human driver did, and nearby agents reacted to it;
this is a proxy for closed-loop collision rates, which need a planner in CARLA ([[planner]],
[[closed-loop-eval]]).

Related: [[benchmark-v2]], [[conformal-prediction]], [[bootstrap-ci]]
