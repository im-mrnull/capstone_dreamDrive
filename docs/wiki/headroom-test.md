# Headroom-First Test — can knowing the future help, and can several futures help?

## Status: 📌 Pre-registered (2026-10-29) — nothing has been driven in this simulator yet

This rule is committed **before** any package is driven with branched agents. Only the
following ran first:
- engineering checks (`counterfactual_sim.py check`: dose 0 reproduces the non-reactive
  episodes 36/36 for every package and ideal reference);
- unit tests;
- log-only statistics: eligibility, branch strength and plausibility (section 3);
- a timing run on 2 dev scenes with an unused draw seed (99), which printed no scores.

The result JSON records this commit.

## 1. Why

Two simulators so far show **no headroom**: a perfect predictor is no better than constant
velocity.
- Non-reactive replay on the dev scenes: oracle 0.901 = CV 0.901.
- Reactive agents: log oracle − CV = −0.001 ([[reactive-test]]).

If perfect knowledge of the future gains nothing, no predictor can, single or multi-future.
That is what the reactive test found. The literature reports the same on nuPlan: perfect
predictions don't significantly improve the integrated prediction–planning methods tested
(Distelzweig et al. 2025).

So this test **first establishes headroom**, using ideal predictors only. The learned models are
driven only if knowing the future, and then knowing its *distribution*, measurably helps.

## 2. Simulator ([`scripts/counterfactual_sim.py`](../../scripts/counterfactual_sim.py))

**Targets** are chosen from the logs alone, before anything is driven. An agent is a target if:
- it is present at the episode start with a full 2 s history;
- it moves at some point in the episode;
- its logged position enters the ego's corridor within the episode. The corridor is the
  route ± (half ego width + half agent width + 0.5 m), from 2 m ahead of the ego out to its
  reach. The reach is the human's progress or 6 m/s × 5 s, whichever is larger, plus 10 m.

Each episode has at most 3 targets, earliest corridor entry first. Everyone else replays the
log.

**Hidden intent.** Each target keeps its **logged path**; only its timing branches (time
warp). The branch starts 2 s before the agent's logged corridor entry, or at the episode start
if that is sooner. Each target's branch is drawn independently from a hash of (scene, start,
persona, draw seed, agent id), so it is identical for every package.

| Branch | Prior | Motion |
|---|---|---|
| logged | 0.5 | the log |
| hesitate | 0.25 | slows towards a stop (deceleration ≤ its class's comfortable braking) until it is DOSE behind its log, then returns to the logged pace (acceleration ≤ class a_max): yielding, hesitating, a braking lead vehicle |
| assertive | 0.25 | up to RATE × its logged pace (acceleration ≤ a_max, speed ≤ 3 m/s for pedestrians or 16.7 m/s otherwise) until it is DOSE ahead, then the logged pace: goes first |

- **Doses**, a ladder fixed now:
  - **primary**: DOSE 1.5 s, RATE 1.3;
  - **strong**: DOSE 3.0 s, RATE 1.6.
- Class limits are the IDM table of [[reactive-test]].
- Beyond the scene's last frame, an advanced agent continues at its final logged velocity.
- **Branches don't react to the ego.**

**Draws:** 5 draw seeds (0–4) × 2 personas per eligible episode.

**Prediction:** predictors see only the simulated past. Once a target leaves its log, the
learned models predict online, exactly as in [[reactive-test]].

**Ideal references** (no learning, no bands; non-target agents get their exact logged
future):

| Reference | What it knows |
|---|---|
| Branch oracle | the realised branch's future (perfect information) |
| Ideal single (most likely) | the most probable branch still consistent with the observed past |
| Ideal single (mean) | the posterior-weighted mean of the consistent branches (the best single trajectory in the L2 sense) |
| Ideal set | every consistent branch with its posterior weight (the ideal multi-future predictor) |

The posterior starts at the prior. A branch is ruled out once it is **0.3 m** from the
observed past. This matches the annotation jitter: the median maximum residual around a smooth
2 s path is 0.21 m. No real predictor can tell branches apart sooner.

## 3. What the generator produces (log-only statistics, dev scenes)

**Eligibility**
- 88 of 112 dev episodes are eligible, with 2.28 targets each.
- That gives **880 episodes per package per dose** (88 × 2 personas × 5 draws).

**Targets by class**

| Class | Targets |
|---|---|
| Scooter riders | 61 |
| Cars | 49 |
| Motorcycle riders | 41 |
| Tour cars | 39 |
| Trucks | 7 |
| Pedestrians | 4 |

**Branch strength (primary dose)**
- Hesitating targets end a median 2.55 s behind their log.
- Assertive targets end a median 1.29 s ahead.
- The branches are 0.5 m off the log after a median 0.7–0.8 s from the episode start.
- For 35% of targets, the branch starts later than the episode start.

**Plausibility.** I used the four-criterion scorecard's limits (speed and acceleration over
0.5 s). The logged trajectories themselves fail them 22% of the time, because of annotation
jitter. The branches add to that:

| Dose | Hesitate (beyond its own log) | Assertive (beyond its own log) |
|---|---|---|
| Primary | +4.2% | +10.2% |
| Strong | +2.4% | +16.5% |

The time warp speeds up the log's jitter along with the motion.

## 4. Packages

**Stage 1** (all planner v2):
- human replay;
- ignore agents;
- CV;
- Branch oracle;
- both ideal singles;
- Ideal set (every consistent branch as an obstacle).

The Ideal set also runs under the separately pre-registered probability-aware planner
(confidential), at its frozen chosen setting.

**Stages 2–3:**
- singles under planner v2: CV, ST-GAT, K-mode top-1, multi-future transformer top-1,
  AgentFormer most-likely;
- sets under the probability-aware planner: K-mode, multi-future transformer, AgentFormer;

as in [[reactive-test]].

## 5. Stages, gates, decisions

**Stage 1. Headroom, ideal references only, 32 dev scenes.**

- **G1, perfect-information headroom** (pooled over draws): the Branch oracle beats CV by
  either:
  - ≥ 0.02 mean episode score; or
  - ≥ 2 points fewer at-fault collision episodes.
- **Dose ladder:** G1 is checked at the primary dose first. If it fails there and passes at
  the strong dose, the strong dose is used from then on. If it fails at both, **stop**: even
  these branches create no prediction headroom, and that is the reported outcome.
- **G2, multi-future headroom** (at the dose G1 chose): the Ideal set under the
  probability-aware planner beats **both** ideal singles (v2) by either:
  - ≥ 0.01 score over the better single score; or
  - ≥ 1 point fewer at-fault episodes than the better single.
- **If G2 fails:** headroom exists, but the current planner can't turn even an ideal
  distribution into a gain. That is reported, and the learned models are **not** driven. The
  next step is a contingency (branching) planner. It will be pre-registered separately and
  must pass G2 before any learned model is driven.

**Stage 2. Learned models, dev pilot** (only if G1 and G2 pass): the gate of
[[reactive-test]]. Some set must beat its own single trajectory by:
- ≥ 0.01 score; or
- ≥ 1 point fewer at-fault episodes.

**Stage 3. Confirmatory** (only if stage 2 passes):
- the 73 primary scenes, each driven by the fold model of its test fold (seed 0), 5 draws;
- the same 18 one-sided paired scene-bootstrap tests as [[reactive-test]] (2,000
  replicates, Holm);
- the same reading rule and lead-model rule.

**This would be the third analysis on these scenes,** after closed-loop v2 and the planner
study. The reactive test stopped at its pilot and never used them. The reuse is disclosed.

**Reported, not decisive:**
- **Value of information:**
  - Branch oracle − CV;
  - Branch oracle − the best ideal single.
- **Value of an ideal distribution:**
  - Ideal set (each planner) − the best ideal single.
- **Share captured by the learned models:** (learned − CV) ÷ (ideal − CV).
- **Collisions with targets:** at-fault episodes involving a target.
- **Other driving metrics:** near-miss, harsh braking, progress, jerk.
- **Plausibility check:** the human replay's at-fault rate.

## 6. Caveats (stated in advance)

- The branches are synthetic and change timing only. There is no alternative *path*
  intent: no turning instead of going straight, and no pedestrian who starts crossing. Their
  parameters are set a priori.
- Pedestrians are rare among targets (4 of 201); two-wheelers and cars dominate.
- The branches don't react to the ego, and the other agents don't react at all.
- Assertive branches are somewhat less plausible than the logs (section 3).
- The ideal references know the branch model, and its prior, exactly.
- The planners were tuned in the non-reactive simulator.

Related: [[reactive-test]], [[closed-loop-cv]], [[planner]], [[negative-results]]
