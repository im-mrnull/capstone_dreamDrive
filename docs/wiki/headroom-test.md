# Headroom-First Test — can knowing the future help, and can several futures help?

## Status: 📌 Pre-registered (2026-10-29) · stage 1 run (2026-10-30): G1 passes at the strong dose (section 7) · matched-planner G2 pre-registered (section 8)

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

## 7. Stage 1 result (2026-10-30, 32 dev scenes, 880 episodes per package per dose)

Run with `counterfactual_sim.py stage1` against rule commit `d47f7cf`. Output:
[`headroom_stage1.json`](../../outputs/benchmark_v2/headroom_stage1.json). All packages use planner v2.

| Package | Primary: score | Primary: at-fault | Strong: score | Strong: at-fault | Strong: VRU at-fault |
|---|---|---|---|---|---|
| Human driver (log replay) | 0.862 | 9.4% | 0.786 | 17.5% | 9.3% |
| No prediction | 0.775 | 14.1% | 0.727 | 18.8% | 10.2% |
| CV | 0.834 | 2.8% | 0.795 | 5.1% | 3.6% |
| Branch oracle | 0.847 | 1.8% | 0.820 | 1.9% | 0.9% |
| Ideal single (most likely) | 0.845 | 1.8% | 0.823 | 2.0% | 0.7% |
| Ideal single (mean) | 0.847 | 1.8% | 0.823 | 2.0% | 0.7% |
| Ideal set | 0.845 | 2.2% | 0.823 | 1.6% | 0.9% |

**G1 (Branch oracle vs CV):**

| Dose | Score gain | At-fault reduction | G1 |
|---|---|---|---|
| Primary | +0.013 | +1.0 pt | fail |
| Strong | +0.025 | +3.2 pt | **pass** |

By the dose ladder (section 5), **the strong dose is used for every later stage.**

This is the first simulator in the project where perfect information beats CV by a margin set in
advance. In the non-reactive and reactive replays the oracle was level with CV.

**Reported, not decisive (strong dose, planner v2):**
- **Value of information:**
  - Branch oracle − CV: +0.025 score, −3.2 pt at-fault;
  - Branch oracle − best ideal single: **−0.003** score, −0.1 pt at-fault.
- **Knowing the realised branch adds nothing beyond knowing the branch model.** The ideal singles
  capture all of the oracle's gain over CV. The likely reason: branches separate from each other
  (0.3 m) within about a second, and the planner replans every 0.2 s, so it catches up once the
  branch is visible.
- **Ideal set vs best ideal single:** ±0.000 score and −0.4 pt at-fault (1.6% vs 2.0%). Under
  planner v2, the ideal distribution adds nothing measurable.
- **Plausibility check:** the human replay's at-fault rate rises to 9.4% (primary) and 17.5%
  (strong), from 0 in the non-reactive replay. Expected rather than a defect: the logged ego cannot
  respond to branches it never saw (for example, it can rear-end a hesitating lead vehicle; the
  breakdown by branch was not computed). It does show that the strong dose produces situations the logged
  drivers did not face.
- **The ideal predictors brake harshly less often than CV** (13–14% vs 20% of episodes at the strong
  dose). They also beat CV on VRU at-fault (0.7–0.9% vs 3.6%).

**G2** compares the Ideal set under the separately pre-registered (confidential) planner with
these ideal singles. It is recorded with that planner.

## 8. Matched-planner G2 (pre-registered 2026-10-30, before any draw 5–19 is driven)

**Why.** G2 (section 5) compared the Ideal set under one planner with the ideal singles under
another (planner v2). So a gain could come from the planner rather than from the distribution.

This section repeats the question with **every predictor driven under every planner**. Singles and
sets get the same planner improvements wherever they apply. The section 5 G2 outcome stands as
reported; this is a new, separately pre-registered test.

**Simulator:** as sections 2–5, at the strong dose (fixed by G1), on the 32 dev scenes.

**Draws** (new; draws 0–4 are not reused):
- **5–9: planner tuning only;**
- **10–19: evaluation only.** That gives 1,760 episodes per cell (88 episodes × 2 personas × 10
  draws).

**Predictors:** Branch oracle (the ceiling), Ideal single (mean), Ideal single (most likely), Ideal
set.

**Planners:**

| Planner | How it uses a set | Where it is specified |
|---|---|---|
| v2 (section 4) | every future is an obstacle (worst case) | [[planner]] |
| Proposal planner (new) | worst case over futures, for collision checks and the IDM leader | below |
| Probability-aware planner | uses the future weights | confidential, pre-registered separately |
| Contingency planner | uses the future weights | confidential, pre-registered separately |

So the grid has 16 cells: 4 planners × 4 predictors.

### Proposal planner ([`scripts/proposal_planner.py`](../../scripts/proposal_planner.py))

PDM-Closed style (Dauner et al., CoRL 2023).

**Proposals:**
- the constant-acceleration × lateral-target candidates (fine accelerations, as in planner v3);
- **IDM car-following proposals** towards 0.2, 0.4, 0.6, 0.8 and 1.0 × the reference speed, for
  each of the 5 lateral targets. The ego follows the nearest predicted agent ahead in its lane
  (worst case over futures).
  - IDM parameters: a_max 1.5, b 2.0, time headway 1.2 s, standstill gap 2 m.
  - The free-road term never brakes harder than b.
  - The first IDM command is executed (clipped to [−6, 2] m/s²).

**Scoring:**
- planner v3's feasibility, soft penalty, clearance (1 m) and jerk (0.2) terms;
- a progress term: prog_w × (1 − progress ÷ the best feasible proposal's progress).

**Option `const`:**
- "always": constant proposals compete with the IDM ones;
- "fallback": constant proposals are used only when no IDM proposal is feasible, like
  PDM-Closed's emergency manoeuvres.

**Checks:**
- With no IDM proposals and no progress term it reproduces planner v2 (48/48 dev episodes for
  each of 3 packages).
- Unit tests: behind a braking lead car between parked cars, planner v2 commands −6 m/s² and
  closes the gap. The proposal planner (fallback mode) stays above −4 m/s² and keeps a positive
  gap.
- Its design was developed on synthetic scenes only. No branched episode was driven with it
  before this commit.

### Tuning (equal treatment)

Each new planner gets **12 settings** on draws 5–9:
- **Proposal planner:** const {always, fallback} × clearance weight {0, 2} × prog_w {0, 2, 5}.
- **Contingency planner:** 12 settings, listed in its confidential pre-registration.

**Selection:** the highest mean score over {Branch oracle, Ideal single (mean), Ideal set}, so
neither predictor type is favoured. The chosen setting is frozen before draws 10–19.

Planner v2 and the probability-aware planner keep their existing frozen settings.

### Latency pre-filter

Per-replan planning time in a single process, on draws 10–11 with the Ideal set and the Ideal
single (mean). A planner is eligible only if, for both predictors:
- **median ≤ 50 ms;**
- **p95 ≤ 100 ms.**

That leaves room for the slowest learned predictor (AgentFormer, 102 ms median) inside the 0.2 s
replan cycle. Stage 2 rechecks latency with the learned 10-mode sets.

### Decisive test: G2-matched

1. **Comparator:** the best single cell over all eligible planners, using the Ideal singles only
   (the oracle is excluded). Take the highest score for the score test and the lowest at-fault
   rate for the at-fault test.
2. **Test:** for each eligible planner, compare its Ideal set with that comparator, one-sided:
   - score gain ≥ **0.01**; or
   - at-fault reduction ≥ **1 point**.
3. **Statistics:** a paired scene bootstrap (2,000 replicates, draws pooled per scene). Holm
   adjustment over the 2 × (number of eligible planners) tests.
4. **G2-matched passes** if at least one test reaches its threshold with a Holm-adjusted
   p < 0.05.

This tests the system-level question: does the best multi-future system beat the best single-trajectory
system, with planner effects held equal?

**Reported, not decisive (every cell):**
- score, at-fault (and VRU at-fault), harsh braking, progress, near-miss, jerk, latency (median /
  p95);
- each as a difference vs the Ideal single (mean) under the **same** planner, and vs the Ideal
  single under v2.

**Attribution:**
- **Value of the distribution per planner:** Ideal set(P) − best Ideal single(P).
- **Ceiling per planner:** Branch oracle(P) − best Ideal single(P).
- **For the contingency planner specifically**, against the ideal single under the same planner
  and the best single overall:
  - Does it reduce harsh braking?
  - Does it improve progress?
  - Does it reduce at-fault collisions?
  - Does it meet the threshold?
  - Does it fit the latency budget?

**Decision:**
- **Pass:** stage 2 (learned models, section 5). The winning planner drives the learned singles
  *and* the learned sets.
- **Fail:**
  - report: "with ideal predictions and matched planners, a multi-future distribution adds no
    measurable driving benefit in this simulator";
  - the learned models are not driven;
  - the next design is path-level intent branches (turn vs straight; a pedestrian who starts to
    cross or doesn't). These keep the futures ambiguous for longer than timing branches do.

**Caveats:**
- These are the same 32 dev scenes as stage 1, but with fresh draws.
- The proposal planner and the contingency planner were designed after seeing the stage 1 result.
- Only the dev scenes are used. A confirmatory run on the 73 primary scenes would follow stage 2.

Related: [[reactive-test]], [[closed-loop-cv]], [[planner]], [[negative-results]]
