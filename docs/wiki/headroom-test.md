# Headroom-First Test — can knowing the future help, and can several futures help?

## Status: 📌 Pre-registered (2026-10-29) · stage 1 run (2026-10-30): G1 passes at the strong dose (section 7) · matched-planner G2 pre-registered (section 8), public cells run (section 9) · primary-scene confirmation pre-registered (section 10), public cells run (section 11): a tuned safety margin matches the ideal set · lateral-intent benchmark stopped at its validity gate; Part 1b pre-registered (section 12), public cells run (section 13) · regime map pre-registered (section 14)

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

## 9. Matched-planner G2 result: public planners (2026-10-30)

Run against rule commit `c30258c`.
- **Tuning (draws 5–9):** [`headroom_matched_tune.json`](../../outputs/benchmark_v2/headroom_matched_tune.json).
- **Evaluation (fresh draws 10–19, 1,760 episodes per cell):**
  [`headroom_matched.json`](../../outputs/benchmark_v2/headroom_matched.json).

**Proposal planner, chosen setting:**
- constant proposals compete with the IDM ones;
- clearance weight 2, progress weight 2.

Its tuning mean was 0.862, against 0.71–0.73 for every fallback-only setting. IDM proposals alone
are too cautious in this simulator.

| Planner | Predictor | Score | At-fault | VRU at-fault | Harsh braking | Near-miss | Progress |
|---|---|---|---|---|---|---|---|
| v2 | Branch oracle | 0.815 | 1.93% | 0.63% | 13.6% | 45.1% | 0.922 |
| v2 | Ideal single (mean) | 0.816 | 2.05% | 0.63% | 13.5% | 45.0% | 0.923 |
| v2 | Ideal single (most likely) | 0.815 | 2.44% | 0.68% | 13.0% | 44.8% | 0.923 |
| v2 | Ideal set | 0.816 | 1.76% | 0.68% | 13.5% | 44.7% | 0.918 |
| Proposal | Branch oracle | 0.857 | 1.65% | 0.68% | 10.4% | 32.4% | 0.933 |
| Proposal | Ideal single (mean) | 0.858 | 1.76% | 0.74% | 9.5% | 32.8% | 0.934 |
| Proposal | Ideal single (most likely) | 0.858 | 1.65% | 0.74% | 9.6% | 33.0% | 0.932 |
| Proposal | **Ideal set** | **0.866** | **1.14%** | 0.74% | 9.3% | 32.1% | 0.935 |

**Latency** (median / p95, both within budget):
- planner v2: 2.2 / 3.2 ms;
- proposal planner: 9.9 / 25.9 ms.

**Reading:**
- **The planner matters most.** Moving every predictor from v2 to the proposal planner gains
  +0.04–0.05 score:
  - near-misses fall from 45% to 32%;
  - harsh braking falls from 13% to 10%;
  - progress rises.
- **With the planner held equal, the ideal distribution adds little.**
  - Under v2: ±0.000 score, −0.3 pt at-fault.
  - Under the proposal planner: +0.008 score (95% CI [−0.001, +0.018]) and −0.5 pt at-fault
    (CI [−0.1, +1.3]), against the best ideal single.
  - Both are below the pre-registered threshold, and neither is significant after Holm.
- **The oracle gains nothing over the ideal singles under either planner** (−0.001). Knowing the
  realised branch is worth no more than knowing the branch model.
- **The set nudges ahead of the oracle under the proposal planner** (0.866 vs 0.857). A planner
  that clears every possible branch drives a little more cautiously than one that trusts a single
  future.

**G2-matched over all four eligible planners**, including the two confidential ones, is recorded
with the confidential planners. The public cells alone do not meet the gate.

## 10. Confirmation on the primary scenes (pre-registered 2026-10-31, before any primary scene is driven here)

**Why.** Under the proposal planner, the Ideal set beat the best ideal single by +0.008 score
(95% CI [−0.001, +0.018]) and 0.5 pt fewer at-fault episodes on the dev scenes (section 9). That
is below the threshold and not significant. Two questions are open:
1. Is the small advantage real?
2. Is it more than **extra caution**? The set also beat the oracle (0.866 vs 0.857), which hints
   that keeping clear of every branch simply makes the ego more careful.

A further hypothesis is new: **contingency planning built on the proposal planner** (confidential,
pre-registered separately).

**Scenes and draws:**
- **Evaluation:** the **73 primary scenes** (`closed_loop_cv.fold_of_scene()`, train split), never
  driven in this simulator.
  - 221 of 262 episodes are eligible (log-only count), in 62 scenes, with 2.32 targets each.
  - Draws 0–9 → **4,420 episodes per cell**.
  - This is the third closed-loop use of these scenes, after closed-loop v2 and the planner
    study; disclosed.
- **Tuning:** dev scenes, draws 5–9 only (no new dev draws).
- **Strong dose** (fixed by G1).

**Cells** (ideal predictors only):

| # | Planner | Predictor | Role |
|---|---|---|---|
| 1 | Proposal (frozen setting, section 9) | Ideal single (mean); Ideal single (most likely) | strong single-future baseline |
| 2 | Proposal + tuned safety margin | Ideal single (mean) | caution control |
| 3 | Proposal | Ideal set (worst case over futures) | multiple futures, no branching |
| 4 | Contingency (confidential) | Ideal single (mean) | contingency architecture control |
| 5 | Contingency (confidential) | Ideal set | branching + multiple futures |
| – | Proposal; contingency | Branch oracle | ceiling, reported |

**Safety-margin control:**
- `margin_m` (an extra buffer on every predicted footprint) is added to the proposal planner. At
  0 it reproduces the frozen planner exactly.
- Tuned for the Ideal single (mean) on dev draws 5–9 over {0, 0.25, 0.5, 0.75, 1.0} m; best score
  wins. This gives the single trajectory its best level of caution.

**Latency pre-filter:**
- as section 8 (median ≤ 50 ms, p95 ≤ 100 ms), single process;
- measured on primary draws 0–1 with the Ideal set and Ideal single (mean).

A planner outside the budget is reported, but its cells cannot be compared.

### Tests

Statistics:
- one-sided paired scene bootstrap, 2,000 replicates, draws pooled per scene;
- score and at-fault for each test;
- Holm over all of them (8 when every planner is eligible).

| Test | Comparison | Question |
|---|---|---|
| **H1 (decisive)** | the best set cell (3, 5) vs the best single cell (1, 2, 4), chosen per metric on the evaluation data | does a multi-future system beat every single-future system, including the cautious one? |
| H2 | 5 vs 4 | do multiple futures help inside the contingency planner? |
| H3 | 5 vs 3 | does branching add to multiple futures? |
| H4 | 3 vs 2 | is the set's advantage more than extra caution? |

Picking the best single on the evaluation data favours the singles, so this choice is
conservative.

**H1 reading rule (fixed now):**
- **"meaningful benefit":** gain ≥ 0.01 score or ≥ 1 pt at-fault, and Holm p < 0.05;
- **"small but real benefit":** Holm p < 0.05 but below the threshold;
- **"no demonstrated benefit":** otherwise.

**Reported:**
- per cell: score, at-fault, VRU at-fault, harsh braking, near-miss, progress, jerk, latency;
- the interaction (5 − 4) − (3 − 1);
- the oracle ceiling per planner.

**Decision:**
- **Meaningful or small-but-real benefit:**
  - stage 2: the learned models, on the dev pilot;
  - the winning planner is used for the learned singles and the learned sets alike;
  - the margin control is included for the learned single.
- **No demonstrated benefit:** report it, and move to path-level intent branches.

**Power (stated in advance):** if the true gain is the dev estimate (+0.008), this test has
roughly a 60–70% chance of finding it significant after Holm. If it does not, that is not proof
of no effect.

**Runners:**
- `counterfactual_sim.py confirm-tune` (margin) → `headroom_confirm_tune.json`;
- `counterfactual_sim.py confirm` (cells 1–3 and the proposal oracle) → `headroom_confirm.json`;
- the confidential runner adds cells 4–5 and computes H1–H4 (`confirm_tests`).

Related: [[reactive-test]], [[closed-loop-cv]], [[planner]], [[negative-results]]

## 11. Confirmation result: public cells (2026-10-31)

Run against rule commit `9d2ab26`.
- **Margin tuning (dev draws 5–9):** [`headroom_confirm_tune.json`](../../outputs/benchmark_v2/headroom_confirm_tune.json).
  Score by margin: 0 m 0.858, 0.25 m 0.861, **0.5 m 0.868 (chosen)**, 0.75 m 0.862, 1.0 m 0.851.
- **Evaluation (73 primary scenes, 62 with eligible episodes, draws 0–9, 4,420 episodes per cell):**
  [`headroom_confirm.json`](../../outputs/benchmark_v2/headroom_confirm.json).

| Planner | Predictor | Score | At-fault | VRU at-fault | Harsh braking | Near-miss | Progress | Jerk (RMS) |
|---|---|---|---|---|---|---|---|---|
| Proposal | Branch oracle | 0.840 | 1.47% | 0.25% | 9.5% | 40.5% | 0.924 | 1.88 |
| Proposal | Ideal single (mean) | 0.838 | 1.38% | 0.27% | 10.5% | 40.5% | 0.923 | 1.96 |
| Proposal | Ideal single (most likely) | 0.840 | 1.38% | 0.25% | 10.4% | 40.3% | 0.924 | 1.95 |
| Proposal | Ideal set | 0.841 | 0.97% | 0.27% | 10.6% | 39.0% | 0.921 | 2.02 |
| Proposal + 0.5 m margin | Ideal single (mean) | **0.854** | **0.97%** | 0.27% | 18.0% | 23.0% | 0.909 | 2.82 |

**Latency** (median / p95, all within budget): proposal 7.2–7.4 / 14.9–15.2 ms; with margin
7.4–7.5 / 15.5 ms.

**Comparisons among the public cells** (paired scene bootstrap, 2,000 replicates, unadjusted;
gain is set minus single, at-fault in points, positive = set better):

| Ideal set vs | Score gain (95% CI) | At-fault gain (95% CI) |
|---|---|---|
| Ideal single (mean) | +0.003 [−0.002, +0.008] | +0.4 pt [−0.3, +1.4] |
| Ideal single (most likely) | +0.002 [−0.004, +0.007] | +0.4 pt [−0.4, +1.5] |
| **Proposal + margin, Ideal single (mean) (H4)** | **−0.013 [−0.024, −0.002]** | ±0.0 pt [−0.5, +0.6] |
| Branch oracle | +0.001 [−0.004, +0.006] | +0.5 pt [−0.2, +1.6] |

**Reading:**
- **The dev advantage shrank on fresh scenes.** Against the plain ideal singles the set gains
  +0.002–0.003 score (dev: +0.008) and 0.4 pt at-fault; no interval excludes zero.
- **A tuned safety margin does at least as well as the set (H4).** The cautious single matches the
  set on at-fault (0.97% each) and *beats* it on score by 0.013; the interval excludes zero in the
  margin's favour. Its price is more harsh braking (18% vs 11%) and slightly less progress, which
  the score weights below the near-miss reduction (39% → 23%).
- **So what the set added under this planner is explained by caution.** A single trajectory with a
  well-chosen buffer reproduces it, more cheaply.
- *Exploratory (not pre-registered):* the margin alone adds +0.016 score over the plain single
  (95% CI [+0.004, +0.028]).

**H1–H3 and the Holm-adjusted family** use the confidential contingency planner and are recorded
with it.

## 12. Lateral Intent Counterfactual Benchmark: stopped at its validity gate (2026-10-31); Part 1b pre-registered

### 12.1 Why, and what it is

- **Background:** under timing branches (sections 2–11), one trajectory plus a tuned safety
  margin matched the ideal set (section 11). There the futures differ only in *when* a road
  user moves.
- **This benchmark:** it changes *where* road users go relative to the ego lane, so that an
  average of futures can be a path no real future takes.
- **Scope:** it tests **lateral** intent only. **Route** intent (turn vs straight) would be a
  separate extension, with route alternatives from other agents' logged paths under geometric
  alignment rules. It is not pursued here.

**The world** ([`scripts/counterfactual_sim.py`](../../scripts/counterfactual_sim.py), `PathWorld`):

| Element | Definition |
|---|---|
| Candidates | as section 2, with the corridor widened by 2 m |
| Context | heading vs the ego route at the branch start: same direction (< 45°), crossing (45–135°), oncoming (> 135°, not branched) |
| Branches (prior 0.5 / 0.25 / 0.25) | logged; **hold lane** / **wait**: never closer than the corridor edge + 0.3 m; **cut in** (to the ego centreline) / **cross** (to the far edge) |
| Lateral limits (own lateral speed, acceleration) | pedestrians 1.0 m/s, 1.0 m/s²; two-wheelers 1.5, 2.0; vehicles 1.0, 1.5. Strong dose: 1.5×. |
| No reversals | hold/wait brakes at the limit and holds where it stops. Cut in/cross is not available to an agent moving away from the centreline. |
| Overlap rule | a candidate whose branch overlaps a logged agent for > 0.5 s where its log does not is not branched (identical in every package) |

**Identity:** with all limits at 0, every episode equals the plain simulator (40/40). The
timing world is unchanged (36/36).

### 12.2 The validity report: the go/no-go gate (dev scenes only, no G1 quantity computed)

The gates were fixed in the approved plan before any of this ran. The report is
[`lateral_validity.json`](../../outputs/benchmark_v2/lateral_validity.json). Values are
primary / strong dose.

| Gate | Needed | Result |
|---|---|---|
| Eligible dev episodes | ≥ 80 | 105 / 105 ✅ |
| **Decision-sensitive episodes**: between two real alternatives, the oracle-driven proposal planner's first second differs (> 1 m/s² mean acceleration, > 0.5 m lateral target, > 1 m position, or manoeuvre class) | ≥ 20% | **16.2% / 18.1% ❌** |
| **Separation at +4 s**, median over all pairs of real alternatives | ≥ 1.5 m | **1.16 / 1.26 m ❌** (1.41 / 1.44 m measured from the branch start) |
| Plausibility: each branch type adds ≤ 10% implausibility over its own log | ≤ 10% | depends on the reading (below) |

**Verdict: no-go.** As pre-registered, the lateral confirmation is **not run**. Decision
sensitivity and separation fail under every reading of the plan.
- One bounded design revision was made on dev data, with its two changes stated before the
  rerun:
  - hold-lane branches keep the log's step length;
  - branches start 1 s instead of 2 s before the corridor approach.
- It did not change the verdict.

**Plausibility depends on the reading:**
- **Branches newly flagged where their log is not** (scorecard limits, heading jumps > 45° per
  0.2 s, lateral reversals, lateral limits; 0.5 s steps):
  - hold lane 4.9% / 4.5%;
  - cut in 11.7% / 7.5%;
  - cross 31% (n = 13).
- **As a rate difference over the log,** every branch type is *below* its log (−15 to −47 pt),
  because the logs carry annotation jitter.

**What the world offers (primary dose):**

| Item | Result |
|---|---|
| Decision-sensitive episodes, any 1 s window of the 5 s episode | 29.5% (not the gate quantity) |
| Episodes where the prior-weighted mean is > 1 m from every branch **and** in the ego corridor while no branch is, or in space no other agent traversed (invalid mean trajectory, IMTR; first 3 s) | 12.4% |
| Targets without a real alternative (no branch more than 0.3 m from the log) | 22% |
| Overlap exclusions | 21% of candidates; denser scenes (21.5 vs 17.8 agents within 30 m); 1 episode lost |
| Median time until a branch is 0.3 m from its log | hold lane 1.3 s (21% ambiguous ≥ 3 s); cut in 0.6 s (1% ≥ 3 s) |
| Targets by context | same direction about 230; crossing about 15 (3 pedestrians) |

No subgroup claims are made.

**Finding.** Lateral intent built from real IDD-3D tracks rarely changes what the ego must do
immediately. A future that differs (cut in) shows itself within about 0.6 s. A future that
stays ambiguous (hold lane) mostly does not need a different action. So, beyond timing
uncertainty, this data offers little room for multi-future prediction to matter in closed loop.
**Multi-future prediction is dropped as a design goal (decision of section 12.6 of the
plan).** The multi-future line is written up as a negative result.

**Engineering record, which the result depends on:**
- **Adversarial code review:** two rounds, each finding checked by two skeptics. It found and
  fixed:
  - a retiming bug in hold-lane branches (noise turned into forward motion);
  - a projection that jumped between route segments;
  - a too-lenient "distinct branch" test;
  - occupancy that counted the target's own track;
  - exclusion counts that left out lost episodes.

  The numbers above are after the fixes.
- **Unit tests:** `tests/test_path_branches.py` (11 tests).
- **GPU benchmark:** CUDA penetration made whole episodes **2.8× slower** with identical
  commands ([`lateral_gpu_benchmark.json`](../../outputs/benchmark_v2/lateral_gpu_benchmark.json)).
  The simulator stays on CPU.

**Statistics built for the confirmation, recorded although unused:**
- **Tests:** a max-vs-max H1 with a least-favourable union-bound sign-flip null, plus pair and
  interaction sign-flip tests.
- **Calibration** ([`lateral_stats_calibration.json`](../../outputs/benchmark_v2/lateral_stats_calibration.json)),
  1,000 synthetic datasets per scenario from the section 10 scene structure:
  - H1 false-positive rate ≤ 3.8% in every null, including composite nulls with unequal
    singles;
  - family-wise rate ≤ 3.2%;
  - pairwise tests 3.9–6.0%, H5 at-fault 7.8% (asymmetric null only);
  - power 95–98%, or 19% with a 3×-noisy set cell.
- **Why the method changed:** the review showed the plan's first proposal (an all-cells-equal
  recentred null) rejects a true null 20–30% of the time when the singles differ, and the
  percentile-bootstrap p-values of section 10 reject 6–9%. They were replaced before any use.

### 12.3 Part 1b: the best single-future setup on fresh draws (pre-registered now)

**Question.** In section 10/11 the confidential contingency planner with an Ideal single
(mean) had the fewest at-fault episodes of any cell. That was exploratory: it was not a
pre-registered comparison. Does it hold on fresh draws?

| | |
|---|---|
| World | timing branches, strong dose (section 7), unchanged |
| Scenes, draws | 73 primary scenes, **draws 20–29**: unused anywhere. Draw history: primary 0–9 in section 10; dev 10–19 in section 8. The lateral confirmation never ran. |
| Cells (all Ideal single (mean)) | confidential contingency planner (section 10 setting); proposal planner (frozen, section 9); proposal + 0.5 m margin (section 10) |
| Primary metric | at-fault rate. Secondary: score. |
| Tests | contingency vs proposal and contingency vs margin, each on at-fault and score; one-sided **scene sign-flip randomisation** tests with **200,000 flips** (Monte Carlo SE about 0.0003 at p = 0.0125); Holm over 4; CIs by scene-cluster bootstrap (2,000) |
| Reading | **confirmed best single setup:** both at-fault tests significant; **better than the plain planner only:** only the proposal comparison significant; **not confirmed:** otherwise. Score is reported with CIs and cannot override the at-fault reading. |
| Reported, outside the rule | each at-fault contrast with its largest-contributing scene removed. In section 10 one scene supplied 15 of the 37 avoided at-fault episodes. |
| Integrity checks in the runner | committed and clean code; same simulator, split and scene-data hashes as the public run; identical episodes and branch draws across the three cells |

**Disclosures:**
- **Reused scenes:** the 73 scenes are used for the fourth time. Fresh draws are new stochastic
  realisations, not new scenes, and draw-invariant scene effects carry over.
- **Power:** limited for the margin comparison (section 10 gap −0.4 pt).

**Runners:**
- `counterfactual_sim.py confirm-1b` (public cells);
- the confidential runner adds the contingency cell and applies `tests_1b`.

## 13. Part 1b result: public cells (2026-10-31)

Run against rule commit `12c857a`: timing world, strong dose, 73 primary scenes, fresh draws
20–29, 4,420 episodes per cell
([`headroom_confirm_1b.json`](../../outputs/benchmark_v2/headroom_confirm_1b.json)).

| Planner | Predictor | Score | At-fault | VRU at-fault | Harsh braking | Near-miss | Progress |
|---|---|---|---|---|---|---|---|
| Proposal | Ideal single (mean) | 0.837 | 1.43% (63) | 0.32% | 10.6% | 40.5% | 0.923 |
| Proposal + 0.5 m margin | Ideal single (mean) | 0.851 | 1.13% (50) | 0.20% | 18.4% | 23.3% | 0.908 |

**Replication on fresh draws.** The section 11 pattern repeats: the margin buys fewer
near-misses and at-fault episodes with about twice as much harsh braking.

**Part 1b reading:** the comparison with the confidential contingency planner, and the
reading under the section 12.3 rule, are recorded with the confidential planner.

## 14. Regime map: recognition delay × commitment interval (pre-registered 2026-10-31, before any dev draw 60–64 or primary draw 30–59 is driven)

### 14.1 Why, and the question

**Where things stand.**
- **No demonstrated advantage.** Under timing uncertainty, multi-future prediction has not
  demonstrated a statistically and practically meaningful advantage over the strongest
  single-future setups (sections 7–13). The lateral-intent benchmark stopped at its validity
  gate (section 12).
- **Measured:** the ceiling is flat. The Branch oracle minus the best ideal single is about 0
  under every planner (sections 7, 9, 11).
- **Hypothesis (not an established cause):**
  - the futures separate within about 0.6–1 s;
  - the ego replans every 0.2 s;
  - so it catches up once the future is visible, and knowing the futures in advance is worth
    little.

**Question.** Under what relationship between how late the ego learns which future is
happening and how long it must commit to an action does multi-future prediction give a
measurable closed-loop safety benefit?

**Final bounded experiment.** This is the last multi-future experiment. No branch types, model
families or worlds are added. Whatever the outcome, the project then moves to the chaos gap.

### 14.2 The map (timing world, strong dose, branches unchanged)

**Rows: recognition delay Δ ∈ {0, 1, 2} s.** Information only: the physical world, and so its
danger, is identical across rows.

**Information model.** The ideal predictors receive:
1. every road user's true current position and logged heading;
2. the branch posterior from the position history **up to t − Δ only** (the 0.3 m rule of
   section 2 unchanged);
3. the branch model.

**Anchoring.** Each branch's future starts at the current position and advances at that branch's
own timing increments. In log time: τ_real[t] + τ_b[t+k] − τ_b[t]. This is the identity for
the realised branch, so the oracle is unchanged and invariant across Δ.

**Excluded:** the realised timing rate, which is the intent cue in this world. No predictor
input comes from frames after t − Δ, except the anchor position.

**Columns: commitment interval R ∈ {0.2, 0.5, 1.0} s.** The ego replans every R.
- **0.2 s is our existing system baseline (5 Hz).** Production systems replan at about 10 Hz,
  which would sit further toward the fast end. So the conclusions apply to this setup, not to a
  10 Hz production system.
- **Hold-aware evaluation.** Proposals are evaluated as executed. A constant proposal is the
  executed command. An IDM proposal holds its first command for R, then follows IDM
  (`proposal_planner.py`, `hold_s` = R). The confidential planner runs under the same commitment
  rule.
- **Exact commitment.** Between replans the ego follows the commanded rollout exactly as
  evaluated (`closed_loop_sim.COMMIT_EXACT`). The frozen simulator executes one-frame rollouts,
  which recompute the lateral rate every frame and drift from the evaluated lateral path. At 1 s
  the drift is large (`tests/test_regime.py`). At 0.2 s it barely moves the metrics but still
  alters many episodes slightly (14.8).

### 14.3 Cells (per regime; common random numbers)

**Margin grid:** {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0} m. The margin is tuned for every planner and
predictor in every regime, on dev:
- the rule is the lowest at-fault rate, then the highest score. It is applied to every planner
  and predictor pair separately, single-future and multi-future alike;
- **this is a change of method from section 10**, which tuned the margin on score. At-fault is
  this experiment's primary metric, and a margin chosen on score can be worse on it. Section 10's
  results are not re-selected under the new rule;
- where the grid's top wins, the grid is extended by 2.5 and 3.0 m.

| Planner | Single-future cells | Multi-future cells |
|---|---|---|
| Proposal (section 9) | Ideal single (mean), margin 0; Ideal single (mean), tuned; Ideal single (most likely), tuned | Ideal set, margin 0; Ideal set, tuned |
| Contingency (confidential, section 10 setting) | Ideal single (mean), margin 0 and tuned | Ideal set, margin 0 and tuned |
| Ceiling | Branch oracle, both planners: once per commitment interval at Δ 0, and at the corner (Δ 2 s, R 1 s) | |

**Best single and best set per regime.** The best of the 5 single cells and the best of the 4 set
cells, fixed on dev by at-fault rate with score as the tie-break. A tuned margin of 0 shares its
episodes with the margin-0 cell.

### 14.4 Draws (gate A: draw provenance)

`counterfactual_sim.py draw-provenance` writes
[`draw_provenance.json`](../../outputs/benchmark_v2/draw_provenance.json). For every result it
gives:
- scene ids and draws, taken from the file itself, or from its runner where the file records
  none;
- rule commit, configuration hash and file hash.

The confidential results are covered in the same way, recorded with the confidential planner.

| Result file | Status | World | Scenes | Draws |
|---|---|---|---|---|
| `headroom_stage1.json` | run | timing, primary and strong doses | dev | 0–4 |
| `headroom_matched_tune.json`, `headroom_confirm_tune.json` | run | timing, strong | dev | 5–9 |
| `headroom_matched.json` | run | timing, strong | dev | 10–19 |
| `headroom_confirm.json` | run | timing, strong | primary | 0–9 |
| `headroom_confirm_1b.json` | run | timing, strong | primary | 20–29 |
| `lateral_validity.json` | run | lateral | dev | none (forced branches) |
| `lateral_gpu_benchmark.json` | run | lateral | dev (8 scenes) | 99 |
| `lateral_stage1.json`, `lateral_tune.json` | pre-registered, never run | lateral | dev | 0–4, 5–9 |
| `lateral_confirm.json` | pre-registered, never run | lateral | primary | 10–19 |

**Checks:**
- the timing-world confirmation used primary draws 0–9;
- no result has driven primary draws 10–19 (they are avoided here);
- the draws below are unused everywhere.

**This experiment's draws:**
- **dev tuning and selection:** dev scenes, draws **60–64** (880 episodes per cell);
- **evaluation:** 73 primary scenes, draws **30–39** (4,420 episodes per cell), or 30–49 / 30–59
  under the sample-size rule (14.6);
- **engineering, before this rule:** dev draws 90–98. These are never dev tuning or evaluation
  data:
  - resolution gate 90–94 (`regime_resolution.json`);
  - pilot and anchoring check 95–96 (`regime_pilot.json`, `regime_anchor_check.json`);
  - smoke test of the runners 97–98, on 2 dev scenes (outputs not kept).

  Each file is labelled engineering. The table above lists them, and no earlier result used
  draws 90–98.

The runners re-check that their draws are free before driving.

**Disclosed:** this is the fifth use of the primary scenes. Fresh draws are not fresh scenes.

### 14.5 Tests (at-fault primary, in percentage points; positive = multi-future better)

G_ij = best-single at-fault rate − best-set at-fault rate in regime ij (cells fixed on dev).

| Test | Statistic | Method |
|---|---|---|
| **P1, additive trend** | Σ c_ij G_ij, c_ij = row rank + column rank − 2 | one-sided scene sign-flip, 200,000 flips; scene-cluster bootstrap CI |
| **P2, adverse corner** (Δ 2 s, R 1 s) | best set vs best single, chosen on the evaluation data over all the corner's cells | `max_vs_max` (least-favourable union bound, section 12) |
| P3, interaction (secondary) | (G_ss − G_sf) − (G_fs − G_ff); first letter resolution, second reaction, s = slowest, f = fastest | sign-flip as P1; one-sided, unadjusted |

**Families.**
- **Primary:** P1 and P2 on at-fault, Holm over 2.
- **Secondary:** P1 and P2 on score, Holm over 2. P3 on both metrics, unadjusted.

**Reported in full, descriptively:**
- the whole 9-cell map: each regime's G with its CI, and its `max_vs_max` (Holm over 9);
- headroom: the column's best oracle at-fault rate − the best single;
- the share of the headroom the set captures;
- row and column effects;
- VRU at-fault, harsh braking, near-miss, progress and jerk;
- within-planner comparisons.

### 14.6 Calibration of the whole pipeline and the sample-size rule

**Each synthetic dataset runs the complete procedure:**
1. synthetic dev data;
2. dev selection of every regime's best single and best set;
3. synthetic primary data;
4. P1, P2 and P3, with Holm;
5. the reading rule.

**Construction:** as section 12. Each scene's mean plus one of that scene's real between-cell
deviations, with a random sign (an exact null).

**The calibration runs the runners' own code:**
- the dev selection `regime_select`, which the tuning runner uses (lowest at-fault rate, then
  highest score);
- the tests `regime_tests`;
- the reading `regime_reading`.

**Not simulated: the margin grid.**
- Each synthetic cell stands for a cell at its tuned margin.
- The margin is chosen on dev only, and dev and evaluation share no scenes and no draws. Under a
  null, every margin has the same truth on the evaluation data, so the choice cannot change the
  evaluation data, and simulating it would change nothing.
- The selection-noise null covers the same mechanism for the cell choice.

**P2 does not use the dev selection.** It compares every corner set with every corner single on
the evaluation data (the union bound).

**Criteria:**
- **nulls:** 5,000 datasets each, with 95% Clopper–Pearson intervals. The upper bound must be
  ≤ 0.06;
- **power:** 1,000 datasets per draw count.

**Pass 1** (`regime_calibration.json`) was built from the section 9 and 11 cells. It was run
before this rule.

| Null scenario | P1 | P2 | P3 | "Regime-dependent" reading |
|---|---|---|---|---|
| all equal, equal variances | 2.4% [2.0, 2.9] | 0.00% [0, 0.07] | 4.9% [4.3, 5.5] | 0.00% [0, 0.07] |
| all equal, unequal variances (corner ×2, one single and one set ×3) | 2.5% [2.1, 2.9] | 0.00% [0, 0.07] | 5.2% [4.6, 5.9] | 0.00% [0, 0.07] |
| selection noise (dev 3× noisier) | 2.8% [2.4, 3.3] | 0.02% [0, 0.11] | 4.9% [4.3, 5.5] | 0.00% [0, 0.07] |
| composite: other singles +0.2 pt | 2.9% [2.4, 3.4] | 0.02% [0, 0.11] | 5.1% [4.5, 5.7] | 0.00% [0, 0.07] |
| constant set advantage 1 pt, no trend | 3.0% [2.5, 3.5] | (power: 0.42%) | 4.9% [4.4, 5.6] | 0.26% [0.14, 0.44] |

Every upper bound is ≤ 0.06. P1 runs at about α/2 because of Holm with a near-1 P2. P2 is very
conservative: the union bound covers 5 singles.

| Power, 1 pt at the corner | 10 draws: P1 / P2 / both | 20 draws | 30 draws | P3 (10 / 20 / 30) |
|---|---|---|---|---|
| additive trend, equal variances | 81% / 80% / 68% | 94% / 99.5% / 93% | 97% / 100% / 97% | – |
| additive trend, unequal variances | 30% / 0.8% / 0.7% | 41% / 2.2% / 1.8% | 53% / 4.6% / 4.4% | – |
| interaction (both slow), equal variances | 82% / 78% / 68% | 93% / 99% / 92% | 98% / 100% / 98% | 58% / 76% / 82% |

**Reading of the power table.** Power depends strongly on how unequal the cells' noise is.
- When one corner cell is three times noisier than the others, P2 cannot show a 1-point
  advantage even at 30 draws. Its own noise leaves it unclear whether the set beats it.
- Pass 2 decides with the map's own dev noise.

**Pass 2 (after dev tuning, no primary data).**
- **Dev:** the regime map's own dev cells per regime.
- **Primary:** the pass 1 primary structure, with each regime's base rate and deviation spread
  scaled as on dev.

**Sample-size rule:**
- the evaluation uses the smallest of 10 / 20 / 30 draws with ≥ 80% power for P1 and for P2 (additive
  trend, 1 pt);
- it is fixed in a committed addendum (`regime_sample_size.json`) before any primary run;
- if 30 draws do not reach 80%, the run uses 30, and a primary null reads "inconclusive";
- if pass 2 looks like the unequal-noise scenario, the "regime-dependent benefit" reading is out
  of reach even at 30 draws. The addendum says so before the run;
- the draw count is never changed after any primary outcome has been seen;
- the sample size is never increased for P3.

**If a test fails the null criterion in pass 2:** this is disclosed in the addendum, and the
test cannot support a positive reading. It is read as not significant.

**GPU.** The sign-flip products are 2.1× faster on CUDA and identical. They are 8% of a dataset's
time, so the whole pipeline would gain at most 1.04×, below the 2× rule. The calibration stays
on CPU.

### 14.7 Reading rule (at-fault)

| Outcome | Reading |
|---|---|
| P1 and P2 significant (Holm) | **"regime-dependent benefit"**, plus one of: "the factors interact (secondary)" if P3 p < 0.05; "additive" if P3 is adequately powered (≥ 80% in pass 2) and not significant; otherwise "interaction inconclusive" |
| only P2 | "benefit in the adverse corner, no demonstrated trend" |
| only P1 | "trend without a demonstrated benefit" |
| neither | "no regime-dependent benefit" only if power was reached **and** corner headroom (best oracle − best single) is ≥ 1 pt. Otherwise "inconclusive" (power) or "no headroom" (oracle) |

**Every reading reports the whole map.** A significant corner result does not imply that the
corner is the only regime where the set helps.

### 14.8 Gates and checks run before this rule (engineering)

**Gate B, intent leakage** (`tests/test_regime.py`). Paired worlds have identical histories up to
t − Δ but different current positions and timing rates in (t − Δ, t]. They give:
- identical posterior weights;
- futures that differ only by the anchor shift, with identical increments.

Worlds with the same current position reached at different rates give identical predictor
output. They also give identical lead tracks and penetrations in the planner.

**Code audit:** every planner read of `pos_now` / `head_now` is the current state (collision
geometry and the predicted track's starting point). Speeds and headings come from the predicted
increments.

**Side effect of anchoring:** in the frozen world, a live alternative branch could start up to
0.3 m from the current position, and the planner's lead speed absorbed that offset. Anchoring
removes it.

**Resolution gate** (log-only, `regime_resolution.json`). Dev scenes, draws 90–94; the Ideal
set's own output.

| Δ | Median resolution | Shift vs Δ 0 (min / median) | 0.2 s replans with ≥ 2 futures | Resolved in the window |
|---|---|---|---|---|
| 0 s | 0.6 s | – | 21% | 97% |
| 1 s | 1.6 s | 1.0 / 1.0 s | 40% | 96% |
| 2 s | 2.6 s | 2.0 / 2.0 s | 59% | 93% |

Gate (median ≥ Δ-0 median + Δ − 0.1 s): **passes** at every level.

**Anchoring at Δ 0 is not a no-op** (`regime_anchor_check.json`: dev draws 95–96; 19,173
target replans, of which 4,245 have two or more live branches).
- **The realised branch is unchanged.**
  - The Branch oracle's prediction never changes.
  - Its 352 episodes are identical with and without anchoring.
- **Only live alternative futures move.**
  - Per target, the largest shift over the 6 s prediction has a median of 0.36 m (p95 0.93 m,
    max 2.6 m; `regime_resolution.json`). The 0.3 m rule bounds only the current position, and
    a log-time offset grows where the logged speed changes.
  - Before a branch starts, every branch shares the realised timing, and nothing moves.
- **The single-future predictions change too, but only while two or more branches are live:**
  - **the most likely:** only when its branch is not the realised one (18% of ambiguous replans;
    never otherwise);
  - **the mean:** 41% of ambiguous replans, p95 0.27 m. In 0.80% the mean jumps by more than any
    branch future moves (2.18 m; max 35 m). Every such jump happens because one branch's future
    runs past the end of the log and drops out of the average; without that, the mean moves at
    most 1.45 m. The same dropout happens in the frozen world, and anchoring moves where it
    happens. Within the first second the change is at most 1.2 m.
- **Collision geometry is unchanged.** Only the ideal predictors read branch futures. The world's
  motion, the collision checks and the at-fault rule never do.

**Pilot** (`regime_pilot.json`, dev draws 95–96; the confidential planner's pilot is recorded with
the confidential planner):

| Commitment | Proposal planner |
|---|---|
| 0.2 s | 1.64 episodes per CPU-second |
| 0.5 s | 2.49 |
| 1 s | 3.26 |
| Peak memory per worker | ≈ 108 MB |

- **Projected compute, both planners:** about 2.2 h of tuning and 3.3 h of evaluation at 10
  draws (6.5 h at 20, 9.8 h at 30), on 28 workers.
- **Oracle:** identical at Δ 0 and 2 s in every column, for both planners.
- **Replans:** every episode replanned exactly every R.

**Identity.** With every regime flag off, the episodes reproduce the frozen simulator:
- `check`: 36/36 timing, 40/40 lateral;
- `proposal_planner.py check`: 48/48;
- Part 1b's public episodes (primary draws 20–21, 6 scenes): 16/16.

**The baseline cell is not section 13's world.** At Δ 0, R 0.2 s, three changes act together:
anchoring, exact commitment and the hold-aware proposals. The table shows each one switched on
alone, over 352 dev engineering episodes (draws 95–96, margin 0). Each entry is the share of
episodes identical to the frozen simulator on every logged field, then the score and the at-fault
episodes.

| Switched on | Branch oracle | Ideal single (mean) | Ideal set |
|---|---|---|---|
| nothing (frozen) | 100%, 0.858, 4 | 100%, 0.864, 3 | 100%, 0.865, 2 |
| anchoring only | 100%, 0.858, 4 | 95%, 0.864, 3 | 92%, 0.866, 2 |
| hold-aware proposals only | 91%, 0.862, 4 | 90%, 0.869, 2 | 88%, 0.862, 3 |
| exact commitment only | 57%, 0.862, 4 | 55%, 0.864, 3 | 40%, 0.865, 2 |
| all three (the baseline cell) | 56%, 0.865, 4 | 51%, 0.871, 1 | 38%, 0.862, 3 |

- **The differences from the old world cannot be attributed to anchoring alone.** Exact
  commitment and the hold-aware proposals are also substantial changes.
- **Exact commitment** changes the most episodes, each by little. On its own it moves the mean
  single's score by −0.0002 and leaves its at-fault count unchanged.
- **The hold-aware proposals** cause most of the metric changes.
- **Anchoring** is the smallest of the three.
- **Checks** (`regime_anchor_check.json`):
  - with all three off, the episodes equal the frozen simulator;
  - with all three on, they equal the map's own regime path.
- The pilot's earlier figure, 86.9% identical, compared only score and at-fault.
- Each cell has 1 to 4 at-fault episodes, so these figures are descriptive only.

**Consequence:** this map's absolute results are not directly comparable with sections 11 and
13 (Part 1b). Every test compares cells within the map, which all share the same world.

### 14.9 Disclosures

- The fifth use of the 73 primary scenes; fresh draws.
- The 0.2 s column is a 5 Hz baseline, not a 10 Hz production system.
- The rows vary information, not behaviour. The recognition delay models intent that is read
  late; the road users' behaviour is unchanged.
- **Simulator limits as before:**
  - timing branches only;
  - the branches don't react to the ego;
  - the other agents don't react at all;
  - the ideal predictors know the branch model and its prior exactly.
- The confidential planner's setting is frozen at section 10, not retuned per regime.
- Margins are tuned on at-fault rate first, not on score as in section 10 (14.3).
- The baseline cell differs from sections 11–13's world (14.8). Absolute rates are not compared
  across them.
- Dev selection uses 880 episodes per cell, so the choice is noisy. Calibration includes that
  noise.
- P2 is conservative (null rejection ≈ 0%), and its power depends on how unequal the cells' noise
  is (14.6).

### 14.10 Runners

| Step | Runner (`counterfactual_sim.py`) | Output |
|---|---|---|
| Gate A | `draw-provenance` | `draw_provenance.json` |
| Resolution gate | `regime-resolution` | `regime_resolution.json` |
| Pilot | `regime-pilot` | `regime_pilot.json` |
| Anchoring check | `regime-anchor-check` | `regime_anchor_check.json` |
| Calibration pass 1 | `regime-calibration` | `regime_calibration.json` |
| Dev tuning (public cells) | `regime-tune` | `regime_tune.json` |
| Evaluation (public cells) | `regime-map` | `regime_map.json` |

The confidential runner adds its cells, the dev selection, calibration pass 2 with the addendum,
and the tests and reading.

**Order:**
1. This rule is committed before dev tuning.
2. Then dev tuning and calibration pass 2.
3. The draw-count addendum is committed before any primary run.

Related: [[negative-results]], [[planner]]
