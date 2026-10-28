# Reactive Closed-Loop Test — does multi-future prediction help when traffic reacts?

## Status: 📌 Pre-registered (2026-10-29) — no reactive result has been looked at

This rule is committed **before** any package is driven in the reactive simulator on the
scenes below. Only engineering checks ran first (`reactive_sim.py check` / `verify-online`,
the unit tests, and a timing run on 2 dev scenes with sim seed 99, which printed no scores).
The result JSON records this commit.

## 1. Why

In the non-reactive replay ([[closed-loop-cv]], [[planner]]) the other road users follow
their logs whatever the ego does. On the development scenes a perfect predictor (the oracle)
scores 0.901 and constant velocity also scores 0.901. So no predictor, single or
multi-future, can show a safety advantage there.

In real mixed traffic, agents yield, hesitate or push on. Their futures then depend on what
the ego does, which is where several futures could matter. This test adds such reactions.

## 2. Simulator ([`scripts/reactive_sim.py`](../../scripts/reactive_sim.py))

**Agent paths**
- Every agent follows its **logged path**.
- A **non-reactive** agent replays its log exactly.

**Reactive agents**
- In each episode, each agent is reactive with probability **p_react**.
- The draw is a hash of (scene, start, persona, sim seed), so it is identical for every
  package and nested in p_react.
- A reactive agent runs **IDM along its path**, with the ego as its leader. The ego counts
  when its footprint, plus a 0.3 m margin on the corridor half-width, lies on the agent's
  next 4 s / 40 m of logged path.
- The agent sees the ego with a **0.4 s delay**.
- IDM's interaction term is subtracted from the acceleration that tracks the logged speed,
  so an unconstrained agent keeps to its log.
- Once delayed, the agent continues its log from where it is (time warp), so its lag
  persists.

**IDM parameters, fixed a priori**

| Class | a_max | b_comf | b_max (m/s²) | Headway T (s) | s0 (m) |
|---|---|---|---|---|---|
| Cars, buses, trucks, other | 1.5 | 2.0 | 6 | 1.0 | 1.0 |
| Two-wheelers | 2.0 | 2.5 | 6 | 0.8 | 0.5 |
| Pedestrians, children, animals | 1.0 | 1.5 | 3 | 0.5 | 0.5 |

**Not modelled:** lateral evasion, reactions between agents, and anything ego-unrelated.

**Online prediction**
- While every agent is still on its log, the saved predictions are used. This is exact,
  because the inputs are identical.
- Once any agent deviates, every model predicts **online** from the simulated 2 s
  histories, using the same windows and the same calibrated bands.
- Online outputs reproduce the saved ones: CV and AgentFormer most-likely exactly; other
  single trajectories within 8 mm; sets within float16 rounding of the saved files.

**Everything else is unchanged:** the instrument, episodes, personas, episode score, and
planner v2 (band ×0, 1 s hard constraint, soft weight 5, least-penetration fallback).
- **With p_react = 0, every episode equals the non-reactive one** (`check`).

## 3. Packages

| Group | Packages | Planner |
|---|---|---|
| Single trajectories | CV, ST-GAT, K-mode top-1, multi-future transformer top-1, AgentFormer most-likely | planner v2 |
| Multi-future sets (10 futures) | K-mode set, multi-future transformer set, AgentFormer set | the probability-aware planner, pre-registered separately (2026-10-28, confidential), at its frozen chosen setting |
| References | log oracle (each agent's logged future from its current log time: intent, not reactions), ignore agents, human replay | planner v2 / log |

**Not driven:**
- **Trajectron++:** its median latency is 213 ms for 10 samples, slower than the 0.2 s
  replan. It also needs its own scene-graph export, so it can't predict online.
- **LED:** 4.2 s per window on CPU, so online prediction is impractical.

Both stay in the four-criterion scorecard (section 6) and in the non-reactive results.

## 4. Scenes, models, seeds

- **Pilot:**
  - the 32 development scenes (frozen val + test);
  - frozen-split models, seed 0;
  - p_react = 0.5, sim seeds 0, 1, 2.
- **Confirmatory:**
  - the **73 primary scenes** of [[closed-loop-cv]], each driven by the fold model of the
    fold in which it is a test scene (seed 0);
  - p_react = 0.5, sim seeds 0, 1, 2.
  - **This is the third analysis on these scenes,** after the closed-loop v2 test and the
    planner study. The reactive simulator is new, but the reuse is disclosed.
- **Sensitivity:** p_react = 1.0 on the confirmatory scenes, reported but not used for any
  decision.

## 5. Gate, hypotheses, decision rules

**Pilot gate** (pooled over the 3 sim seeds): the confirmatory run happens only if at least
one set beats its own single trajectory on the dev scenes by either:
- ≥ 0.01 in mean episode score; or
- ≥ 1 point fewer at-fault collision episodes.

If no set passes, the pilot is the reported outcome: multi-future prediction shows no
measurable advantage even with reactive agents, and the primary scenes are not used.

**Confirmatory hypotheses** (primary scenes, p_react = 0.5):
- one-sided paired scene bootstrap, 2,000 replicates;
- each scene's episodes from all 3 sim seeds are pooled, and the scene is the resampling
  unit;
- **Holm across all 18.**

For each set S ∈ {K-mode, multi-future transformer, AgentFormer} and each comparator
C ∈ {S's own single trajectory, CV, ST-GAT}:
- **(i)** S has fewer at-fault collision episodes than C;
- **(ii)** S has a higher mean episode score than C.

**Reading, fixed now.** "Multi-future prediction improves safety in reactive mixed traffic"
is **supported** only if, for at least one set S, both of these hold:
- (i) is supported against all three of its comparators;
- the 95% CI of S's score difference to each comparator has a lower bound ≥ −0.01. Safety
  must not be bought with lost progress.

Otherwise it is **not supported**. If (ii) also fails throughout, the report says "no worse"
only where every score CI lower bound is ≥ −0.01.

**Lead multi-future model.** Criterion 4 (driving usefulness) decides:
1. Eligible sets: those with median GPU latency ≤ 200 ms (the replan interval). The latency
   comes from `safety_latency.json`, or `multifuture_scorecard.json` for the multi-future
   transformer.
2. Leader: the lowest at-fault rate on the primary scenes (pilot scenes if the gate fails).
3. Tie: sets whose paired at-fault CI vs the leader includes 0.
4. Tie-breaks, in order:
   1. at-fault collisions with pedestrians / two-wheelers;
   2. the energy score (section 6, lower is better);
   3. latency.

**Reported, not decisive:**
- near-miss, harsh braking, jerk and progress;
- agents that deviated from their log per episode;
- the "headroom" (log oracle − CV), reactive vs non-reactive;
- the human replay's at-fault rate (plausibility: it should not rise);
- learned single trajectories vs CV;
- the p_react = 1.0 sensitivity.

## 6. Four-criterion profile (open loop, reported)

[`scripts/multifuture_scorecard.py`](../../scripts/multifuture_scorecard.py) →
`outputs/benchmark_v2/multifuture_scorecard.json`. It covers K-mode, the multi-future
transformer, AgentFormer, LED and Trajectron++, on frozen test windows, 3 seeds, K = 10,
with seed-pooled scene-bootstrap CIs.

| Criterion | Metrics |
|---|---|
| 1. Prediction quality | most-likely ADE / FDE, minADE@10 / minFDE@10 |
| 2. Diversity / coverage | miss rate@10 (minFDE > 2 m), endpoint spread, spread ÷ most-likely FDE, implausible-sample rate |
| 3. Probability quality | energy score (all; probability-weighted where a model has probabilities); NLL of the best mode, top-mode accuracy and ECE for K-mode and the multi-future transformer |
| 4. Driving usefulness | this test |

**Result (2026-10-29).** Frozen test windows, 3 seeds, K = 10. Lower is better except where noted.

| Model | ML ADE / FDE | minADE / minFDE@10 | Miss@10 | Spread (m) | Implausible | Energy | Top-mode acc. ↑ / ECE | GPU ms |
|---|---|---|---|---|---|---|---|---|
| K-mode decoder | 4.15 / 9.21 | **2.01 / 3.56** | 0.47 | 42.6 | 0.40 | **3.59** | 0.52 / 0.11 | 19 |
| Multi-future transformer | 3.98 / 8.68 | 2.47 / 4.88 | 0.50 | 52.8 | 0.21 | 3.96 | 0.66 / 0.13 | 5 |
| AgentFormer | 3.75 / 8.22 | 2.43 / 4.65 | 0.41 | 6.6 | **0.15** | 3.97 | — | 102 |
| LED | **3.75 / 8.13** | 2.49 / 5.39 | **0.40** | 7.9 | 0.46 | 3.96 | — | 43 |
| Trajectron++ | 3.91 / 8.38 | 2.38 / 4.40 | 0.41 | 6.8 | 0.42 | 3.75 | — | 213 ✗ |

- **No model wins on every criterion.**
  - AgentFormer / LED have the best most-likely trajectories.
  - The K-mode decoder has the best coverage (minFDE@10) and energy score.
  - AgentFormer has the fewest physically implausible samples.
- **The energy-score CIs overlap widely**, e.g. K-mode [2.43, 5.16] vs AgentFormer [2.56, 5.75],
  so criterion 3 doesn't separate the models.
- **The two K-mode models fan out much more** (42–53 m mean endpoint spread). About 40% of their
  modes carry under 1% probability. Unweighted spread counts those modes equally, while the
  probability-weighted energy score discounts them.
- **Their top-mode probabilities are moderately miscalibrated** (ECE 0.11–0.13).
- **Trajectron++ fails the real-time filter** (213 ms > 200 ms).

Source: `outputs/benchmark_v2/multifuture_scorecard.json`.

## 7. Caveats (stated in advance)

- The reaction model is simple and uncalibrated: IDM along logged paths, no swerving, no
  agent–agent reactions. Its parameters are set a priori, with no Indian-traffic reaction
  data.
- One model seed per fold. Sim seeds vary only who reacts.
- The planners were tuned in the non-reactive simulator.

Related: [[closed-loop-cv]], [[planner]], [[closed-loop-test]], [[safety-scorecard]]
