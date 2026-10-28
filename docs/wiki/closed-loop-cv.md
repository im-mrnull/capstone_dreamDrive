# Closed-Loop Test v2 — cross-validated, fixed instrument, fixed planner

## Status: ✅ Complete (2026-10-28) — primary result and amendment 3 in

The first closed-loop test ([[closed-loop-test]], 22 scenes) could not separate the predictors.
It also showed three problems:
- a ~6% false-collision floor;
- 90% bands that paralysed the planner;
- too few scenes.

This protocol fixes all three. It was committed **before** any fold model drove any of the
primary scenes. The result JSON records the commit as `rule_commit`.

## 1. Instrument v2 (collision measurement)

All of these were calibrated on the recorded human driver's replay only, so they involve no
predictor.

- **Ego-pose glitches:** the motion reconstruction sometimes jumps the ego more than 1 m off
  its 9-frame median path. Episodes containing such a frame are excluded for every package.
- **Ego orientation:** taken from the logged pose yaw plus the measured axis offset, both for
  the human replay and for the route's orientation along its length. This is the frame the
  annotations were made in; the path tangent is noise at walking pace.
- **Exact oriented rectangles** replace the circle approximations for ground-truth contacts.
- **Ego footprint 3.4 × 1.7 m, centred 0.3 m ahead of the sensor.** This is the region no
  annotated agent edge enters around the sensor at a robust percentile, slow frames included.
  The planner still plans with its larger circle footprint.
- **Collision tolerance:** a contact counts beyond 0.25 m of overlap.
- **Result:** the human replay shows at-fault contacts in 0/72 val, 0/152 test and 6/524
  former-train episodes (1.1%), down from 0%, 5.7% and 10.3%. Replaying the human's timing on
  the smoothed route gives 8/524.

## 2. Planner v2 (band / planner interaction)

A 30-configuration grid ran on the 32 development scenes (frozen val + test) with the 8
original single-trajectory models at seed 0
([`closed_loop_dev.py`](../../scripts/closed_loop_dev.py),
`outputs/benchmark_v2/closed_loop_dev_grid.json`). Its parameters:
- hard-constraint horizon 0.5 / 1 / 3 s;
- soft penalty beyond it, weight 0 / 5;
- fallback: least penetration or straight emergency brake;
- band scale 0 / 0.5 / 1.

The criterion was fixed before the run: the highest mean episode score across the 8 models.
**Chosen: band ×0, hard constraint 1.0 s, soft weight 5, least-penetration fallback.**

On dev, every band setting scored lower:
- the best ×1 configuration: 0.813;
- the best ×0.5: 0.862;
- ×0: 0.894.

Harsh braking fell from 37–50% to about 9%.

## 3. Cross-validation

- 5 folds over all 105 scenes ([`make_cv_folds.py`](../../scripts/make_cv_folds.py)): md5 rank,
  dealt round-robin.
- Per fold: 21 test, 10 val (model selection + conformal calibration), 74 train.
- Models: CV, MLP, ST-GAT, map-free transformer, LED (mean-denoised). One seed per fold,
  trained with the exact benchmark-v2 settings.
- Every scene is driven by the models of the fold in which it is a test scene.
- **Primary set: the 73 scenes of the frozen train split.** No closed-loop result was ever
  looked at on them.
- Development set (32 scenes) and all 105 scenes are reported separately.
- References (not eligible): human driver replay, oracle (true future), ignoring agents.

## 4. Metrics and decision rule

The episodes, personas and episode score are unchanged from [[closed-loop-test]]: 0 after an
at-fault collision, otherwise (5·progress + 2·no-harsh-braking + 2·no-near-miss) / 9.

**Pick:** on the primary set, the highest mean score. Packages whose paired 95% scene-bootstrap
CI vs the leader includes 0 are tied. Ties are then broken by the lowest rate of at-fault
collisions with pedestrians or two-wheelers (paired), then by latency.

**Pre-registered hypotheses** (primary set, one-sided paired scene bootstrap, Holm-corrected
across all 16):
- **H1:** ST-GAT, the transformer and LED each score higher than CV.
- **H1b:** the same three each have fewer at-fault collision episodes than CV.
- **H2:** each of the 5 packages has fewer at-fault collision episodes than ignoring agents.
- **H3:** for each package, the chosen band setting (×0) scores higher than the full 90% band
  (×1), with the same planner.

## 5. Amendment (2026-10-27, before any primary result): instrument sensitivity

Every instrument setting above was calibrated on the human replay. To show the conclusions
don't hinge on those exact values, the same pre-registered analysis (primary set, same
hypotheses, same decision rule) is re-run with one setting varied at a time
(`python scripts/closed_loop_cv.py sensitivity` → `closed_loop_cv_sensitivity.json`):

| Setting | Variants (primary in bold) |
|---|---|
| Collision tolerance | 0, **0.25**, 0.5 m |
| Ego footprint | **3.4 × 1.7 m, centred +0.3 m (measured)**; 4.5 × 1.7 m, centred +0.5 m (nominal car) |
| Pose-glitch threshold | 0.5, **1.0**, 2.0 m |
| Agent footprint sizes | −20%, **nominal**, +20% |

The headline stays the primary analysis. For each hypothesis the report states in how many of
the 7 variants it remains supported. A hypothesis is called **robust** only if it is supported
in the primary analysis and in every variant.

## 6. Amendment 2 (2026-10-27, before any primary result): multi-future packages

The project's subject is multi-future prediction, so the confirmatory test also drives three
multi-future packages:
- the K-mode decoder, top-1-selected version: its top-1 and its 10 modes. It is trained per
  fold with the benchmark settings (`--k 10 --select top1`).
- LED's first 10 of 20 leapfrog samples. This is the same fold model as LED (mean-denoised).

Each set's band is calibrated on the fold's val scenes from the min-over-samples error. The
planner, its configuration and the band settings (×0 chosen, ×1 reported) are unchanged.

These are a **secondary** analysis. They don't enter the decision rule or the primary
hypothesis family. Their own family is Holm-corrected separately, on the primary set:
- **S1:** each set scores higher than its own single trajectory (K-mode set vs K-mode top-1;
  LED set vs LED mean).
- **S2:** each set has fewer at-fault collision episodes than its own single trajectory.

These test whether multi-future output helps *this* planner, which treats every sample as an
obstacle. A probability-aware multi-future planner is the separate, later step.

## 7. Amendment 3 (2026-10-27, before any of these models is trained): all multi-future models

To answer whether multi-future prediction is needed for safety, three more multi-future
models are trained per fold and driven in the same closed loop. They use the same scenes,
planner, instrument and band settings (×0 chosen, ×1 reported). Each is a single trajectory
plus its 10-future set, and their bands are calibrated per fold:
- **Multi-future map-free transformer** (`model8`): the transformer backbone with 10 output
  heads and a learned probability per future. It is trained like the K-mode decoder:
  best-of-K loss + 0.1 × probability cross-entropy, selected on val top-1, 300 epochs.
- **AgentFormer:** the benchmark settings (2 Hz, 100 epochs), with 10 prior samples.
- **Trajectron++:** the benchmark export and settings (100 epochs), re-exported per fold, with
  10 samples.

They are run by `closed_loop_cv.py train-extra / prepare-extra / run-extra`, with results in
`closed_loop_cv_extra.json`. Comparisons pair with the primary run's episodes on the
**primary set**.

This is a separate secondary family (Holm-corrected within itself, outside the decision
rule):
- **S1 / S2:** each new set has a higher score than / fewer at-fault collision episodes than
  its own single trajectory (3 models × 2 tests).
- **T1 / T2:** the multi-future transformer's top-1 has a higher score than / fewer at-fault
  collisions than the single-output map-free transformer. In other words, does training with
  multiple futures help the deployable output?

The ST-GAT backbone already has a multi-future version, the K-mode decoder (amendment 2), so
no separate multi-future ST-GAT is built.

These tests use the existing planner, which treats every future as an obstacle. Whether
*probability-aware* multi-future planning helps is the next, separately pre-registered study.

## 8. Results (primary set: 73 scenes, 524 episodes per package)

Full tables: README "Closed-loop test v2" (generated from `outputs/benchmark_v2/closed_loop_cv.json`
and `closed_loop_cv_sensitivity.json`). The rule commit is `24684e2`.

- **Pick: CV.** All 5 single-trajectory packages are tied on score (0.793–0.802). Tie-break 1
  leaves all of them tied; ST-GAT is lowest on pedestrian / two-wheeler at-fault collisions
  (1.3%), but not separated. Tie-break 2 is latency.
- **H1 / H1b:** not supported for any learned model.
- **H2:** supported for all 5 (16.4% → 2–3% at-fault).
- **H3:** supported for all 5 (≈ 0.80 without the band vs ≈ 0.66 with it).
- **Sensitivity:** the pick is CV in all 7 variants. H2 and H3 are supported in 7/7 (robust);
  H1 / H1b in 0/7.
- **S1 / S2 (multi-future sets, planner v2):** not supported. Both sets are *significantly
  worse* than their own single trajectory:

  | Set | Score (vs its single) | At-fault (vs its single) |
  |---|---|---|
  | K-mode | 0.626 vs 0.791 | 13.4% vs 4.6% |
  | LED | 0.682 vs 0.793 | 7.4% vs 3.4% |

  Every 95% CI excludes zero in the harmful direction.
- **References:**
  - the oracle scores 0.821, so the planner is the limit on these scenes;
  - the human replay scores 0.934;
  - ignoring agents scores 0.729.
- **Development and all-105 sets:** the pick is CV in both.
- **Amendment 3** (`closed_loop_cv_extra.json`; primary set, planner v2, band ×0):

  | Model | Single trajectory: score / at-fault | 10-future set: score / at-fault |
  |---|---|---|
  | Multi-future transformer | 0.790 / 3.6% | 0.438 / 19.3% |
  | AgentFormer | 0.800 / 2.9% | 0.793 / 3.4% |
  | Trajectron++ | 0.799 / 3.4% | 0.752 / 4.4% |

  - **S1 / S2:** not supported for any of the three. The multi-future transformer set is
    significantly worse on both score and collisions, and the Trajectron++ set on score. The
    AgentFormer set is level with its single trajectory (score CI [−0.023, 0.008]).
  - **T1 / T2:** not supported. The multi-future transformer's top-1 is no better than the
    single-output transformer (score CI [−0.026, 0.005]).
  - Across all five multi-future models, no set beats its own single trajectory under
    planner v2, and harsh braking rises with how much the set blocks: 25% (AgentFormer) up
    to 93% (multi-future transformer).

**Caveats:**
- the agents don't react;
- footprints are nominal;
- there is one seed per fold;
- the planner was tuned on the development scenes (excluded from the primary set).

Related: [[closed-loop-test]], [[safety-scorecard]], [[bootstrap-ci]]
