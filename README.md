# DreamDrive — Trajectory Prediction on IDD-3D (Chaotic Mixed Traffic)

Multi-future trajectory prediction for unstructured Indian traffic, benchmarked on the
[IDD-3D](https://idd3d.github.io/) dataset (105 scenes, 10 fps, 22 agent classes in the benchmark).
Dataset documentation: [`DATASHEET.md`](DATASHEET.md).
Observation window 2.0 s (20 frames) → prediction horizon 6.0 s (60 frames), `dt = 0.1 s`.

**Capstone project PW26_CGM_04** — Idhant Singh, Rishav Ghosh, Shagal Verma, Shourya Rai (PES University).

---

## Results integrity policy

Every number in the results tables below is generated from a committed metrics JSON by
[`scripts/make_tables.py`](scripts/make_tables.py); CI fails if a table is hand-edited
(`python scripts/make_tables.py --check`). A result without an artifact does not appear here.

**Retractions (July 2026).** Earlier versions of this repository and the Phase-2 report
contained results that were not backed by any experiment artifact, including a claimed
66%/131% ADE/FDE degradation of Trajectron++ on collision data (a placeholder — the
experiment was never run), a benchmark table whose values matched no artifact, and a
Trajectron++ "correction" that divided metrics by 10 without a new evaluation. All are
retracted; see `research/report.txt` §7.2–7.3 for the retraction text.

**Corrections (October 2026).** The conformal evaluation's "valid" regime split test
*agents* rather than test *scenes*, which leaked correlated agents across the split and
overstated coverage. It now splits scenes and averages over 100 random scene splits. Two
findings built on it, that conformal "fails under the val→test shift" and that "stronger
predictors survive the shift better", are withdrawn because a scene-level analysis does not
support them. See the conformal section below.

**Correction: Trajectron++ was scored on an easier population (October 2026).** Its
training script scores it at every valid node-timestep (23,570 test instances, many late in a
scene with under 2 s of future left), while every other model is scored on the 2,177
canonical test windows. Re-scored on the canonical windows from the same trained checkpoints
([`scripts/eval_trajectronpp_canonical.py`](scripts/eval_trajectronpp_canonical.py)), its
deployable test ADE is 3.91 ± 0.08 m, not 2.58 ± 0.04 m. Three claims are withdrawn:

- that Trajectron++ was the best deployable model;
- that the scene bootstrap separated it from ST-GAT;
- that it had the tightest conformal band (±10.0 m) and lost the least coverage from val
  to test (5.2 points). On the canonical windows its band is 13.8 m, mid-table.

The dense-protocol rows stay in the table for reference, labelled and never ranked. A test
now checks that every ranked row and every conformal input is scored on the same population
([`tests/test_eval_population_contract.py`](tests/test_eval_population_contract.py)).

## The ego-motion correction (read this first)

Raw IDD-3D labels store `psr.position` in the **per-frame ego/LiDAR frame**. Uncompensated,
every trajectory mixes agent motion with ego motion — median apparent *pedestrian* speed was
**5.05 m/s** (impossible). [`scripts/0_compensate_ego_motion.py`](scripts/0_compensate_ego_motion.py)
estimates per-frame SE(2) ego motion from the annotations themselves (RANSAC rigid alignment;
static objects such as parked vehicles are the inliers) and re-expresses all positions in a
per-scene world frame. After compensation the median pedestrian speed is **0.91 m/s**
(walking pace). Diagnostics: [`outputs/ego_compensation_stats.json`](outputs/ego_compensation_stats.json);
regression tests: [`tests/test_speed_sanity.py`](tests/test_speed_sanity.py).

Consequences:

- All **legacy ego-frame results are deprecated** (kept below for the record only). They are
  not comparable to any published benchmark, and part of the learned models' apparent
  advantage in that frame came from implicitly learning ego motion.
- The compensation is an approximation pending official ego poses / LiDAR odometry
  (raw `.pcd` files are not in this repo). Fallback statistics are recorded per scene.

## Pipeline

```
IDD-3D label JSONs (per-frame, ego-relative)
  │ 1_parse_idd3d_positions.py         → outputs/parsed_positions/*.csv        (ego frame)
  │ 0_compensate_ego_motion.py         → outputs/parsed_positions_world/*.csv  (world frame)
  │ 2_consolidate_trajectories.py      → outputs/trajectories_world/trajectories.csv
  │ 3_make_windows_obs20_pred60.py     → outputs/windows_world_obs20_pred60.npz     (4,820 windows)
  │ 4_export_multiagent_trajpp_format.py → outputs/multiagent_world_v1/scenes/*.npz (105 scenes)
  │ 5_export_stgat_format.py           → outputs/stgat_dataset_world_obs20_pred60.npz (608 seqs, 4,443 agents)
  │ 6_export_trajectronpp_world.py     → outputs/trajectronpp_world/{train,val}.pkl  (dt=0.1)
  └ models: model1 (CV) · model2 (MLP) · model3 (ST-GAT-inspired) · model4 (K-mode decoder)
            7_train_eval_trajectronpp.py (upstream Trajectron++)
```

## Results — Benchmark v2 (canonical dataset, frozen test set)

The current headline benchmark. Protocol:

- **Data:** [`outputs/canonical_world_obs20_pred60.npz`](outputs/canonical_world_obs20_pred60_stats.json)
  — world-frame, 718 sequences, **10,576 agents** (the legacy all-80-frames survivor filter
  discarded 58% of predictable agents; agents now need only a full 2 s observation + ≥1 future
  step, with per-timestep `fut_mask`). Every model evaluates on this same population.
- **Split:** frozen [`splits/scene_splits.json`](splits/scene_splits.json) (73/10/22 scenes,
  hash-based, committed). Model selection uses val only; **test numbers are final**.
- **Seeds:** {0,1,2} affect model init/shuffling only, never the split. Cells are mean ± std.
- **Metrics:** masked ADE over valid future steps; FDE at each agent's last valid step;
  sum-and-divide aggregation. Oracle rows select the best mode *after seeing ground truth* —
  upper bounds, never comparable with deployable rows.

<!-- BEGIN AUTOGENERATED: benchmark v2 results (scripts/make_tables.py) -->
| Model | K | Deployable single output? | Test ADE (m) | Test FDE (m) | Val ADE (m) | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| CV (vel_k=5) | 1 | yes | 4.36 | 9.78 | 2.43 | 0 |
| MLP | 1 | yes | 3.97 ± 0.01 | 8.66 ± 0.05 | 2.31 ± 0.01 | 0,1,2 |
| ST-GAT-inspired (MHA) | 1 | yes | 3.41 ± 0.07 | 7.35 ± 0.16 | 2.38 ± 0.03 | 0,1,2 |
| Map-free transformer | 1 | yes | 3.57 ± 0.11 | 7.85 ± 0.18 | 2.46 ± 0.02 | 0,1,2 |
| K-mode decoder — oracle best-of-10 | 10 | no (oracle) | 1.88 ± 0.05 | 3.18 ± 0.11 | 1.32 ± 0.03 | 0,1,2 |
| K-mode decoder — learned top-1 | 10 | yes | 4.31 ± 0.05 | 9.48 ± 0.24 | 3.38 ± 0.36 | 0,1,2 |
| K-mode decoder (top-1-selected) — oracle best-of-10 | 10 | no (oracle) | 2.01 ± 0.07 | 3.56 ± 0.19 | 1.69 ± 0.09 | 0,1,2 |
| K-mode decoder (top-1-selected) — learned top-1 | 10 | yes | 4.15 ± 0.10 | 9.21 ± 0.26 | 2.61 ± 0.13 | 0,1,2 |
| Trajectron++ — most-likely | 1 | yes | 3.91 ± 0.08 | 8.38 ± 0.24 | 2.73 ± 0.14 | 0,1,2 |
| Trajectron++ — oracle best-of-20 | 20 | no (oracle) | 2.08 ± 0.02 | 3.64 ± 0.04 | 1.35 ± 0.04 | 0,1,2 |
| Trajectron++ hetero (unicycle) — most-likely | 1 | yes | 4.55 ± 0.17 | 10.43 ± 0.54 | 3.13 ± 0.18 | 0,1,2 |
| Trajectron++ hetero (unicycle) — oracle best-of-20 | 20 | no (oracle) | 2.22 ± 0.06 | 4.12 ± 0.06 | 1.39 ± 0.07 | 0,1,2 |
| AgentFormer (upstream, map-free, 2 Hz) — most-likely | 1 | yes | 3.75 ± 0.00 | 8.22 ± 0.27 | 2.44 ± 0.07 | 0,1,2 |
| AgentFormer (upstream, map-free, 2 Hz) — oracle best-of-20 | 20 | no (oracle) | 2.24 ± 0.33 | 4.13 ± 0.74 | 1.25 ± 0.16 | 0,1,2 |
| LED (leapfrog diffusion) — mean-denoised | 1 | yes | 3.75 ± 0.01 | 8.13 ± 0.09 | 2.03 ± 0.02 | 0,1,2 |
| LED (leapfrog diffusion) — oracle best-of-20 | 20 | no (oracle) | 1.98 ± 0.07 | 4.10 ± 0.18 | 1.03 ± 0.04 | 0,1,2 |
| Trajectron++ — most-likely [dense node-timesteps] | 1 | yes | 2.58 ± 0.04 | 5.58 ± 0.07 | 2.14 ± 0.07 | 0,1,2 |
| Trajectron++ — oracle best-of-20 [dense node-timesteps] | 20 | no (oracle) | 1.37 ± 0.01 | 2.39 ± 0.02 | 1.10 ± 0.02 | 0,1,2 |

Every row is scored on the same canonical test windows except those marked *[dense node-timesteps]*: Trajectron++'s own protocol, which predicts at every valid timestep and so includes many late-scene instances with under 2 s of future. Those rows are easier by construction, shown for reference only, and never ranked against the others.

Source artifact: [`outputs/benchmark_v2/summary.json`](outputs/benchmark_v2/summary.json) (aggregated from per-seed metrics JSONs in the same directory).
<!-- END AUTOGENERATED: benchmark v2 results -->

### How certain are these numbers? (scene-level bootstrap)

The ± above is seed-to-seed spread only. The larger uncertainty is *which* 22 scenes
landed in the test split. Agents within a scene share an ego drive, a layout and each
other's interactions, so they are not independent, and an agent-level CI would be far too
narrow. [`scripts/bootstrap_ci.py`](scripts/bootstrap_ci.py) resamples **test scenes**
with replacement (seed-0 model of each family, from its prediction dump). It reports
per-model CIs and **paired** differences: both models are scored on the same resampled scenes,
which removes the shared scene difficulty from the comparison. For the top models and LED it
also compares 3-seed means, resampling each model's training seeds as well as the scenes.

<!-- BEGIN AUTOGENERATED: bootstrap CIs (scripts/make_tables.py) -->
| Model (seed 0) | Test ADE (m) [95% CI] | Test FDE (m) [95% CI] | Test scenes | = seed-0 artifact? |
| --- | --- | --- | --- | --- |
| ST-GAT (MHA) | 3.33 [2.16, 4.98] | 7.16 [4.63, 10.62] | 22 | yes |
| Map-free transformer | 3.43 [2.19, 5.10] | 7.62 [4.86, 11.24] | 22 | yes |
| LED (mean-denoised) | 3.74 [2.33, 5.65] | 8.09 [5.00, 12.26] | 22 | yes |
| AgentFormer (2 Hz, most-likely) | 3.75 [2.41, 5.54] | 8.34 [5.27, 12.41] | 22 | yes |
| Trajectron++ (most-likely) | 3.89 [2.40, 5.91] | 8.41 [5.32, 12.62] | 22 | yes |
| MLP | 3.96 [2.49, 5.93] | 8.62 [5.50, 12.83] | 22 | yes |
| K-mode (top-1) | 4.05 [2.63, 6.08] | 9.05 [5.80, 13.35] | 22 | yes |
| CV (vel_k=5) | 4.36 [2.57, 6.80] | 9.78 [5.92, 14.99] | 22 | yes |

Paired comparisons of the seed-0 models adjacent in the ranking (same scene resample for both; negative = the first model is better):

| Comparison | ΔADE (m) | 95% CI | Separated at 95%? |
| --- | --- | --- | --- |
| ST-GAT (MHA) vs Map-free transformer | -0.10 | [-0.36, 0.13] | no |
| Map-free transformer vs LED (mean-denoised) | -0.31 | [-0.66, -0.08] | yes |
| LED (mean-denoised) vs AgentFormer (2 Hz, most-likely) | -0.01 | [-0.31, 0.30] | no |
| AgentFormer (2 Hz, most-likely) vs Trajectron++ (most-likely) | -0.15 | [-0.63, 0.23] | no |
| Trajectron++ (most-likely) vs MLP | -0.07 | [-0.36, 0.24] | no |
| MLP vs K-mode (top-1) | -0.08 | [-0.64, 0.49] | no |
| K-mode (top-1) vs CV (vel_k=5) | -0.31 | [-0.99, 0.21] | no |

Seed-pooled paired comparisons for the models with a dump per seed (3-seed means, seeds 0, 1, 2). Each replicate resamples the test scenes for both models and each model's seeds with replacement, so the CI covers training-seed variation as well:

| Comparison | ΔADE (m), 3-seed means | 95% CI | Separated at 95%? | Seed pairings where the first is better |
| --- | --- | --- | --- | --- |
| ST-GAT (MHA) vs Map-free transformer | -0.17 | [-0.48, 0.13] | no | 8 of 9 |
| ST-GAT (MHA) vs LED (mean-denoised) | -0.34 | [-0.76, 0.01] | no | 9 of 9 |
| Map-free transformer vs LED (mean-denoised) | -0.17 | [-0.55, 0.11] | no | 9 of 9 |

Where the two tables disagree, cite the seed-pooled row: a single seed can be a model's luckiest or unluckiest run.

Per-class ADE disparity for ST-GAT (MHA) (classes with n ≥ 50): **12.5×**, 95% CI 6.7–24.0× (worst Car, 20 test scenes; best Scooter, 6 test scenes).

Source artifact: [`outputs/benchmark_v2/bootstrap_ci.json`](outputs/benchmark_v2/bootstrap_ci.json) (10,000 scene resamples of the 22 test scenes).
<!-- END AUTOGENERATED: bootstrap CIs -->

How to read it: absolute test ADEs carry wide CIs, so an ADE from this benchmark should not
be compared with one measured on a different split. Rankings should only be claimed where the
paired CI excludes zero (the "Separated" column). Model pairs that are not separated are ties
on this test set.

### Per-class fairness audit (test split)

Per-IDD-3D-class error for the deployable model with the lowest mean test ADE (it is not
separated from the runner-up; see the bootstrap section). `ADE per path-meter` normalizes by
how far the agent actually travels, so fast classes aren't trivially "worse".

<!-- BEGIN AUTOGENERATED: per-class audit (scripts/make_tables.py) -->
Lowest mean test ADE among deployable models: **ST-GAT-inspired (MHA)** (per-class stats from one representative seed; see the bootstrap section for whether it is separated from the runner-up).

| Class | n | ADE (m) | FDE (m) | Mean GT path (m) | ADE per path-meter |
| --- | --- | --- | --- | --- | --- |
| MotorcyleRider | 396 | 4.12 | 9.12 | 25.6 | 0.161 |
| Pedestrian | 372 | 1.10 | 2.17 | 4.2 | 0.263 |
| Car | 336 | 5.25 | 11.31 | 25.1 | 0.209 |
| ScooterRider | 328 | 5.38 | 12.00 | 28.8 | 0.187 |
| Truck | 278 | 2.56 | 5.55 | 11.8 | 0.217 |
| TourCar | 247 | 4.62 | 9.42 | 22.8 | 0.203 |
| Scooter | 50 | 0.53 | 1.19 | 1.6 | 0.328 |

ADE disparity (worst/best class, n ≥ 50): **10.25×**
<!-- END AUTOGENERATED: per-class audit -->

### Where the oracle-vs-deployable gap comes from (Trajectron++)

On the canonical windows, Trajectron++'s most-likely output (3.91 m) is 1.83 m behind its
oracle best-of-20 (2.08 m). That gap is *not* a selection failure. A best-of-K sweep plus a
mean-over-samples readout (same 20 draws, no extra compute;
[`scripts/eval_trajectronpp_canonical.py`](scripts/eval_trajectronpp_canonical.py)
`sampling_curve`) splits it into a calibration slice and a coverage slice:

![Best-of-K decomposition of the Trajectron++ oracle-vs-deployable gap](outputs/benchmark_v2/gap_decomposition.png)

*(Regenerate from the artifact: `python scripts/plot_gap_decomposition.py`.)*

<!-- BEGIN AUTOGENERATED: trajectronpp gap decomposition (scripts/make_tables.py) -->
| Readout | Test ADE (m) | Meaning |
| --- | --- | --- |
| Random single sample (best-of-1 ≈ mean-sample) | 4.55 | one draw, blind |
| **Most-likely (deployable)** | **3.91** | argmax latent + GMM mode |
| Oracle best-of-20 | 2.08 | min over samples, picked in hindsight |

Best-of-K curve (K = 1, 2, 5, 10, 20): **4.51 → 3.54 → 2.79 → 2.38 → 2.08**.

- **Calibration value = 0.64 m** (random draw → most-likely): the deployable mode beats a random sample — the mode selection is genuinely informative.
- **Coverage value = 1.83 m** (most-likely → oracle): the irreducible cost of committing to one future. The curve is still falling at the largest K, the signature of genuine multimodality rather than sampling noise — only multiple deployed modes can close it.

Source artifact: [`outputs/benchmark_v2/summary.json`](outputs/benchmark_v2/summary.json) (`trajectronpp_gap_decomposition`, 3-seed test mean on the canonical windows).
<!-- END AUTOGENERATED: trajectronpp gap decomposition -->

### Calibrated uncertainty bands (split-conformal wrapper)

A post-hoc wrapper ([`scripts/conformal_wrapper.py`](scripts/conformal_wrapper.py)) turns each
deployable point predictor into an uncertainty band around its forecast, with no retraining.
The nonconformity score is the per-step position error. CV and MLP are reproduced in numpy;
the neural models are fed from prediction dumps
([`scripts/dump_predictions.py`](scripts/dump_predictions.py),
[`scripts/eval_trajectronpp_canonical.py`](scripts/eval_trajectronpp_canonical.py) for
Trajectron++, and the AgentFormer/LED drivers' `--save-pred-npz`). Every dump is the seed-0
model on the canonical windows. Two kinds of band:

- **per-step**: a radius r_t per future step. Guarantees each *step* is covered with
  probability ≥ 1 − α, but not that the whole future stays inside.
- **simultaneous**: r_t = q · σ_t, where σ_t is a per-step scale fitted on half of the
  calibration scenes and q is the conformal quantile of each agent's max_t error_t / σ_t on
  the other half. The guarantee covers the agent's **whole valid future**, which is what a
  planner needs.

Each comes either with one global radius or with per-IDD-3D-class radii (Mondrian conformal
prediction). Two evaluation regimes:

- **B (same distribution):** calibrate on one half of the **test scenes**, evaluate on the
  other half, repeated over 100 random scene splits.
- **A (deployment):** calibrate on val, evaluate on test, with a scene-bootstrap CI.

<!-- BEGIN AUTOGENERATED: conformal coverage (scripts/make_tables.py) -->
| Predictor | B: step coverage, mean (5–95%) over 100 scene folds | A: val→test step coverage [95% CI] | B folds at or below A | Trajectory coverage: per-step band | Trajectory coverage: simultaneous band | Mean radius (m): per-step / simultaneous |
| --- | --- | --- | --- | --- | --- | --- |
| Trajectron++ (most-likely) | 87.2% (70.0%–97.8%) | 77.9% [64.2%, 89.1%] | 17% | 74.6% | 86.4% | 13.8 / 27.8 |
| LED (mean-denoised) | 87.1% (71.2%–97.7%) | 73.3% [57.8%, 86.2%] | 9% | 73.5% | 86.6% | 12.5 / 26.4 |
| AgentFormer (2 Hz, most-likely) | 87.5% (69.1%–97.6%) | 75.7% [61.6%, 87.3%] | 12% | 73.4% | 87.0% | 12.1 / 23.7 |
| ST-GAT (MHA) | 87.7% (73.2%–98.0%) | 79.3% [66.8%, 89.1%] | 18% | 73.3% | 86.5% | 11.0 / 21.9 |
| Map-free transformer | 87.6% (71.4%–97.6%) | 80.8% [67.8%, 91.2%] | 22% | 73.6% | 86.7% | 11.3 / 22.0 |
| MLP | 87.1% (71.3%–97.5%) | 74.1% [59.3%, 86.5%] | 8% | 72.1% | 86.2% | 13.5 / 28.9 |
| K-mode (top-1) | 88.0% (74.0%–97.8%) | 75.7% [63.3%, 85.3%] | 10% | 74.0% | 86.7% | 13.5 / 26.6 |
| CV (vel_k=5) | 87.0% (70.5%–98.3%) | 73.8% [57.7%, 87.1%] | 11% | 74.0% | 86.1% | 16.3 / 32.3 |

Per-class step coverage (regime B, mean over scene folds; target 90%): one global radius vs per-class (Mondrian) radii:

| Predictor | Car: global → per-class | Pedestrian: global → per-class | MotorcyleRider: global → per-class |
| --- | --- | --- | --- |
| Trajectron++ (most-likely) | 80.2% → 87.3% | 95.9% → 88.6% | 87.3% → 88.2% |
| LED (mean-denoised) | 79.6% → 88.3% | 94.2% → 87.7% | 88.3% → 88.6% |
| AgentFormer (2 Hz, most-likely) | 80.3% → 88.1% | 95.6% → 87.1% | 88.5% → 89.1% |
| ST-GAT (MHA) | 79.1% → 87.9% | 95.2% → 88.0% | 88.2% → 89.3% |
| Map-free transformer | 80.3% → 88.3% | 96.3% → 88.1% | 87.3% → 89.0% |
| MLP | 79.5% → 87.8% | 94.5% → 88.3% | 87.5% → 88.3% |
| K-mode (top-1) | 80.9% → 87.8% | 96.3% → 88.3% | 85.5% → 89.1% |
| CV (vel_k=5) | 78.4% → 86.9% | 95.0% → 88.0% | 87.8% → 88.5% |

Source artifact: [`outputs/benchmark_v2/conformal_metrics.json`](outputs/benchmark_v2/conformal_metrics.json) (α = 0.1, 2,000 scene resamples for the regime-A CIs).
<!-- END AUTOGENERATED: conformal coverage -->

What this shows (α = 0.1):

1. **On 22 test scenes, scene clustering is what limits conformal coverage.** Split by scene,
   regime-B coverage lands a little under target on average and varies widely from one scene
   split to the next (5–95% range in column 2). The guarantee is exact for exchangeable units,
   but agents within a scene are correlated, and only ~11 scenes calibrate each split.
   *Correction:* an earlier version of this analysis split test **agents** 50/50 and reported
   near-nominal coverage for every model. That put agents of the same scene, and Trajectron++'s
   overlapping per-timestep windows of the same agent, on both sides of the split, so it was
   optimistic. It has been replaced.
2. **The val→test drop is not distinguishable from scene sampling.** Regime-A coverage is
   below the regime-B mean for every model, but it sits inside the spread that random
   no-shift scene splits produce (column 4: share of those splits at or below A). *Correction:*
   the earlier claims that "naive conformal fails under the val→test shift" and that "stronger
   predictors survive the shift better" are withdrawn as not supported at this sample size.
3. **Per-step bands do not cover whole futures; simultaneous bands do, at a cost.** The
   share of agents whose entire future stays inside the per-step band (column 5) is far below
   the 90% target. The simultaneous band (column 6) recovers most of it, and its radius
   (column 7) is roughly twice as wide. That width is the main obstacle to using these bands
   for planning, and tighter trajectory-level bands are the next research step.
4. **Per-class radii fix per-class unfairness, robustly across scene splits** (second
   table). One global radius under-covers Cars and over-covers Pedestrians. Per-class radii
   bring both close to the overall level, widening the Car band and roughly halving the
   Pedestrian band. Per-class calibration is standard Mondrian conformal prediction; the
   contribution is showing that heterogeneous mixed traffic needs it.
5. **A more accurate forecaster buys a tighter band.** At a fixed coverage target the band
   radius tracks deployable ADE (figure below).

![Uncertainty efficiency: lower ADE buys a tighter conformal band at fixed 90% coverage](outputs/benchmark_v2/uncertainty_efficiency.png)

![The val→test coverage drop is within scene-sampling variability](outputs/benchmark_v2/robustness_gap.png)

Per-class coverage, global vs per-class radii, for one scene split: one figure per model in
`outputs/benchmark_v2/conformal_coverage_*.png`.

*(Regenerate: `python scripts/conformal_wrapper.py && python scripts/plot_conformal_coverage.py &&
python scripts/plot_uncertainty_efficiency.py`. The bands are wide because they wrap 6 s point
predictors; their absolute width reflects the base model's accuracy.)*

**Robustness to the coverage target (α).** α = 0.1 (90%) is a reporting convention, not a
safety-derived spec. [`scripts/conformal_alpha_sensitivity.py`](scripts/conformal_alpha_sensitivity.py)
re-runs the evaluation at 80% / 90% / 95% (B = mean over scene splits, A = val→test):

![Conformal coverage across targets α](outputs/benchmark_v2/conformal_alpha_sensitivity.png)

<!-- BEGIN AUTOGENERATED: conformal alpha sensitivity (scripts/make_tables.py) -->
| Model | B @ 80% | A @ 80% | B @ 90% | A @ 90% | B @ 95% | A @ 95% |
| --- | --- | --- | --- | --- | --- | --- |
| Trajectron++ (most-likely) | 78.2% | 66.2% | 87.2% | 77.9% | 92.8% | 86.7% |
| LED (mean-denoised) | 78.2% | 59.1% | 87.1% | 73.3% | 92.9% | 82.7% |
| AgentFormer (2 Hz, most-likely) | 78.7% | 65.0% | 87.5% | 75.7% | 92.8% | 84.4% |
| ST-GAT (MHA) | 78.7% | 66.1% | 87.7% | 79.3% | 93.1% | 88.8% |
| Map-free transformer | 78.5% | 65.9% | 87.6% | 80.8% | 93.1% | 88.8% |
| MLP | 78.0% | 60.1% | 87.1% | 74.1% | 92.9% | 83.9% |
| K-mode (top-1) | 78.7% | 62.0% | 88.0% | 75.7% | 93.5% | 85.9% |
| CV (vel_k=5) | 77.6% | 64.1% | 87.0% | 73.8% | 92.8% | 82.7% |

Mean per-step band radius (regime B, scene-fold mean) at each target:

| Model | 80% | 90% | 95% |
| --- | --- | --- | --- |
| Trajectron++ (most-likely) | 8.2 m | 13.8 m | 20.0 m |
| LED (mean-denoised) | 8.0 m | 12.5 m | 18.1 m |
| AgentFormer (2 Hz, most-likely) | 7.8 m | 12.1 m | 17.2 m |
| ST-GAT (MHA) | 6.8 m | 11.0 m | 16.8 m |
| Map-free transformer | 7.0 m | 11.3 m | 16.2 m |
| MLP | 8.5 m | 13.5 m | 19.2 m |
| K-mode (top-1) | 8.2 m | 13.5 m | 21.7 m |
| CV (vel_k=5) | 9.9 m | 16.3 m | 23.9 m |

Source artifact: [`outputs/benchmark_v2/conformal_sensitivity.json`](outputs/benchmark_v2/conformal_sensitivity.json) (100 scene folds per target).
<!-- END AUTOGENERATED: conformal alpha sensitivity -->

The qualitative picture is the same at every target: regime B lands near the target but slightly
under it, regime A is lower, and the band widens as the target tightens.

### Which model is safest? (pre-registered safety scorecard)

Mean ADE ranks forecasters, not safety. A planner needs a predictor that foresees the agents that
actually come close to the ego vehicle, without flagging everyone, and does so for pedestrians
and two-wheelers too, fast enough for 10 Hz. The rule below was committed **before** any
scorecard number was computed; full definitions are in
[`docs/wiki/safety-scorecard.md`](docs/wiki/safety-scorecard.md).

- **Candidates (12, seeds 0–2, canonical windows).** There are 8 single-trajectory predictors and
  4 multi-future sets (LED, AgentFormer and Trajectron++ with 10 samples each, and the K-mode
  decoder's 10 modes). Each gets a per-step, per-class 90% conformal band calibrated on val.
- **Conflict.** An agent's true position comes within 5 m of the ego's actual position (from
  `outputs/ego_trajectories.csv`) at the same future step.
- **Primary metric.** The test ROC-AUC for flagging conflict agents, sweeping a scale on the
  band. It is reported as the 3-seed mean with a seed-pooled scene-bootstrap CI.
- **Pick.** The highest primary AUC wins. Packages tied with it (paired 95% CI includes 0) are
  then compared on the same AUC over vulnerable road users (pedestrians, children and
  two-wheeler riders). Any remaining tie goes to the lowest GPU latency.

<!-- BEGIN AUTOGENERATED: safety scorecard (scripts/make_tables.py) -->
Rule committed in `a33f7f0` before any of these numbers existed. Test split: 2,177 agents, 276 conflicts (195 of them pedestrians or two-wheeler riders).

| Package | Output | Conflict AUC [95% CI] | Pedestrians & two-wheelers AUC | Recall / false alarms at the 90% band | Latency (ms, median) |
| --- | --- | --- | --- | --- | --- |
| ST-GAT (MHA) | 1 trajectory + band | 0.961 [0.932, 0.981] | 0.965 [0.937, 0.983] | 96% / 22% | 8.0 |
| Map-free transformer | 1 trajectory + band | 0.960 [0.928, 0.981] | 0.959 [0.920, 0.982] | 97% / 22% | 3.2 |
| K-mode (top-1) | 1 trajectory + band | 0.958 [0.927, 0.981] | 0.961 [0.927, 0.984] | 97% / 23% | 19.4 |
| LED (mean-denoised) | 1 trajectory + band | 0.958 [0.925, 0.981] | 0.958 [0.919, 0.981] | 96% / 20% | 27.4 |
| MLP | 1 trajectory + band | 0.957 [0.923, 0.982] | 0.959 [0.918, 0.983] | 96% / 24% | 0.6 |
| AgentFormer (most-likely) | 1 trajectory + band | 0.957 [0.924, 0.981] | 0.956 [0.917, 0.981] | 96% / 21% | 58.4 |
| Trajectron++ (most-likely) | 1 trajectory + band | 0.952 [0.918, 0.977] | 0.949 [0.906, 0.976] | 97% / 27% | 195.2 |
| CV (vel_k=5) | 1 trajectory + band | 0.951 [0.914, 0.979] | 0.954 [0.912, 0.980] | 95% / 22% | 0.0 |
| Trajectron++ set (10 samples) | 10 futures + band | 0.942 [0.907, 0.969] | 0.934 [0.887, 0.968] | 98% / 28% | 212.8 |
| AgentFormer set (10 samples) | 10 futures + band | 0.940 [0.907, 0.966] | 0.937 [0.890, 0.969] | 97% / 24% | 102.3 |
| LED set (10 samples) | 10 futures + band | 0.935 [0.893, 0.965] | 0.934 [0.881, 0.969] | 97% / 32% | 42.6 |
| K-mode set (10 modes) | 10 futures + band | 0.898 [0.850, 0.934] | 0.875 [0.808, 0.926] | 100% / 53% | 18.7 |

How the pre-registered rule decided:

1. Leader on P (conflict AUC): ST-GAT (MHA) (0.961).
2. Tied with the leader on P (paired 95% CI includes 0): CV (vel_k=5), MLP, Map-free transformer, K-mode (top-1), AgentFormer (most-likely), LED (mean-denoised), Trajectron++ (most-likely).
3. Tie-break 1 (vulnerable road users): best ST-GAT (MHA) (0.965); still tied: CV (vel_k=5), MLP, Map-free transformer, K-mode (top-1), AgentFormer (most-likely), LED (mean-denoised), Trajectron++ (most-likely).
4. Tie-break 2 (latency): CV (vel_k=5) at 0.0 ms.

**Rule outcome: CV (vel_k=5).**

Exploratory (added after the result, not used for the pick): current distance alone flags conflicts with AUC 0.754, and 58% of conflict agents are already within 10 m when the prediction is made. On the 116 emerging conflicts (more than 10 m away at prediction time), the best package is Map-free transformer at 0.956, against 0.937 for constant velocity.

Source artifacts: [`outputs/benchmark_v2/safety_scorecard.json`](outputs/benchmark_v2/safety_scorecard.json), [`outputs/benchmark_v2/safety_latency.json`](outputs/benchmark_v2/safety_latency.json) (2,000 seed-pooled scene resamples; latency on NVIDIA RTX 4500 Ada Generation).
<!-- END AUTOGENERATED: safety scorecard -->

What this means:

1. **The rule picks constant velocity because the test cannot separate the predictors, not
   because CV is safest.** All eight single-trajectory predictors flag near-ego conflicts with
   AUC 0.95–0.96, and their paired CIs overlap, both for all agents and for pedestrians and
   two-wheelers. The rule therefore falls through to latency, where CV is trivially fastest.
2. **Prediction matters, but motion extrapolation provides almost all of the benefit.**
   Current distance alone reaches AUC 0.75. Any of the predictors lifts it to about 0.95, and
   CV already does most of that. On the conflicts that need anticipation (agents more than
   10 m away when the prediction is made), the learned models are about 0.01–0.02 AUC ahead of
   CV, which is not separated with 22 test scenes.
3. **Multi-future sets do worse here.** A band around each of 10 samples covers more road,
   so they raise more false alarms at the same coverage.
4. **Accuracy gains do not yet show up as safety gains.** The learned models are clearly more
   accurate than CV (ST-GAT vs CV: −1.03 m ADE, separated), including near the ego. On this
   open-loop proxy, that accuracy does not translate into measurably better conflict
   detection. Showing a safety benefit needs closed-loop evaluation, with a planner in CARLA
   and collision rates.

This is an open-loop proxy: the ego path is what the human driver did.

### Closed-loop test (pre-registered)

Each predictor drives the same planner through the recorded test scenes, in
[`scripts/closed_loop_sim.py`](scripts/closed_loop_sim.py). The other road users replay their
logs and do not react.

- **Episodes.** There are 176 per package and seed: 22 scenes × 4 start times × 2 ego personas.
  The *nominal* persona drives at the human's speed; the *assertive* one drives at 6 m/s or
  more.
- **Planner.** Every 0.2 s it chooses among 35 speed-and-swerve manoeuvres that keep clear of
  each agent's predicted footprint, inflated by that package's val-calibrated 90% band.
- **Score.** An episode scores 0 after a collision the ego caused. Otherwise it combines
  progress (relative to the human driver), the absence of harsh braking and the absence of
  near misses, in nuPlan style.
- **Pick.** It follows the same rule as the scorecard: highest score, then pedestrian and
  two-wheeler collisions, then latency.

The protocol, including the planner fixes made on the **validation** scenes with reference
drivers only, is in [`docs/wiki/closed-loop-test.md`](docs/wiki/closed-loop-test.md). It was
committed before any learned predictor was run on the test scenes.

<!-- BEGIN AUTOGENERATED: closed-loop test (scripts/make_tables.py) -->
Rule committed in `b6f1a43` before any learned predictor ran on the test scenes. 22 test scenes, 176 episodes per package and seed; 3-seed means.

| Package | Episode score [95% CI] | At-fault collision episodes | …with pedestrians / two-wheelers | Progress | Harsh braking | Near misses | Score without the band |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LED (mean-denoised) | 0.745 [0.607, 0.866] | 10.6% | 8.5% | 0.86 | 37% | 23% | 0.831 |
| ST-GAT (MHA) | 0.745 [0.613, 0.857] | 12.1% | 8.5% | 0.87 | 40% | 23% | 0.825 |
| Map-free transformer | 0.739 [0.606, 0.853] | 10.2% | 9.1% | 0.86 | 40% | 25% | 0.819 |
| CV (vel_k=5) | 0.733 [0.606, 0.846] | 12.5% | 10.2% | 0.86 | 43% | 23% | 0.800 |
| AgentFormer (most-likely) | 0.724 [0.587, 0.841] | 14.0% | 11.7% | 0.87 | 42% | 25% | 0.809 |
| Trajectron++ (most-likely) | 0.715 [0.582, 0.836] | 14.4% | 8.3% | 0.86 | 45% | 23% | 0.799 |
| K-mode (top-1) | 0.706 [0.579, 0.818] | 13.3% | 9.7% | 0.84 | 49% | 22% | 0.803 |
| MLP | 0.705 [0.570, 0.828] | 13.3% | 12.5% | 0.85 | 45% | 25% | 0.810 |
| AgentFormer set (10 samples) | 0.704 [0.562, 0.830] | 13.4% | 11.4% | 0.82 | 46% | 24% | 0.750 |
| Trajectron++ set (10 samples) | 0.694 [0.552, 0.816] | 15.2% | 10.2% | 0.84 | 52% | 25% | 0.739 |
| LED set (10 samples) | 0.693 [0.560, 0.813] | 15.5% | 13.3% | 0.83 | 50% | 25% | 0.737 |
| K-mode set (10 modes) | 0.524 [0.403, 0.643] | 22.3% | 16.1% | 0.73 | 81% | 37% | 0.509 |
| *Human driver (log replay)* (reference) | 0.928 | 5.7% | 2.3% | 1.00 | 0% | 12% | — |
| *Oracle (true future, no band)* (reference) | 0.848 | 6.2% | 6.2% | 0.90 | 6% | 26% | — |
| *No prediction (ignore agents)* (reference) | 0.795 | 15.9% | 14.2% | 0.92 | 2% | 26% | — |

How the pre-registered rule decided:

1. Leader on the episode score: LED (mean-denoised) (0.745).
2. Tied with the leader (paired 95% CI includes 0): CV (vel_k=5), ST-GAT (MHA), Map-free transformer, K-mode (top-1), AgentFormer (most-likely), Trajectron++ (most-likely), AgentFormer set (10 samples), LED set (10 samples), Trajectron++ set (10 samples).
3. Tie-break 1 (at-fault collisions with pedestrians / two-wheelers, lower is better): best Trajectron++ (most-likely) (0.083); still tied: LED (mean-denoised), CV (vel_k=5), ST-GAT (MHA), Map-free transformer, K-mode (top-1), AgentFormer (most-likely), AgentFormer set (10 samples), LED set (10 samples), Trajectron++ set (10 samples).
4. Tie-break 2 (latency): CV (vel_k=5).

**Rule outcome: CV (vel_k=5).**

Source artifact: [`outputs/benchmark_v2/closed_loop.json`](outputs/benchmark_v2/closed_loop.json) (2,000 seed-pooled scene resamples).
<!-- END AUTOGENERATED: closed-loop test -->

What this means:

1. **The closed loop confirms the scorecard: no predictor is demonstrably safest.** Scores span
   0.69–0.75 for every single-trajectory model and three of the sets, and their paired CIs
   overlap. The rule therefore falls through to latency and picks constant velocity again. Only
   the K-mode set is clearly worse: its spread-out modes block the road, so the planner brakes
   hard in 81% of episodes.
2. **The measurement has a floor of about 6%.** The human replay registers at-fault contacts in
   5.7% of episodes. These come from two footprint artifacts (a diagonal car overlap and a 52°
   route-heading error in a turn), each counted in several overlapping episodes. The
   perfect-prediction oracle, at 6.2%, sits on that floor.

Exploratory, added after the result and not used for the pick
([`scripts/closed_loop_explore.py`](scripts/closed_loop_explore.py), the `exploratory` block of
the artifact). These are 116 paired comparisons without multiple-comparison correction, so
single "separated" results are hypotheses, not findings.

- **The 90% band hurts every single-trajectory model.** All eight score significantly higher
  without it (+0.07 to +0.11). As a hard constraint in dense traffic it leaves the planner with
  no clear manoeuvre, so it brakes hard (37–50% of episodes) and its fallback escapes cause
  more collisions. Used this way, most models do worse than ignoring other agents altogether.
  Bands this wide are not usable as planning constraints; tighter, trajectory-level bands are
  needed.
- **Without the band, prediction pays.** Every single-trajectory model, CV included, cuts
  at-fault collision episodes by 4–7 points against ignoring agents (15.9%).
- **Without the band, LED (+0.031) and ST-GAT (+0.025) edge out CV on the episode score.** The
  map-free transformer is borderline (+0.019, CI −0.001 to +0.038). This suggests learned
  predictors do help in closed loop, but it needs confirmation on data this analysis has not
  seen.

### Closed-loop test v2: cross-validated, with a fixed instrument and planner (pre-registered)

The first closed-loop test exposed three problems:
- a ~6% false-collision floor in the measurement;
- 90% bands that paralysed the planner;
- too few test scenes to separate the predictors.

The second test addresses all three. The full protocol is in
[`docs/wiki/closed-loop-cv.md`](docs/wiki/closed-loop-cv.md), committed before any fold model
drove a primary scene.

- **Measurement.** Three fixes, each calibrated on the recorded human driver alone:
  - Ego-pose glitches from the motion reconstruction are excluded.
  - The ego is oriented by its logged yaw.
  - Collisions are tested with exact rectangles and a footprint measured from the data.

  The human replay's false at-fault rate falls to 0% (test), 0% (val) and 1.1% (former train
  scenes).
- **Planner.** A 30-configuration grid on the 32 already-seen scenes chose a 1 s hard safety
  horizon, a soft penalty beyond it, and **no conformal band**. Every band setting scored lower
  there (0.813 with the full band against 0.894 without). Harsh braking dropped from about 45%
  of episodes to about 9%.
- **More scenes.** 5-fold scene cross-validation retrains CV, MLP, ST-GAT, the transformer and
  LED, so each of the 105 scenes is driven by models that never saw it. The **primary** analysis
  uses the 73 scenes that were never evaluated before. Hypotheses (Holm-corrected):
  - learned models beat CV on score (H1) and on at-fault collisions (H1b);
  - every predictor beats ignoring agents (H2);
  - the band hurts (H3).

<!-- BEGIN AUTOGENERATED: closed-loop cv (scripts/make_tables.py) -->
Rule committed in `24684e2` before any fold model drove a primary scene. Primary set: 73 scenes never evaluated before, 524 episodes per package; one model per fold (seed 0); planner v2 without the band.

| Package | Episode score [95% CI] | At-fault collision episodes | …with pedestrians / two-wheelers | Harsh braking | Score with the full 90% band |
| --- | --- | --- | --- | --- | --- |
| CV (vel_k=5) | 0.802 [0.755, 0.842] | 2.5% | 2.3% | 18% | 0.657 |
| Map-free transformer | 0.801 [0.753, 0.841] | 2.9% | 2.1% | 19% | 0.668 |
| MLP | 0.799 [0.752, 0.840] | 3.4% | 2.5% | 22% | 0.666 |
| ST-GAT (MHA) | 0.798 [0.753, 0.839] | 2.1% | 1.3% | 21% | 0.653 |
| LED (mean-denoised) | 0.793 [0.744, 0.835] | 3.4% | 2.9% | 22% | 0.672 |
| K-mode (top-1) (secondary) | 0.791 [0.743, 0.833] | 4.6% | 3.8% | 20% | 0.664 |
| K-mode set (10 modes) (secondary) | 0.626 [0.564, 0.689] | 13.4% | 9.4% | 63% | 0.537 |
| LED set (10 samples) (secondary) | 0.682 [0.629, 0.733] | 7.4% | 4.8% | 56% | 0.621 |
| *Human driver (log replay)* (reference) | 0.934 [0.902, 0.961] | 1.1% | 0.0% | 0% | — |
| *Oracle (true future, no band)* (reference) | 0.821 [0.785, 0.856] | 0.4% | 0.0% | 13% | — |
| *No prediction (ignore agents)* (reference) | 0.729 [0.661, 0.790] | 16.4% | 9.7% | 2% | — |

How the pre-registered rule decided:

1. Leader on the episode score: CV (vel_k=5) (0.802).
2. Tied with the leader (paired 95% CI includes 0): MLP, ST-GAT (MHA), Map-free transformer, LED (mean-denoised).
3. Tie-break 1 (at-fault collisions with pedestrians / two-wheelers, lower is better): best ST-GAT (MHA) (0.013); still tied: CV (vel_k=5), MLP, Map-free transformer, LED (mean-denoised).
4. Tie-break 2 (latency): CV (vel_k=5).

**Rule outcome: CV (vel_k=5).**

Differences are signed so that positive values favour the hypothesis.

| Hypothesis (one-sided, Holm) | Difference, 95% CI | p (Holm) | Supported | Supported in the 7 instrument variants |
| --- | --- | --- | --- | --- |
| H1 ST-GAT (MHA) higher score than CV | [-0.019, 0.011] | 1.000 | no | 0/7 |
| H1b ST-GAT (MHA) fewer at-fault collisions than CV | [-0.014, 0.025] | 1.000 | no | 0/7 |
| H1 Map-free transformer higher score than CV | [-0.016, 0.014] | 1.000 | no | 0/7 |
| H1b Map-free transformer fewer at-fault collisions than CV | [-0.017, 0.009] | 1.000 | no | 0/7 |
| H1 LED (mean-denoised) higher score than CV | [-0.021, 0.004] | 1.000 | no | 0/7 |
| H1b LED (mean-denoised) fewer at-fault collisions than CV | [-0.024, 0.004] | 1.000 | no | 0/7 |
| H2 CV (vel_k=5) fewer at-fault collisions than ignoring agents | [0.084, 0.199] | 0.000 | yes | 7/7 |
| H2 MLP fewer at-fault collisions than ignoring agents | [0.076, 0.191] | 0.000 | yes | 7/7 |
| H2 ST-GAT (MHA) fewer at-fault collisions than ignoring agents | [0.087, 0.205] | 0.000 | yes | 7/7 |
| H2 Map-free transformer fewer at-fault collisions than ignoring agents | [0.081, 0.197] | 0.000 | yes | 7/7 |
| H2 LED (mean-denoised) fewer at-fault collisions than ignoring agents | [0.075, 0.191] | 0.000 | yes | 7/7 |
| H3 CV (vel_k=5) chosen band setting (x0) scores higher than x1 | [0.111, 0.180] | 0.000 | yes | 7/7 |
| H3 MLP chosen band setting (x0) scores higher than x1 | [0.096, 0.167] | 0.000 | yes | 7/7 |
| H3 ST-GAT (MHA) chosen band setting (x0) scores higher than x1 | [0.109, 0.181] | 0.000 | yes | 7/7 |
| H3 Map-free transformer chosen band setting (x0) scores higher than x1 | [0.096, 0.170] | 0.000 | yes | 7/7 |
| H3 LED (mean-denoised) chosen band setting (x0) scores higher than x1 | [0.086, 0.158] | 0.000 | yes | 7/7 |
| S1 K-mode set (10 modes) higher score than K-mode (top-1) | [-0.217, -0.111] | 1.000 | no | — |
| S2 K-mode set (10 modes) fewer at-fault collisions than K-mode (top-1) | [-0.138, -0.044] | 1.000 | no | — |
| S1 LED set (10 samples) higher score than LED (mean-denoised) | [-0.149, -0.075] | 1.000 | no | — |
| S2 LED set (10 samples) fewer at-fault collisions than LED (mean-denoised) | [-0.069, -0.013] | 1.000 | no | — |

Instrument sensitivity: the rule picks CV (vel_k=5) in every one of the 7 variants.

Other scene sets (reported, not used for the decision): development (32 frozen val + test scenes): pick CV (vel_k=5); all 105 scenes: pick CV (vel_k=5).

Amendment 3 (secondary, same primary scenes and planner): three more multi-future models, each as its single trajectory and its 10-future set.

| Package | Episode score [95% CI] | At-fault collision episodes | …with pedestrians / two-wheelers | Harsh braking | Score with the full 90% band |
| --- | --- | --- | --- | --- | --- |
| MF transformer (top-1) | 0.790 [0.742, 0.833] | 3.6% | 2.5% | 22% | 0.650 |
| MF transformer set (10 modes) | 0.438 [0.379, 0.494] | 19.3% | 12.0% | 93% | 0.425 |
| AgentFormer (most-likely) | 0.800 [0.755, 0.840] | 2.9% | 2.5% | 19% | 0.664 |
| AgentFormer set (10 samples) | 0.793 [0.743, 0.838] | 3.4% | 2.9% | 25% | 0.673 |
| Trajectron++ (most-likely) | 0.799 [0.751, 0.841] | 3.4% | 3.1% | 20% | 0.650 |
| Trajectron++ set (10 samples) | 0.752 [0.697, 0.798] | 4.4% | 3.4% | 44% | 0.617 |

| Hypothesis (one-sided, Holm within amendment 3) | Difference, 95% CI | p (Holm) | Supported |
| --- | --- | --- | --- |
| S1 MF transformer set (10 modes) higher score than MF transformer (top-1) | [-0.415, -0.291] | 1.000 | no |
| S2 MF transformer set (10 modes) fewer at-fault collisions than MF transformer (top-1) | [-0.228, -0.084] | 1.000 | no |
| S1 AgentFormer set (10 samples) higher score than AgentFormer (most-likely) | [-0.023, 0.008] | 1.000 | no |
| S2 AgentFormer set (10 samples) fewer at-fault collisions than AgentFormer (most-likely) | [-0.017, 0.002] | 1.000 | no |
| S1 Trajectron++ set (10 samples) higher score than Trajectron++ (most-likely) | [-0.072, -0.022] | 1.000 | no |
| S2 Trajectron++ set (10 samples) fewer at-fault collisions than Trajectron++ (most-likely) | [-0.029, 0.010] | 1.000 | no |
| T1 MF transformer (top-1) higher score than Map-free transformer | [-0.026, 0.005] | 1.000 | no |
| T2 MF transformer (top-1) fewer at-fault collisions than Map-free transformer | [-0.021, 0.002] | 1.000 | no |

Source artifacts: [`outputs/benchmark_v2/closed_loop_cv.json`](outputs/benchmark_v2/closed_loop_cv.json), [`outputs/benchmark_v2/closed_loop_cv_sensitivity.json`](outputs/benchmark_v2/closed_loop_cv_sensitivity.json), [`outputs/benchmark_v2/closed_loop_cv_extra.json`](outputs/benchmark_v2/closed_loop_cv_extra.json).
<!-- END AUTOGENERATED: closed-loop cv -->

**What the confirmatory test shows:**

1. **No predictor is demonstrably safest in closed loop.** The five single-trajectory
   packages are within 0.01 of each other (0.793–0.802). The rule's tie-breaks pick constant
   velocity. ST-GAT has the fewest collisions with pedestrians and two-wheelers (1.3% vs
   2.1–2.9%), but that difference is not separated either. Learned models do not beat
   constant velocity on score or on collisions (H1 and H1b are unsupported in every variant).
2. **Prediction itself matters.** Every predictor cuts at-fault collision episodes from 16.4%
   (ignoring agents) to 2–3%. This holds in all 7 instrument variants.
3. **The conformal band hurts this planner** (≈ 0.66 with the full band vs ≈ 0.80 without),
   again in all 7 variants.
4. **Planning against every future of a multi-future set is unsafe.** The K-mode decoder's 10
   modes and LED's 10 samples score far below their own single trajectory: 0.626 vs 0.791 and
   0.682 vs 0.793. They also cause *more* at-fault collisions: 13.4% vs 4.6%, and 7.4% vs 3.4%.
   Both 95% CIs exclude zero in the harmful direction. When ten futures block the road, the
   planner's fallback manoeuvres cause the collisions. Amendment 3 extends this to all five
   multi-future models. The multi-future transformer's set is the worst package in the test
   (0.438, 19.3% at-fault), and the Trajectron++ set also scores significantly below its own
   single trajectory (0.752 vs 0.799). The AgentFormer set (0.793) is the only set level with
   its single trajectory (0.800), and it brakes harshly least often of the sets (25% vs 44–93%).
   No set is better than its single trajectory on any test (S1/S2 unsupported for all five).
   Whether multi-future prediction pays off therefore depends on how the planner uses it.
   That is the subject of separate, ongoing work.
5. **The planner, not prediction accuracy, is now the limit.** The oracle, which drives
   against the true future, scores only 0.821; the recorded human scores 0.934. On these
   scenes the remaining gap is mostly progress and harsh braking, which no predictor can close
   on its own.
6. **Training with ten futures does not improve the deployable trajectory.** The
   multi-future transformer's top-1 (0.790, 3.6% at-fault) is no better than the
   single-output transformer (0.801, 2.9%); T1 and T2 are unsupported. AgentFormer and
   Trajectron++ most-likely trajectories (0.800 and 0.799) join the five-way tie of point
   predictors. They are not in the decision rule, but they would not change it.

Caveats:
- the agents don't react;
- footprints are nominal;
- there is one seed per fold;
- the planner was tuned on the development scenes (excluded from the primary set).

### Honest takeaways (Benchmark v2)

1. **Two models share the top: the ST-GAT-inspired model and the map-free transformer.**
   ST-GAT has the lowest mean test ADE (3.41 ± 0.07; seeds 3.33 / 3.39 / 3.50) and the
   transformer is next (3.57 ± 0.11; seeds 3.43 / 3.71 / 3.58). They are tied on this test
   set: the seed-pooled bootstrap cannot separate their 3-seed means (ΔADE −0.17 m, 95% CI
   [−0.48, 0.13]). Both have a lower test ADE than the three published models re-run here
   on every seed: LED 3.75 ± 0.01, AgentFormer 3.75 ± 0.00, Trajectron++ 3.91 ± 0.08.
2. **That lead over the published models is consistent but not statistically separated.**
   Against LED, the closest of them, both top models are better in all 9 seed pairings. But
   once the comparison uses 3-seed means and the CI covers both test scenes and training
   seeds, it does not exclude zero at 95%: the transformer is −0.17 m [−0.55, 0.11] and
   ST-GAT −0.34 m [−0.76, 0.01]. The seed-0-only comparison (−0.31 m, separated) used the
   transformer's best seed, so it overstated the gap. With 22 test scenes, "consistently
   lower ADE" is the defensible claim, not "significantly better". The published models also
   run under this benchmark's constraints: no map, no yaw, and AgentFormer at 2 Hz.
3. **Validation and test disagree on the order.** LED has the best val ADE on every seed
   (2.03 ± 0.02, against 2.38 for ST-GAT) and the best oracle among the K = 20 samplers
   (1.98), so choosing one model on val would have picked LED. On test it ties AgentFormer
   behind the top two, and its val→test conformal coverage drop is the largest
   (87.1% → 73.3%). With 10 val scenes, val is a weak selector, so rankings here are claimed
   only where the paired test-scene CI excludes zero.
4. **Interaction modeling earns its keep on the full population.** In v1 (survivor-filtered
   data) the ST-GAT-inspired model lost to smoothed constant velocity. On the canonical
   dataset, which restores the 58% of agents the old filter discarded, it beats MLP
   (ΔADE −0.63 m, CI [−1.07, −0.28]) and CV (−1.03 m, CI [−1.96, −0.32]).
5. **No multimodal model turns its samples into a better single forecast.** The K-mode
   decoder's oracle best-of-10 (test ADE 1.88) is the best oracle in the table, ahead of
   LED (1.98), Trajectron++ (2.08) and AgentFormer (2.24) at K = 20. But every multimodal
   model loses 1.5–2.4 m when it has to commit to one output (K-mode 1.88 → 4.31, LED
   1.98 → 3.75, AgentFormer 2.24 → 3.75, Trajectron++ 2.08 → 3.91). None of those single
   outputs beats the two single-output models above, so mode selection, not mode diversity,
   remains the bottleneck. The K-mode decoder's learned top-1 also has the largest seed
   variance in the table (val ADE ± 0.36).
6. **The frozen test split is genuinely harder than val** (every deployable model's test ADE
   is 1.3–1.9× its val ADE). v1's val-only numbers were systematically optimistic. This gap
   is exactly what the frozen test set was created to expose.
7. **Raw errors concentrate on fast two-wheeler riders and cars, but normalization flips the
   story.** For the lowest-ADE model (table above), per-class ADE is highest for
   ScooterRider / Car / TourCar / MotorcyleRider (4.1–5.4 m) and lowest for scooters and
   pedestrians (0.5–1.1 m). Yet per meter actually traveled, pedestrians err *more*
   (0.263 ADE/path-m) than motorcycle riders (0.161). The raw gap mostly tracks path length.
   Relative to how far they move, slow, erratic agents remain the hard case.
8. **The K-mode decoder's "mode-selection problem" was partly a checkpoint-selection
   artifact.** The original pipeline selected the saved checkpoint by val *oracle* ADE, then
   reported that same checkpoint's top-1 — but the best epoch for coverage isn't the best
   epoch for the classifier. Selecting on val top-1 ADE instead (same architecture, same
   loss) improves deployable test ADE from 4.31 ± 0.05 to 4.15 ± 0.10 and sharply stabilizes
   it across seeds (val ADE std 0.36 → 0.09), at the honest cost of worse oracle coverage
   (test 1.88 → 2.01, val 1.32 → 1.69 — a real trade-off, not free).
   **EWTA annealing did not help further (second honest negative result).** A val-only grid
   (seed 0, 150 epochs) compared plain WTA-with-top-1-selection against evolving-WTA variants
   (winner set annealed 10→1 over training, with hard or soft mode-classification targets):
   plain WTA won outright (val top-1 ADE 2.46 vs 2.60 hard-EWTA vs 2.76 soft-EWTA), so no
   EWTA variant was carried to the final 3-seed run. Even after both fixes, the K-mode
   decoder's top-1 remains well behind ST-GAT (3.41) — mode selection is improved, not solved.
   **A content-aware scorer also did not help (third negative result), and the reason is
   the interesting part.** We rebuilt the mode-scoring head to rank each mode from its
   *decoded trajectory* (endpoint, waypoints, path length, heading) rather than from context
   only — a unit test confirms it genuinely reads the trajectory (perturbing one mode's path
   moves only that mode's logit). On the val gate it was a wash (val top-1 ADE 2.51 vs the
   2.46 context-only baseline, within seed noise), so it was not promoted to a 3-seed run.
   The mechanism explains why: this decoder is *deterministic*, so each mode's trajectory is
   a fixed function of the per-mode context state the scorer already used — the trajectory
   features are redundant, not new information. The classifier was never information-starved;
   the bottleneck is the winner-takes-all *assignment target* itself. That points the next
   attack at the objective/structure (jointly-trained mixture density, or anchor/goal modes),
   not the scorer's input. Grid + per-seed artifacts:
   [`outputs/benchmark_v2/mfp_ewta_grid/`](outputs/benchmark_v2/mfp_ewta_grid/),
   [`outputs/benchmark_v2/mfp_content_grid/`](outputs/benchmark_v2/mfp_content_grid/),
   `mfp_top1sel_seed*.json`.
9. **Unicycle dynamics with approximate heading do not pay off (honest negative result).**
   This variant models wheeled classes as VEHICLE nodes with Unicycle dynamics
   (nuScenes-style; heading derived from velocity direction, since IDD-3D yaw isn't in this
   repo's parsed data) plus rotation augmentation. It was selected over no augmentation on a
   val grid (2.12 vs 2.26, dense protocol). On the canonical test windows it *degrades*
   deployable most-likely (4.55 ± 0.17 vs 3.91 ± 0.08). The loss is concentrated on moving
   vehicles (riders +1.0 to +1.3 m, cars +1.0 m), while pedestrians are unchanged (−0.15 m).
   Oracle best-of-20 is slightly worse too (2.22 ± 0.06 vs 2.08 ± 0.02). Reading: the unicycle integrates
   heading-estimate noise into its deterministic rollout, which sampling absorbs but
   most-likely decoding cannot. True yaw (raw dataset / perception) is the prerequisite for
   dynamics-aware modeling to help here. Grid + per-seed artifacts:
   [`outputs/benchmark_v2/tpp_hetero_grid/`](outputs/benchmark_v2/tpp_hetero_grid/),
   `trajectronpp_hetero_canonical_seed*.json`.

## Results — world frame v1 (superseded by Benchmark v2)

⚠️ Superseded: no test set (selection and reporting shared the val split), two different eval
populations (CV/MLP on single-agent windows vs. interaction models on multi-agent sequences),
survivor-biased agent filter, single seeds. Kept for the record; numbers below are val-only.

<!-- BEGIN AUTOGENERATED: world-frame results (scripts/make_tables.py) -->
| Model | Eval set | K | Deployable single output? | Val ADE (m) | Val FDE (m) | Artifact |
| --- | --- | --- | --- | --- | --- | --- |
| CV (vel_k=1) | windows val | 1 | yes | 6.27 | 13.81 | [`outputs/baselines_world/cv_velk1_metrics_world_obs20_pred60.json`](outputs/baselines_world/cv_velk1_metrics_world_obs20_pred60.json) |
| CV (vel_k=5) | windows val | 1 | yes | 4.82 | 11.20 | [`outputs/baselines_world/cv_velk5_metrics_world_obs20_pred60.json`](outputs/baselines_world/cv_velk5_metrics_world_obs20_pred60.json) |
| MLP | windows val | 1 | yes | 4.56 | 10.25 | [`outputs/baselines_world/mlp_metrics_world_obs20_pred60.json`](outputs/baselines_world/mlp_metrics_world_obs20_pred60.json) |
| ST-GAT-inspired (MHA) | multi-agent val | 1 | yes | 6.41 | 13.40 | [`outputs/baselines_world/stgat_mha_metrics_world_obs20_pred60.json`](outputs/baselines_world/stgat_mha_metrics_world_obs20_pred60.json) |
| K-mode variety decoder — oracle best-of-K | multi-agent val | 10 | no (oracle) | 3.66 | 6.12 | [`outputs/baselines_world/mfp_k10_metrics_world_obs20_pred60.json`](outputs/baselines_world/mfp_k10_metrics_world_obs20_pred60.json) |
| K-mode variety decoder — learned top-1 | multi-agent val | 10 | yes | 7.70 | 15.66 | [`outputs/baselines_world/mfp_k10_metrics_world_obs20_pred60.json`](outputs/baselines_world/mfp_k10_metrics_world_obs20_pred60.json) |
| Trajectron++ — most-likely (K=1) | multi-agent val | 1 | yes | 3.99 | 8.72 | [`outputs/baselines_world/trajectronpp_metrics_world_obs20_pred60.json`](outputs/baselines_world/trajectronpp_metrics_world_obs20_pred60.json) |
| Trajectron++ — best-of-20 | multi-agent val | 20 | no (oracle) | 1.79 | 2.97 | [`outputs/baselines_world/trajectronpp_metrics_world_obs20_pred60.json`](outputs/baselines_world/trajectronpp_metrics_world_obs20_pred60.json) |
<!-- END AUTOGENERATED: world-frame results -->

Honest takeaways from v1 (kept for the record; superseded by the Benchmark v2 section above):

1. **Oracle multi-future coverage helps a lot** — Trajectron++ best-of-20 (ADE 1.79) and the
   K-mode decoder's best-of-10 (ADE 3.66) both beat every single-output model. The
   multi-future premise of the project survives the frame correction.
2. **Trajectron++ most-likely had the lowest val ADE** (3.99 / FDE 8.72, against the plain
   MLP's 4.56 / 10.25), but on a different population: node-timesteps with full 1.9 s history
   and 6 s future, n = 3,491. Benchmark v2 shows the population matters. On matched windows,
   Trajectron++ is not the best deployable model (see the October 2026 correction).
3. **The homegrown interaction model lost its edge in the world frame.** The ST-GAT-inspired
   model no longer beats smoothed constant velocity; its ego-frame advantage was substantially
   an artifact of learning ego motion.
4. **Learned top-1 mode selection needs rework** — it degrades badly in the world frame
   (7.70 ADE vs. 3.66 oracle). The v1 reading that Trajectron++'s most-likely decoding had
   solved this did not survive matched evaluation (Benchmark v2, takeaway 5).

## Results — legacy ego frame (deprecated, for the record)

⚠️ Ego-motion-contaminated data; kept only to document the project's history. Do not cite.
Additional protocol flaws: CV rows are evaluated on *all* windows (train+val); MLP and
ST-GAT v1/v2 use split seed 42 while v4/v5 and the K-mode decoder use seed 0.

<!-- BEGIN AUTOGENERATED: legacy ego-frame results (scripts/make_tables.py) -->
| Model | Eval set | ADE (m, ego-frame) | FDE (m, ego-frame) | Artifact |
| --- | --- | --- | --- | --- |
| CV (vel_k=1) | ALL 4,821 windows (no split) | 3.42 | 7.64 | [`outputs/baselines/1_cv_metrics_obs20_pred60.json`](outputs/baselines/1_cv_metrics_obs20_pred60.json) |
| CV (vel_k=5) | ALL 4,821 windows (no split) | 2.67 | 6.30 | [`outputs/baselines/1.1_cv_metrics_obs20_pred60_velk5.json`](outputs/baselines/1.1_cv_metrics_obs20_pred60_velk5.json) |
| MLP (seed 42) | windows val | 2.70 | 6.37 | [`outputs/baselines/2_mlp_metrics_obs20_pred60.json`](outputs/baselines/2_mlp_metrics_obs20_pred60.json) |
| ST-GAT-inspired v1 (CPU, seed 42) | multi-agent val | 2.37 | 5.80 | [`outputs/baselines/3_stgat_metrics_obs20_pred60.json`](outputs/baselines/3_stgat_metrics_obs20_pred60.json) |
| ST-GAT-inspired v2 (CUDA, seed 42) | multi-agent val | 2.31 | 5.54 | [`outputs/baselines/3.1_stgat_metrics_obs20_pred60_cuda.json`](outputs/baselines/3.1_stgat_metrics_obs20_pred60_cuda.json) |
| ST-GAT-inspired v4 (simple+AMP, seed 0) | multi-agent val | 1.89 | 4.53 | [`outputs/baselines/3.2_stgat_metrics_obs20_pred60_cuda_simple_amp_v4.json`](outputs/baselines/3.2_stgat_metrics_obs20_pred60_cuda_simple_amp_v4.json) |
| ST-GAT-inspired v5 (MHA+AMP, seed 0) | multi-agent val | 1.87 | 4.48 | [`outputs/baselines/3.3_stgat_metrics_obs20_pred60_cuda_mha_amp_noes_v5.json`](outputs/baselines/3.3_stgat_metrics_obs20_pred60_cuda_mha_amp_noes_v5.json) |
| K-mode decoder K=10 — oracle (no mode head) | multi-agent val | 0.97 | 1.89 | [`outputs/baselines/4.1_mfp_metrics_obs20_pred60_cuda_k10_noes_top1.json`](outputs/baselines/4.1_mfp_metrics_obs20_pred60_cuda_k10_noes_top1.json) |
| K-mode decoder K=10 — fixed mode-0 top-1 | multi-agent val | 7.33 | 19.83 | [`outputs/baselines/4.1_mfp_metrics_obs20_pred60_cuda_k10_noes_top1.json`](outputs/baselines/4.1_mfp_metrics_obs20_pred60_cuda_k10_noes_top1.json) |
| K-mode decoder K=10 — oracle (with mode head) | multi-agent val | 1.05 | 1.92 | [`outputs/baselines/4.2_mfp_metrics_obs20_pred60_cuda_k10_noes_argmax_log1_v2.json`](outputs/baselines/4.2_mfp_metrics_obs20_pred60_cuda_k10_noes_argmax_log1_v2.json) |
| K-mode decoder K=10 — learned top-1 | multi-agent val | 2.18 | 4.86 | [`outputs/baselines/4.2_mfp_metrics_obs20_pred60_cuda_k10_noes_argmax_log1_v2.json`](outputs/baselines/4.2_mfp_metrics_obs20_pred60_cuda_k10_noes_argmax_log1_v2.json) |
<!-- END AUTOGENERATED: legacy ego-frame results -->

## Reproducing

```bash
pip install -r requirements.txt
# GPU training additionally:
pip install torch --index-url https://download.pytorch.org/whl/cu124

# World-frame data (from committed parsed positions):
python scripts/0_compensate_ego_motion.py
python scripts/2_consolidate_trajectories.py --input-dir outputs/parsed_positions_world --output-csv outputs/trajectories_world/trajectories.csv
python scripts/3_make_windows_obs20_pred60.py --input-csv outputs/trajectories_world/trajectories.csv --output-npz outputs/windows_world_obs20_pred60.npz
python scripts/4_export_multiagent_trajpp_format.py --parsed-positions-dir outputs/parsed_positions_world --out-dir outputs/multiagent_world_v1
python scripts/5_export_stgat_format.py --scenes-dir outputs/multiagent_world_v1/scenes --out outputs/stgat_dataset_world_obs20_pred60.npz --write-stats outputs/stgat_dataset_world_obs20_pred60_stats.json

# Baselines (exact commands used for the artifacts in outputs/baselines_world/):
python scripts/model1_baseline_cv_eval.py --input-npz outputs/windows_world_obs20_pred60.npz --vel-k 5 --split-by-scene --seed 0 --val-frac 0.2 \
  --save-metrics-json outputs/baselines_world/cv_velk5_metrics_world_obs20_pred60.json --save-pred-npz outputs/baselines_world/cv_velk5_pred_world_obs20_pred60.npz
python scripts/model2_baseline_mlp_train_eval.py --input outputs/windows_world_obs20_pred60.npz --seed 0 \
  --save-best outputs/baselines_world/mlp_best_world_obs20_pred60.npz --save-metrics-json outputs/baselines_world/mlp_metrics_world_obs20_pred60.json \
  --save-pred-npz outputs/baselines_world/mlp_pred_val_world_obs20_pred60.npz
python scripts/model3_stgat_train_eval.py --data outputs/stgat_dataset_world_obs20_pred60.npz --device cuda --pool mha --amp --loss mse \
  --epochs 300 --patience 0 --lr 0.001 --weight-decay 1e-5 --batch-size 128 --hidden-dim 256 --emb-dim 64 --num-heads 4 --gat-layers 2 --dropout 0.1 \
  --lr-plateau --lr-plateau-factor 0.5 --lr-plateau-patience 10 --min-lr 1e-5 --tf-start 1.0 --tf-end 1.0 --tf-decay-epochs 1 --seed 0 \
  --save-metrics-json outputs/baselines_world/stgat_mha_metrics_world_obs20_pred60.json --save-ckpt outputs/baselines_world/stgat_mha_ckpt_world_obs20_pred60.pt \
  --save-pred-npz outputs/baselines_world/stgat_mha_pred_val_world_obs20_pred60.npz
python scripts/model4_mfp_train_eval.py --data outputs/stgat_dataset_world_obs20_pred60.npz --device cuda --k 10 --batch-size 64 \
  --epochs 300 --patience 0 --mode-cls-weight 0.1 --top1-by argmax --seed 0 \
  --save-metrics-json outputs/baselines_world/mfp_k10_metrics_world_obs20_pred60.json --save-ckpt outputs/baselines_world/mfp_k10_ckpt_world_obs20_pred60.pt \
  --save-pred-npz outputs/baselines_world/mfp_k10_pred_val_world_obs20_pred60.npz
```

### Trajectron++ (upstream, end to end)

```bash
git clone --depth 1 https://github.com/StanfordASL/Trajectron-plus-plus.git third_party/trajectron-plus-plus
python scripts/patch_trajectronpp.py          # Py3.12/NumPy2/Torch2 compatibility (idempotent)
pip install dill orjson pathos seaborn matplotlib scipy tensorboard tqdm
python scripts/6_export_trajectronpp_world.py --dt 0.1
python scripts/7_train_eval_trajectronpp.py --epochs 100
python scripts/eval_trajectronpp_canonical.py              # benchmark rows: seeds 0-2 on the canonical windows
python scripts/eval_trajectronpp_canonical.py --hetero     # same for the unicycle variant
```

Script 7 trains with the upstream `train.py` and writes
`outputs/baselines_world/trajectronpp_metrics_world_obs20_pred60.json` (both oracle
best-of-20 and the deployable most-likely output). It scores Trajectron++'s own dense protocol
(every valid node-timestep). The ranked benchmark rows come from the last two commands, which
re-score the same checkpoints on the canonical windows every other model uses. All agents are exported as point-mass
`PEDESTRIAN` nodes; class-conditioned dynamics is future work. `dt=0.1` is the default —
the historical `dt=0.4` bug cannot recur through this path.

### AgentFormer (upstream, map-free base CVAE)

```bash
git clone --depth 1 https://github.com/Khrylx/AgentFormer.git third_party/agentformer
python scripts/patch_agentformer.py           # Py3.12/NumPy2/Torch2 compatibility (idempotent)
pip install easydict glob2
cd scripts && python model6_agentformer_train_eval.py --data ../outputs/canonical_world_obs20_pred60.npz \
  --splits-file ../splits/scene_splits.json --epochs 100 --seed 0 --stride 5 \
  --save-metrics-json ../outputs/benchmark_v2/agentformer_seed0.json \
  --save-ckpt ../outputs/benchmark_v2/agentformer_ckpt_seed0.pt \
  --save-pred-npz ../outputs/benchmark_v2/preds_agentformer.npz      # seed 0 only
# plumbing check only (CPU, tiny model, smallest scenes, ~7 s):
python scripts/model6_agentformer_train_eval.py --smoke --device cpu \
  --save-metrics-json /tmp/af_smoke.json --save-ckpt /tmp/af_smoke.pt
```

**Rate: AgentFormer runs at 2 Hz (`--stride 5`), its native nuScenes regime.** It attends
over every agent × timestep token and decodes autoregressively, so at 10 Hz the largest
IDD-3D scene (61 agents) needs ~35 GB for a single training step. At stride 5 it sees 4
observed + 12 future keyframes (that scene: 1.8 GB). Its 12 predicted keyframes are linearly
interpolated back to 10 Hz from the last observed position
([`dd_agentformer.upsample_keyframes`](scripts/dd_agentformer.py)), so it is scored on the same
agents, frames and metric as every other model. The cost of the coarser input is borne by
AgentFormer alone.

The driver runs the upstream AgentFormer **base CVAE** in a map-free configuration
(`input_type=['scene_norm','vel']`, `use_map=False`, no heading; standard-normal prior),
fed canonical sequences directly by [`scripts/dd_agentformer.py`](scripts/dd_agentformer.py) so
the eval population matches the shared contract. It writes the multimodal JSON schema
(deployable `most_likely` = z=0 rollout, oracle `best_of_k` = min over K=20 prior samples).
[`scripts/run_benchmark_v2.py`](scripts/run_benchmark_v2.py) launches any missing seed
automatically when the repo is present. The DLow diversity head is out of scope for this
configuration.

### LED (upstream leapfrog diffusion)

```bash
git clone --depth 1 https://github.com/MediaBrain-SJTU/LED.git third_party/led
cd scripts && python model7_led_train_eval.py --data ../outputs/canonical_world_obs20_pred60.npz \
  --splits-file ../splits/scene_splits.json --seed 0 --core-epochs 100 --init-epochs 100 \
  --save-metrics-json ../outputs/benchmark_v2/led_seed0.json \
  --save-ckpt ../outputs/benchmark_v2/led_ckpt_seed0.pt \
  --save-pred-npz ../outputs/benchmark_v2/preds_led.npz              # seed 0 only
# plumbing check only (CPU, few sequences, ~10 s):
cd scripts && python model7_led_train_eval.py --data ../outputs/canonical_world_obs20_pred60.npz \
  --splits-file ../splits/scene_splits.json --smoke --device cpu \
  --save-metrics-json /tmp/led_smoke.json --save-ckpt /tmp/led_smoke.pt
```

LED (Mao et al., CVPR 2023) has two networks: a DDPM core denoiser, and a "leapfrog"
initializer that jumps the reverse process to its last few steps. Upstream ships only
the stage-2 trainer plus an NBA-pretrained core, and its code is hard-wired to NBA shapes (11
agents, 10/20 frames). [`scripts/dd_led.py`](scripts/dd_led.py) reuses the upstream modules
and makes only those shapes parametric. The driver trains **both stages** on IDD-3D at the full
10 Hz: the core with upstream's own (unused) noise-estimation loss, then the initializer with
the core frozen. Deployable output: the initializer's mean estimate, denoised with zero noise.
Oracle: best of the K = 20 leapfrog samples. One deliberate deviation is that upstream's
uncertainty loss broadcasts an (N,1) tensor against an (N,) tensor into an N×N outer product;
here each agent's uncertainty is paired with its own error, as the paper describes.

## Testing & CI

```bash
pytest tests -q
python scripts/make_tables.py --check
```

[`tests/`](tests/) contains: synthetic-scene recovery tests for the ego-motion compensation,
**pedestrian/car speed sanity checks on the committed world-frame data** (these would have
caught the ego-frame bug automatically), window-centering and dataset-consistency invariants,
metric-math correctness, the conformal engine (finite-sample radius, scene-level folds that never
split a scene, trajectory-level coverage of the simultaneous band), the scene-level bootstrap
(calibrated CIs on clustered data), the AgentFormer/LED adapters, the README↔artifact provenance
check, and a guard that the retracted claims never resurface in tracked documents. [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
runs all of it on every push and pull request.

## Model naming honesty

- *ST-GAT-inspired* — LSTM encoder + agent self-attention + LSTM decoder. **Not** the
  published ST-GAT (Huang et al., 2019).
- *K-mode variety decoder* (formerly "MFP") — best-of-K variety loss + learned mode-scoring
  head. **Not** Multiple Futures Prediction (Tang & Salakhutdinov, NeurIPS 2019).
- *Map-free transformer* — a plain Transformer (per-agent temporal encoder + agent
  self-attention + query decoder) on agent-relative displacements. **Not** a QCNet/map-based
  model: IDD-3D is lane-free, so there is no map branch and no map-anchored frame.
- *Trajectron++* — the actual upstream implementation (Salzmann et al., 2021), patched for
  modern Python/PyTorch by [`scripts/patch_trajectronpp.py`](scripts/patch_trajectronpp.py).
- *AgentFormer* — the actual upstream implementation (Yuan et al., ICCV 2021), patched for
  modern Python/PyTorch by [`scripts/patch_agentformer.py`](scripts/patch_agentformer.py) and
  run as the **base CVAE in a map-free configuration** (no map/lane channel, no heading input;
  standard-normal prior, so most-likely = the z=0 rollout). It is driven through its model
  interface on the shared canonical population by
  [`scripts/dd_agentformer.py`](scripts/dd_agentformer.py) — **not** its own frame-sliding data
  loader — so it is scored on exactly the same agents as every other model. The DLow diversity
  head is out of scope for this configuration. Runs at 2 Hz with predictions interpolated to
  10 Hz (see the AgentFormer reproduction notes).
- *LED* — the upstream modules (Mao et al., CVPR 2023) with NBA shape constants made
  parametric, both training stages run on IDD-3D by
  [`scripts/model7_led_train_eval.py`](scripts/model7_led_train_eval.py). It is **not** the
  authors' released checkpoint (that core is trained on NBA basketball).

## Known limitations / next steps

- Ego compensation is annotation-based RANSAC (20.7% of frames fall back to a
  constant-velocity ego assumption); replace it with official ego poses when available. The raw
  IDD-3D label JSONs are not in this repo, so yaw and box sizes are unavailable.
- The test split is 22 scenes and val is 10. Absolute ADEs carry wide scene-bootstrap CIs, and
  conformal calibration on ~10 scenes is noisy. Use paired comparisons.
- Trajectron++'s export keeps only the longest contiguous run of each track. So 29 canonical
  windows (24 val, 5 test), whose track has a gap, have no Trajectron++ prediction, and
  constant velocity stands in. They are counted as `n_cv_fallback` in its metrics JSON.
- The seed-pooled significance test covers ST-GAT, the transformer and LED. The other models
  are compared on their seed-0 dumps only.
- Every ranked model is scored on the same windows, from the same 2 s observed history, and
  none sees the future at prediction time (checked in
  [`tests/test_eval_population_contract.py`](tests/test_eval_population_contract.py)).
  Trajectron++ keeps two upstream advantages:
  - it trains on every node-timestep of the train scenes, including agents with under 2 s
    of history. The other models train on the canonical train windows only.
  - at test time its scene graph also includes agents that are present but not part of the
    canonical windows.

  Both favour Trajectron++, so they make the other models' lead over it conservative.
  AgentFormer runs at 2 Hz, a cost it bears alone.
- Agent class labels are used only by the class-conditional conformal radii; no forecaster
  consumes them yet.
- CARLA is used only for visualization: [`scripts/viz_*.py`](scripts/) replay recorded
  IDD-3D and Argoverse 2 scenes with a model's predictions and conformal band. The
  collision-split evaluation and synthetic collision generation remain **planned work**, and
  no results exist yet (see report §7.3 for the protocol).
