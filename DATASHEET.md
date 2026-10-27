# Datasheet: DreamDrive IDD-3D Trajectory Benchmark (v2)

Following *Datasheets for Datasets* (Gebru et al., CACM 2021). This describes the
**derived trajectory-prediction benchmark** built in this repository from IDD-3D, not
IDD-3D itself; for the source dataset, see the IDD-3D paper (Dokania et al., WACV 2023)
and <https://idd3d.github.io/>.

Every number below is computed from a committed artifact. The artifact paths appear next to the
numbers.

---

## 1. Motivation

**Why was it created?** Public trajectory-prediction benchmarks (nuScenes, Argoverse,
Waymo) are recorded in lane-based Western traffic. IDD-3D records unstructured Indian
traffic: lane-free, dense, with many two-wheelers, but it was released for 3D detection and
tracking only. This benchmark turns its tracks into a forecasting task with a frozen test
set, so models can be compared on chaotic mixed traffic.

**Who created it?** Capstone team PW26_CGM_04, PES University (Idhant Singh, Rishav Ghosh,
Shagal Verma, Shourya Rai).

**Funding.** None beyond the university capstone program.

## 2. Composition

**What does an instance represent?** One *agent-window*: an agent observed for 2.0 s
(20 frames at 10 Hz) and the following 6.0 s (60 frames) to predict, in a per-scene world
frame (meters), with a per-step validity mask for the future. Agents are grouped into
*sequences*: all agents co-present in one 8 s window of one scene. Interaction models
consume a whole sequence.

| Quantity | Value | Source |
|---|---|---|
| Scenes (IDD-3D sequences, ~10 s each, 10 Hz) | 105 | `splits/scene_splits.json` |
| Annotated frames | 10,457 | `outputs/parsed_positions_world/*.csv` |
| Unique object tracks | 4,905 | `outputs/parsed_positions/*.csv` |
| Sequences (8 s windows, stride 3 frames, ≥ 2 agents) | 718 | `outputs/canonical_world_obs20_pred60_stats.json` |
| Agent-windows (instances) | 10,576 | same |
| ... with a fully observed 6 s future | 4,523 (42.8%) | same |
| Mean valid future steps per instance | 40.5 of 60 | `outputs/canonical_world_obs20_pred60.npz` |

**Instances are not independent.** Up to 7 overlapping windows come from each 10 s scene,
so one physical agent can appear in several instances, and agents in a scene share an ego
drive, a road layout and each other's interactions. Every split is therefore made at the scene
level. Uncertainty estimates in this repo resample scenes (`scripts/bootstrap_ci.py`), and
conformal calibration folds are split by scene (`scripts/conformal_wrapper.py`).

**Splits** (frozen; `md5(salt:scene_id)` buckets, salt `dreamdrive-v2.9`):

| Split | Scenes | Sequences | Agent-windows |
|---|---|---|---|
| train | 73 | 498 | 7,406 |
| val | 10 | 70 | 993 |
| test | 22 | 150 | 2,177 |

The test split is empirically harder than val: every benchmarked model's test ADE is
1.2–1.8× its val ADE. With only 10 val scenes, choices made on val, and conformal
calibration on val, carry substantial sampling noise.

**Labels.** Each instance carries the IDD-3D object class of its track. The benchmark
contains 22 classes. By agent-window count: Pedestrian 2,058; Car 1,758; MotorcyleRider
1,670 (*sic*, IDD-3D's spelling); TourCar 1,469; ScooterRider 1,436; Truck 955; Scooter 324;
Bus 269; Motorcycle 246; Unknown 87; Van 45; BicycleGroup 32; Animal 32; Bicycle 31;
BicycleRider 30; PoliceCar 29; Misc 29; Unknown1 28; ConcreteTruck 27; Child 10; Crane 8;
Unknown3 3. The raw parsed data has one more type (`DontCare`, 1 track) that does not
survive windowing. Riderless `Motorcycle`/`Scooter` objects are mostly parked, so they are
near-static agents.

**Missing information.** There are no maps or lanes (the roads are lane-free), no traffic signals,
no box size or heading in the parsed positions, and no official ego poses (see §4). Futures are
truncated wherever a track ends or leaves the annotated range; the mask records this.

**Confidential or sensitive data.** The benchmark has positions and class labels only. It has no images,
point clouds, faces or plates. Locations are scene-relative world frames, not GPS.

## 3. Collection process

The data was recorded and annotated by the IDD-3D authors: multi-camera and LiDAR driving on Indian
roads, with 3D boxes and track IDs. This repository uses only the annotation JSONs. From each object's
box center (`psr.position`, x/y ground plane), `scripts/1_parse_idd3d_positions.py`
writes per-frame positions. **The raw label JSONs are not in this repository**, and the current
maintainers do not hold a copy, so the parse step cannot be re-run here. Everything downstream
starts from the committed `outputs/parsed_positions/*.csv`.

## 4. Preprocessing

1. **Ego-motion compensation** (`scripts/0_compensate_ego_motion.py`). IDD-3D positions are
   stored in the moving ego/LiDAR frame. Per-frame SE(2) ego motion is estimated from the
   annotations themselves: RANSAC rigid alignment of consecutive frames, where static objects are the
   inliers (inlier threshold 0.12 m, 200 iterations, ≥ 3 inliers). All positions are then
   re-expressed in a per-scene world frame. Validation: median pedestrian speed drops from 5.05 m/s
   (raw) to 0.91 m/s (world frame). **2,162 of 10,457 frames (20.7%)** fall back to a
   constant-velocity ego assumption (too few static inliers). This is an approximation;
   official ego poses would replace it (`outputs/ego_compensation_stats.json`).
2. **Consolidation and windowing** (`scripts/2_…`, `4_…`, `5_…`). Tracks are merged per scene, cut
   into 8 s windows (stride 3 frames), and every agent with a full 2 s history and at least one valid
   future step is kept. The earlier "all 80 frames present" filter removed 58% of
   predictable agents and changed model rankings; it is no longer used.
3. **Splits** (`scripts/8_make_scene_splits.py`). These are hash-based and seed-independent. Never change
   them without a new version tag: test numbers are final.

**Known errors and noise.** Ego compensation is approximate (above). World-frame speed
tails are plausible but not perfect (pedestrian p99 = 12.5 m/s, which suggests residual
compensation or annotation errors in a small share of frames). Box centers jitter at the
annotation level; no smoothing is applied.

## 5. Uses

**Intended uses.** Benchmarking map-free trajectory forecasting in unstructured, heterogeneous
traffic; per-class error audits; studying calibrated uncertainty (conformal prediction)
under scene clustering and split shift.

**Not suitable for:**
- claims about safety in deployment;
- city-scale or country-scale generalization (105 short scenes from one dataset);
- tasks that need maps, headings or vehicle extents;
- per-class conclusions for rare classes. Several classes appear in only 2–4 test scenes, and their
  bootstrap CIs are wide (`outputs/benchmark_v2/bootstrap_ci.json`).

**Risks.** The splits are small: an absolute test ADE has a wide scene-bootstrap CI (for the best
model, Trajectron++ seed 0: 2.54 m, 95% CI 1.64–3.87 m). Comparisons should use paired statistics. Treating agent-windows as
independent samples overstates the effective sample size.

## 6. Distribution

The benchmark is distributed as derived files in this repository: parsed and world-frame positions, the
canonical NPZ, splits and metrics. **Licence: open item.** IDD-3D's terms of use govern
any redistribution of data derived from it. Confirm with the IDD-3D providers that
publicly sharing per-frame derived positions is permitted before a formal release.

## 7. Maintenance

The PW26_CGM_04 team maintains it through the GitHub repository. Results are protected by CI:
`scripts/make_tables.py --check` fails if a README table is not reproduced from a
committed artifact. Earlier unbacked results were retracted (README, "Results integrity
policy"). Any change to the data or splits must bump the split salt and invalidates
all test numbers.
