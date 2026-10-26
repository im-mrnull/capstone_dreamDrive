# DreamDrive — Project Index

> **What is this?** A compact map of the entire project. AI agents should read this
> file FIRST before exploring the codebase. Each section links to a detailed note.

## Vault Structure

| Folder | Purpose |
|--------|---------|
| `raw/` | Quick captures — meeting notes, research dumps, AI conversation logs |
| `wiki/` | Polished knowledge — completed work, TODO specs, architecture decisions |
| `output/` | Deliverables — report sections, presentation content, paper drafts |

---

## Project Summary

- **Goal:** Multi-future trajectory prediction for unstructured Indian traffic (IDD-3D dataset)
- **Dataset:** IDD-3D — 105 scenes, 10 fps, 25 agent classes, ego-motion compensated
- **Window:** 2.0s observation (20 frames) → 6.0s prediction (60 frames), dt=0.1s
- **Population:** 10,576 agents (canonical dataset, no survivor filtering)
- **Split:** 73/10/22 train/val/test scenes (frozen, hash-based)
- **Seeds:** {0, 1, 2} for all neural models; CI-enforced provenance
- **Team:** Idhant Singh, Rishav Ghosh, Shagal Verma, Shourya Rai (PES University, PW26_CGM_04)

## Key File Locations

| Directory | Contents |
|-----------|----------|
| `scripts/` | All data processing, training, evaluation, and analysis scripts (29 files) |
| `outputs/benchmark_v2/` | Final result JSONs, checkpoints, prediction dumps, plots (54+ files) |
| `outputs/` | All intermediate data: parsed positions, windows, exported formats |
| `splits/` | Frozen `scene_splits.json` |
| `tests/` | 15 pytest files — unit tests + provenance enforcement |
| `.github/workflows/ci.yml` | CI pipeline: pytest + `make_tables.py --check` |
| `research/` | Report text and presentation notes |

## Data Pipeline (in order)

1. `scripts/1_parse_idd3d_positions.py` → `outputs/parsed_positions/*.csv`
2. `scripts/0_compensate_ego_motion.py` → `outputs/parsed_positions_world/*.csv`
3. `scripts/2_consolidate_trajectories.py` → `outputs/trajectories_world/trajectories.csv`
4. `scripts/3_make_windows_obs20_pred60.py` → `outputs/windows_world_obs20_pred60.npz`
5. `scripts/4_export_multiagent_trajpp_format.py` → `outputs/multiagent_world_v1/`
6. `scripts/5_export_stgat_format.py` → `outputs/stgat_dataset_world_obs20_pred60.npz`
7. `scripts/6_export_trajectronpp_world.py` → `outputs/trajectronpp_world/`

## Models Benchmarked (Benchmark v2 — Final)

| Model | Script | Test ADE (m) | Test FDE (m) | Status |
|-------|--------|-------------|-------------|--------|
| CV (vel_k=5) | `model1_baseline_cv_eval.py` | 4.36 | 9.78 | ✅ Done |
| MLP | `model2_baseline_mlp_train_eval.py` | 3.97±0.01 | 8.66±0.05 | ✅ Done |
| ST-GAT (MHA) | `model3_stgat_train_eval.py` | 3.41±0.07 | 7.35±0.16 | ✅ Done |
| Map-free transformer | `model5_transformer_train_eval.py` | 3.57±0.11 | 7.85±0.18 | ✅ Done |
| K-mode decoder (top-1) | `model4_mfp_train_eval.py` | 4.31±0.05 | 9.48±0.24 | ✅ Done |
| Trajectron++ (most-likely) | `7_train_eval_trajectronpp.py` | **2.58±0.04** | **5.58±0.07** | ✅ Best |
| AgentFormer | `model6_agentformer_train_eval.py` | — | — | ⚠️ Code only |

## Wiki Notes — Completed Work (13 items)

- [[ego-motion-compensation]] — dataset flaw discovery, speed 5.05→0.91 m/s
- [[canonical-dataset]] — 10,576 agents, frozen 73/10/22 split
- [[benchmark-v2]] — 6 models, 3-seed, CI-enforced results table
- [[fairness-audit]] — 16× ADE disparity across 15 classes
- [[conformal-prediction]] — 6 models, class-conditional uncertainty bands
- [[gap-decomposition]] — oracle vs deployable analysis (Trajectron++)
- [[negative-results]] — hetero T++, EWTA annealing, content scorer
- [[ci-provenance]] — structural anti-fabrication guards

## Wiki Notes — Not Started (12 items, priority order)

| Priority | Item | Impact |
|----------|------|--------|
| 🔴 1 | [[chaos-gap]] — internal note, local only | HIGHEST |
| 🔴 2 | [[carla-harness]] — Replay IDD-3D trajectories in CARLA | HIGH |
| 🔴 3 | [[planner]] — Frenét/MPC with conformal margins | HIGH |
| 🟡 4 | [[closed-loop-eval]] — Prediction → planning → collision rates | MEDIUM |
| 🟡 5 | [[bootstrap-ci]] — Scene-level resampling CIs | MEDIUM |
| 🟡 6 | [[datasheet]] — Datasheets-for-Datasets provenance doc | MEDIUM |
| 🟠 7 | [[rss-violations]] — internal note, local only | MEDIUM |
| 🟠 8 | [[two-wheeler]] — Lean dynamics, swarm behavior | MEDIUM |
| ⚪ 9 | Scenario suite / near-miss mining | LOW (needs CARLA) |
| ⚪ 10 | Indianized CARLA TM | LOW (needs CARLA) |
| ⚪ 11 | Culture-aware metrics | LOW |
| ⚪ 12 | Failure gallery + demo video | LOW |

## Partial Items (4)

- **AgentFormer:** Code exists, no 3-seed results yet
- **Named benchmark + packaging:** Internal harness exists, not packaged
- **Conformal → planner:** Radii computed, no planner consumes them
- **Two-wheeler:** Per-class conformal + unicycle ablation done, no dedicated analysis

## Latest Review

- [[2026-10-26-claude-project-review]]: methodology issues, literature/novelty check,
  prioritized publication roadmap, patent caveat

## Retraction Notice

Earlier versions contained fabricated results. All retracted. CI now enforces: every number
must trace to a committed artifact. See `research/report.txt` §7.2–7.3.
