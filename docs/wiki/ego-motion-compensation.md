# Ego-Motion Compensation (Dataset Flaw Discovery)

## Status: ✅ Fully Done — Citable Finding

## The Problem

Raw IDD-3D labels store `psr.position` in the per-frame ego/LiDAR frame. Without
compensation, every trajectory mixes agent motion with ego vehicle motion. This made
the median apparent pedestrian speed **5.05 m/s** — impossibly fast (running pace).

## The Fix

`scripts/0_compensate_ego_motion.py` estimates per-frame SE(2) ego motion using RANSAC
rigid alignment on static objects (parked vehicles as inliers), then re-expresses all
positions in a per-scene world frame.

- **After compensation:** median pedestrian speed = **0.91 m/s** (walking pace) ✅
- **Method:** RANSAC rigid alignment, static object inliers
- **Limitation:** Approximation pending official ego poses / LiDAR odometry

## Key Files

| File | Purpose |
|------|---------|
| `scripts/0_compensate_ego_motion.py` | Main compensation script |
| `outputs/ego_compensation_stats.json` | Per-scene diagnostics (16KB) |
| `outputs/parsed_positions_world/*.csv` | Compensated world-frame positions |
| `tests/test_ego_compensation.py` | Regression tests |
| `tests/test_speed_sanity.py` | Speed distribution sanity checks |

## Impact

- All legacy ego-frame results are **deprecated** (they learned ego motion, not agent motion)
- This is the foundation for all Benchmark v2 results
- Publishable finding: IDD-3D raw labels need ego-motion compensation
