# Ego-Motion Compensation (Dataset Flaw Discovery)

## Status: ✅ Done · ⚠️ the estimate fails in 73 of 105 scenes (found 2026-11-25; see "Known failure" below)

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

## Known failure (found 2026-11-25)

In many scenes the RANSAC estimate locks onto moving traffic instead of static objects. The whole world frame,
road users included, then moves with that error.
- **Symptoms:**
  - the estimated travel runs backwards;
  - the estimated ego speed jumps by up to 24 m/s within 0.5 s;
  - many frames fall back to the constant-velocity ego.
- **Example:** in reversed scenes, the median pedestrian speed is 8.8 m/s.
- **Check:** `scripts/ego_quality.py`. Ego-only part, thresholds fixed on dev before it was applied to test:
  - travel not reversed;
  - at most 10 fallback frames;
  - ego speed change of at most 5 m/s within 0.5 s.
- **Result:** 32 of 105 scenes pass (26 dev, 6 test). The 73 failing scenes fail on:
  - speed jumps: 59;
  - fallback frames: 34;
  - reversed travel: 22;
  - a scene can fail more than one check.
- **Which scenes pass:** dense, pedestrian-rich scenes pass more often. That is plausible, because RANSAC then has
  enough static matches.
- **Effect on the reported results:** [[ego-quality-sensitivity]] re-scores every reported result on the clean
  scenes. The closed-loop and planner-headroom conclusions hold. The open-loop advantage of learned models over
  constant velocity, and the val→test drop in band coverage, come from the flagged scenes.
- **Not done:** re-estimating the ego motion and retraining on corrected data.
