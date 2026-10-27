"""Scene-level bootstrap for benchmark v2 statistics.

Agents in the same IDD-3D scene share an ego drive, a road layout and each
other's interactions, so they are not independent draws. Resampling agents would
understate the uncertainty; the honest unit of resampling is the scene (the same
unit the frozen split is drawn in).

Every statistic here is a ratio of per-scene sums -- mean ADE is
sum(agent ADE) / n_agents, coverage is covered_steps / valid_steps -- so one
bootstrap replicate is a multinomial count vector over scenes and the statistic
is (counts @ num) / (counts @ den). Paired comparisons reuse the same count
matrix for both models, which is what makes the difference CI paired.
"""

from __future__ import annotations

import numpy as np


def scene_codes(scene_ids: np.ndarray, scenes: np.ndarray | None = None):
    """Map per-row scene ids to integer codes. Pass `scenes` (a fixed, sorted
    scene list) so several models index the same scene axis for paired tests."""
    if scenes is None:
        scenes = np.asarray(sorted(set(map(str, scene_ids))), dtype=object)
    pos = {s: i for i, s in enumerate(scenes)}
    return np.asarray([pos[str(s)] for s in scene_ids], dtype=np.int64), scenes


def per_scene_sums(codes: np.ndarray, n_scenes: int, values: np.ndarray) -> np.ndarray:
    """Sum `values` (rows) into scene bins."""
    return np.bincount(codes, weights=np.asarray(values, dtype=np.float64),
                       minlength=n_scenes)


def bootstrap_counts(n_scenes: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """(n_boot, n_scenes) multinomial resample counts: each replicate draws
    n_scenes scenes with replacement."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_scenes, np.full(n_scenes, 1.0 / n_scenes), size=n_boot)


def ratio_replicates(counts: np.ndarray, num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Bootstrap replicates of sum(num)/sum(den) over resampled scenes."""
    d = counts @ den
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, (counts @ num) / np.where(d > 0, d, 1.0), np.nan)


def ci(replicates: np.ndarray, level: float = 0.95) -> list:
    """Percentile interval of the finite replicates."""
    r = replicates[np.isfinite(replicates)]
    lo, hi = np.percentile(r, [50 * (1 - level), 50 * (1 + level)])
    return [float(lo), float(hi)]


def ratio_ci(codes: np.ndarray, n_scenes: int, num_rows: np.ndarray, den_rows: np.ndarray,
             counts: np.ndarray, level: float = 0.95) -> dict:
    """Point estimate + scene-bootstrap CI of sum(num_rows)/sum(den_rows)."""
    num = per_scene_sums(codes, n_scenes, num_rows)
    den = per_scene_sums(codes, n_scenes, den_rows)
    reps = ratio_replicates(counts, num, den)
    return {"estimate": float(num.sum() / den.sum()), "ci95": ci(reps, level),
            "replicates": reps}
