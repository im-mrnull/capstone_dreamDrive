# Canonical Dataset + Frozen Splits

## Status: ✅ Fully Done

## What Changed from Legacy

The old pipeline used an "80-frame survivor filter" that discarded **58% of predictable
agents**. This reversed model rankings (ST-GAT lost its edge because the hard agents
were filtered out).

The canonical dataset requires only:
- Full 2s observation window (20 frames)
- ≥1 future step
- Per-timestep `fut_mask` for variable-length futures

## Population

- **10,576 agents** across 718 sequences (vs ~4,400 in legacy filtered data)
- All in world frame (ego-motion compensated)
- 25 IDD-3D agent classes preserved

## Frozen Splits

- **73 train / 10 val / 22 test** scenes
- Hash-based assignment (deterministic, seed-independent)
- Stored in `splits/scene_splits.json`
- Test set is genuinely harder: every model's test ADE = 1.2–1.8× val ADE

## Key Files

| File | Purpose |
|------|---------|
| `outputs/canonical_world_obs20_pred60.npz` | Canonical dataset (2.5MB) |
| `outputs/agent_windows_canonical_obs20_pred60.npz` | Agent windows (4.8MB) |
| `splits/scene_splits.json` | Frozen scene splits |
| `scripts/8_make_scene_splits.py` | Split generation script |
| `scripts/9_make_agent_windows_from_canonical.py` | Window extraction |
| `tests/test_canonical_dataset.py` | Dataset integrity tests |
| `tests/test_splits.py` | Split consistency tests |
