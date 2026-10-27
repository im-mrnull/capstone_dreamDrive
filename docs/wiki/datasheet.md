# Datasheet for the Benchmark

## Status: ✅ Done (Oct 2026) — `DATASHEET.md` (repo root)

Gebru et al. (2021) format: motivation, composition, collection, preprocessing, uses,
distribution, maintenance. Every number is traced to a committed artifact.

## Facts worth remembering (corrections to older docs)

- **22 classes** in the benchmark (23 raw IDD-3D label types incl. `DontCare`), not "25".
- **10,576 = agent-windows, not agents.** They come from **4,905 unique tracks**: each
  10 s scene gives up to 7 overlapping 8 s windows (stride 3 frames).
- Only 4,523 (42.8%) windows have a fully observed 6 s future; mean 40.5/60 valid steps.
- Ego compensation falls back to constant velocity on 2,162 / 10,457 frames (20.7%).
- Per-split: train 73 scenes / 7,406 windows; val 10 / 993; test 22 / 2,177.

## Open items

- **Licence:** confirm IDD-3D terms allow publicly redistributing derived per-frame
  positions (this repo is public).
- Raw IDD-3D label JSONs are not available to the team → no yaw / box sizes.

Related: [[canonical-dataset]], [[ego-motion-compensation]]
