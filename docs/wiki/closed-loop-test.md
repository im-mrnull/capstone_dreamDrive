# Closed-Loop Test — does a predictor make a planner drive safely?

## Status: 📌 Pre-registered (2026-10-27) — test results pending

The open-loop [[safety-scorecard]] could not separate the predictors: all single-trajectory
models tied with constant velocity. This test puts each predictor inside the same planner and
drives the recorded test scenes. The rule below, and the simulator code in the same commit, were
fixed **before** any learned predictor was run on the test scenes. The result JSON records the
commit as `rule_commit`.

## Setup (user choices: Python replay simulator, non-reactive agents, speed + small swerves)

- **Scenes / episodes:** the 22 test scenes (10 s each).
  - Episodes start at 1.9, 2.9, 3.9 and 4.9 s and last 5 s.
  - Two ego personas: *nominal* (reference speed = the human driver's) and *assertive*
    (reference speed at least 6 m/s, which forces interaction).
  - That gives 176 episodes per package and seed.
- **Other road users:** they replay their logged tracks and do not react to the ego.
  - Footprints are nominal per class, because IDD-3D box sizes are not in this repo:
    - pedestrian r 0.3 m;
    - two-wheeler 1.9 × 0.7 m;
    - car 4.3 × 1.7 m;
    - truck/crane 6.8 × 2.4 m;
    - bus 8.5 × 2.5 m;
    - other r 0.5 m.
  - Headings come from the logged motion.
- **Ego:** a 4.5 × 1.7 m car centred 0.5 m ahead of the logged sensor position. This was measured
  from where annotated agents never appear around the sensor on train+val scenes.
  - It follows the human driver's smoothed route in Frenet coordinates.
- **Planner:** one design, identical for every package.
  - It replans every 0.2 s over 35 manoeuvres: accelerations {−6, −4, −2, −1, 0, 1, 2} m/s² ×
    lateral offsets {−1.5, −0.75, 0, 0.75, 1.5} m, reached over 1.5 s.
  - Horizon 3 s, checked every 0.1 s for the first second and every 0.2 s after.
  - A manoeuvre is *clear* if the ego footprint never overlaps any agent's predicted footprint
    inflated by the package's **val-calibrated 90% conformal band** (per class, per step) plus a
    fixed 0.25 m buffer.
  - Among clear manoeuvres it minimises ((v_end − v_ref)/2)² + 0.05a² + 0.3d² + 0.5|d − d_now|.
  - If none is clear, it takes the one with the least relative penetration.
- **Predictions:** every agent with a full 2 s history at the planning step gets the package's
  prediction (or its 10 samples for sets).
  - Agents seen for less than 2 s get constant velocity with the CV band, the same for every
    package.

## Metrics

- **At-fault collision** (nuPlan-style for non-reactive replay): a footprint overlap while the
  ego moves at ≥ 0.5 m/s and either:
  - the agent is in front (centre ≥ 1 m ahead of the ego centre, laterally within the ego plus
    agent half-widths, plus half the ego's); or
  - the ego is swerving toward it (lateral speed > 0.1 m/s, agent not behind the ego's rear
    half).

  Excluded: an agent cutting in sideways at > 2 m/s.
- **Near miss:** a footprint gap < 0.5 m to an agent no more than 1 m behind the sensor.
  **Harsh braking:** any chosen deceleration ≤ −4 m/s².
  **Progress:** ego distance ÷ human distance over the episode, capped at 1, and 1 when the human
  moved < 2 m.
- **Episode score:** 0 after an at-fault collision; otherwise
  (5·progress + 2·no-harsh-braking + 2·no-near-miss) / 9.

## Decision rule

1. **Primary:** the mean episode score on the 22 test scenes for each of the 12 packages from the
   scorecard, 3-seed mean, with a seed-pooled scene bootstrap (2,000 replicates; episodes of a
   scene resampled together, shared by all packages; each package's seeds resampled).
2. **Leader** = highest score. **Tied set** = packages whose paired difference vs the leader has a
   95% CI including 0.
3. **Tie-break 1:** the lowest rate of episodes with an at-fault collision with a pedestrian or
   two-wheeler rider. Packages tied with the best (paired CI) stay.
4. **Tie-break 2:** the lowest median latency (`safety_latency.json`).

**References (reported, not eligible):**
- human driver (log replay): checks the footprints;
- oracle (true future, no band): an upper bound;
- no prediction (ignore agents): shows what prediction buys.

**Reported, not used for the pick:** the same packages without the band (λ = 0); per-persona
results; at-fault, near-miss, harsh-braking and progress rates.

## Development record (validation scenes, reference drivers only — no learned predictor)

The simulator was debugged on the 10 val scenes with the three reference drivers only:

1. Ego footprint centred on the sensor → re-centred 0.5 m ahead (measured).
   Truck 8.5 m → 6.8 m, the human replay uses the raw logged position.
2. At-fault convention added. The perfect-prediction planner was "colliding" with a sideways
   8 m/s tracking glitch and a truck beside the route.
3. Planner checks densified (0.1 s for the first second) and a 0.25 m buffer added, after
   swerve contacts of a few cm.

Final val check: human replay 0% at-fault, oracle 0%, ignore-agents 3.7%.

**Caveats:** non-reactive agents; nominal footprints; ego path = the human's route; 22 test scenes.

Related: [[safety-scorecard]], [[planner]], [[closed-loop-eval]], [[benchmark-v2]]
