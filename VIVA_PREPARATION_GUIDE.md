# DreamDrive: Viva Preparation Guide

This is your comprehensive preparation guide for the DreamDrive Capstone Project Progress Review 1. It covers every slide, technical deep dives, likely panel questions, and how to defend your specific implementations based on the actual repository code.

---

## SLIDE 1 & 2 — Title and Outline

### 1. What this slide means
These are standard introductory slides setting the context.

### 2. 30–40 second script
"Good morning, respected panel. We are team PW26_CGM_04, and our project is DreamDrive. We are addressing a critical gap in autonomous driving: predicting the multi-future trajectories of agents in chaotic, unstructured Indian traffic. Today, we'll cover our Phase 2 progress, including dataset engineering, our 8-model benchmark, and our conformal uncertainty pipeline."

### 9. What I MUST remember
- Project ID: PW26_CGM_04.
- Core problem: Proactive prediction in *chaotic* traffic, not just structured Western roads.

---

## SLIDE 3 — Abstract and Scope

### 1. What this slide means
Explains the motivation. Reactive driving (single prediction) fails in chaotic traffic because you can't assume agents will follow lane rules. We need proactive, multi-future prediction to anticipate risks.

### 2. 30–40 second script
"Current autonomous systems are largely reactive and assume structured environments with lane discipline. In Indian traffic, this assumption breaks down. An auto-rickshaw might weave between cars, or a pedestrian might dart across the road. To drive safely, a vehicle cannot rely on a single deterministic forecast. DreamDrive aims to simulate multiple plausible futures for surrounding agents, allowing an autonomous planner to proactively avoid risks before they happen."

### 6. Likely panel questions
**Q1: Why is proactive prediction better than reactive?**
A: Reactive systems respond only after an agent makes a move, leading to sudden braking. Proactive systems predict multiple possible moves in advance, allowing the planner to gently adjust speed and position early.

**Q2: What do you mean by 'chaotic, unstructured traffic'?**
A: Traffic lacking strict lane discipline, high heterogeneity (rickshaws, animals, carts), and informal driving behaviors like weaving and occluded pedestrian crossings.

---

## SLIDE 4 — Summary of Work Done in Phase 2

### 1. What this slide means
This is the executive summary of your actual engineering contributions. It sets the agenda for the rest of the technical defense.

### 2. 30–40 second script
"In Phase 2, we accomplished several major engineering milestones. We discovered and fixed a critical ego-motion bug in the IDD-3D dataset that corrupted all trajectories. We built a canonical dataset containing over 10,000 agents. We implemented a rigorous benchmark with 3-seed evaluation to find the best predictor. We then wrapped these models in a conformal prediction pipeline to generate guaranteed safety bubbles. Finally, we automated our reporting with a CI/CD pipeline to ensure 100% reproducibility and documented three advanced techniques that failed."

### 3. Technical deep dive
- **Ego-Motion Bug Fix:** Handled in `scripts/0_compensate_ego_motion.py` using RANSAC SE(2) transforms.
- **Canonical Dataset:** `outputs/canonical_world_obs20_pred60.npz`. Replaced a flawed filtering rule that discarded 58% of agents.
- **CI/CD Honesty Checks:** `make_tables.py --check` and `tests/test_provenance.py` run on GitHub Actions.
- **3 Negative Results:** Unicycle physics, EWTA annealing, Content-Aware scorer.

### 6. Likely panel questions
**Q1: What exactly did you fix in the dataset?**
A: The raw IDD-3D dataset provided positions relative to the moving ego-vehicle. This meant stationary objects appeared to move backwards at the ego-vehicle's speed. We applied RANSAC to align consecutive frames and converted all coordinates to a static world frame.

---

## SLIDE 5 — Design Constraints

### 1. What this slide means
Explains the real-world limitations you faced and why the project is scoped the way it is.

### 2. 30–40 second script
"We operated under three main constraints. First, dataset constraints: Indian datasets like IDD-3D are small compared to Western datasets like Waymo, and have a massive domain gap. Second, computational constraints: generating and training on multiple futures (K trajectories) requires significant GPU memory, limiting real-time deployment. Third, simulation constraints: CARLA doesn't natively support Indian driving behaviors, meaning we must synthetically script traffic to validate our models in closed-loop."

### 3. Technical deep dive
- **Domain Gap:** Western data has lanes; Indian data has dense percolation.
- **Best-of-K training:** Models like Trajectron++ and AgentFormer generate multiple futures and use an oracle loss (Minimum over K (MoK) loss) during training, which scales computational cost linearly with K.

### 7. Counter-question chain
**Panel:** Why didn't you just collect your own data?
**Me:** Data collection requires a LiDAR-equipped vehicle, calibration, and hundreds of hours of manual annotation. IDD-3D is the only publicly available dataset of its kind for Indian traffic.
**Panel:** Then how do you plan to use CARLA if it doesn't support Indian traffic?
**Me:** We plan to import our real-world IDD-3D trajectories directly into CARLA as scripted actors, bypassing CARLA's default traffic manager.

---

## SLIDE 6 & 7 — Architecture

### 1. What this slide means
Outlines the data-to-prediction-to-evaluation workflow.

### 2. 30–40 second script
"Our architecture follows a rigorous pipeline. Raw IDD-3D labels undergo ego-motion compensation and are windowed into 2-second observation and 6-second prediction sequences. This feeds into our model zoo, ranging from simple Constant Velocity baselines to advanced generative models like Trajectron++. Finally, predictions are passed to our evaluation and conformal analysis modules to generate both deterministic error metrics and probabilistic safety boundaries."

### 3. Technical deep dive
- **2s obs / 6s pred:** We use 20 observation frames and 60 prediction frames at 10 FPS. This is standard in the trajectory prediction literature (e.g., nuScenes).
- **Scripts:** `3_make_windows_obs20_pred60.py` and `4_export_multiagent_trajpp_format.py` implement this.

### 9. What I MUST remember
- Observation = 2 seconds (20 frames).
- Prediction = 6 seconds (60 frames).
- Frequency = 10 FPS.

---

## SLIDE 8 — Data Pipeline

### 1. What this slide means
This is where the critical Ego-Motion bug is explained. It details the preprocessing.

### 2. 30–40 second script
"Our data pipeline is the foundation of our benchmark. The most critical step was our ego-motion compensation module. We discovered that the raw IDD-3D labels were in an ego-relative frame, meaning stationary pedestrians appeared to sprint at 5 m/s. We wrote a script using RANSAC rigid alignment to estimate the camera car's motion and project all trajectories into a static world frame. This successfully brought the median pedestrian speed down to a realistic 0.91 m/s."

### 3. Technical deep dive
- **RANSAC (Random Sample Consensus):** We match agent positions across frame `t` and `t+1`. RANSAC randomly selects pairs of points to compute a rigid SE(2) transformation (rotation + translation). It then counts how many other points fit this transformation (inliers). The transform with the most inliers is assumed to be the static world (parked cars, buildings), and the inverse of this transform is the ego-vehicle's motion.
- **Repository implementation:** `scripts/0_compensate_ego_motion.py`. It uses SVD (Umeyama's method) to solve for the SE(2) matrix inside the RANSAC loop.
- **Numbers:** 5.05 m/s -> 0.91 m/s. Sourced from `outputs/ego_compensation_stats.json`.

### 6. Likely panel questions
**Q1: Why did you use RANSAC instead of just subtracting the ego car's velocity?**
A: The IDD-3D dataset did not provide the ego car's velocity or poses. We had to reverse-engineer the ego motion purely from the movement of the annotated objects.

### 7. Counter-question chain
**Panel:** But what if all objects in the scene are moving? RANSAC would fail.
**Me:** You are absolutely correct. If the entire scene is a dense, uniformly moving platoon of cars, RANSAC might latch onto them. To mitigate this, we added a smoothness constraint (constant-velocity fallback) in `0_compensate_ego_motion.py` that prefers transforms close to the previous frame if inliers drop too low. We logged every time this fallback triggered—it happened in about 2100 frames out of 105 scenes.

---

## SLIDE 9 & 10 — Tools & Technical Dependencies

### 1. What this slide means
Standard tools list. Do not spend much time here.

### 2. 30–40 second script
"Our stack is built on Python and PyTorch, utilizing NumPy for our dataset engineering and conformal prediction modules. For modeling, we integrated the published Trajectron++ repository into our pipeline. Looking forward to Phase 3, we will use the CARLA simulator to test our trajectory predictions in closed-loop."

### 8. Vulnerabilities / things they may challenge
Do NOT claim you wrote Trajectron++ from scratch.
**Me:** "We adapted the open-source Trajectron++ architecture to process our custom canonical IDD-3D dataset, which required writing custom data exporters and training wrappers." (`scripts/6_export_trajectronpp_world.py`)

---

## SLIDE 11 — List of Modules

### 1. What this slide means
A quick roadmap of the codebase.

### 2. 30–40 second script
"We structured the project into 6 distinct modules. M1 handles our ego-motion fix and dataset creation. M2 and M3 represent our model zoo, containing 8 variants ranging from CV to Trajectron++. M4 is our novel conformal uncertainty module. M5 orchestrates our automated benchmarking, and M6 enforces integrity through our CI/CD pipeline."

---

## SLIDE 12 — Table 1: Benchmark Results

### 1. What this slide means
This is the most important empirical slide. It proves which model is best and shows that you ran rigorous experiments.

### 2. 30–40 second script
"This is our main benchmark result across 8 model variants, evaluated on a frozen test set across 3 random seeds. The key takeaway is that Trajectron++ is our best deployable model, achieving an ADE of 2.58 meters. You will also notice two variants of the K-mode decoder and Trajectron++ labelled 'oracle'. These are theoretical ceilings where the model gets to cheat by picking the best of its K generated futures. The large gap between deployable and oracle highlights the difficulty of mode selection in Indian traffic."

### 3. Technical deep dive
- **ADE (Average Displacement Error):** Mean L2 distance between predicted and ground truth waypoints over the 6-second future.
- **FDE (Final Displacement Error):** L2 distance at exactly t=6.0s (frame 60).
- **Oracle vs Deployable:** A deployable model must output a single prediction without seeing the future. An oracle model generates K futures and is evaluated on the one closest to the ground truth.
- **Implementation:** Evaluation logic is in `scripts/dd_data.py`. Table is auto-generated by `scripts/make_tables.py` using `outputs/benchmark_v2/summary.json`.

### 6. Likely panel questions
**Q1: What does the ± value mean?**
A: It is the standard deviation across 3 independent training runs with different random seeds. It proves our results are statistically significant and not just a lucky convergence.

**Q2: Why is the CV (Constant Velocity) baseline so bad (4.36m)?**
A: CV assumes agents move in straight lines at constant speeds. In chaotic Indian traffic, agents constantly weave, stop, and turn, making linear extrapolation fail miserably.

### 7. Counter-question chain
**Panel:** Why don't you deploy the Oracle model if it gets 1.37m ADE?
**Me:** By definition, an oracle cannot be deployed. It requires knowing the true future to select the best prediction. It only exists as an analytical tool to show the *potential* quality of the generated distributions.

---

## SLIDE 13 — Trajectron++ Oracle-vs-Deployable (Gap Decomposition)

### 1. What this slide means
Breaks down the 1.21m error difference between deployable T++ (2.58m) and oracle T++ (1.37m).

### 2. 30–40 second script
"Here we visualize the 1.21-meter gap between Trajectron++'s deployable performance and its oracle potential. The green zone shows that the model's internal mode-selector is working—it beats a completely random guess by 0.47 meters. However, the massive pink zone represents the inherent cost of committing to a single future in unpredictable traffic. Because the curve is still falling at K=20, it proves the model is generating genuinely diverse, multimodal futures, not just random noise."

### 3. Technical deep dive
- **X-axis:** K, the number of samples drawn from the model's latent distribution.
- **Y-axis:** Minimum ADE among those K samples (oracle ADE).
- **Code:** Plotted by `scripts/plot_gap_decomposition.py`. Data is generated by passing `--eval-best-of-k` to `7_train_eval_trajectronpp.py`.

### 6. Likely panel questions
**Q1: What do you mean by 'wasted knowledge'?**
A: The model's decoder is capable of generating a path with only 1.37m error (the oracle). The knowledge of the correct path is inside the model. But the model's likelihood estimator cannot confidently identify *which* of its 20 samples is the correct one, so it deploys a worse prediction.

---

## SLIDE 14 — Uncertainty Efficiency

### 1. What this slide means
Proves that a more accurate prediction model allows for smaller, tighter safety boundaries.

### 2. 30–40 second script
"This plot links prediction accuracy to planning efficiency. We applied conformal prediction to wrap all models in a 90% guaranteed safety bubble. The X-axis is the model's accuracy, and the Y-axis is the size of the required safety circle. Trajectron++ only needs a 10-meter circle, while the CV baseline needs 16 meters to offer the same 90% safety guarantee. The strong linear correlation of 0.93 proves that investing in better prediction models directly buys you tighter, more efficient maneuvers for the autonomous vehicle."

### 3. Technical deep dive
- **Conformal Prediction:** We compute nonconformity scores (Euclidean error) on a calibration set. The 90th percentile of these errors becomes the radius of our safety bubble.
- **Implementation:** `scripts/plot_uncertainty_efficiency.py` utilizing `outputs/benchmark_v2/conformal_metrics.json` generated by `scripts/conformal_wrapper.py`.

### 6. Likely panel questions
**Q1: What exactly is a 10m circle here? Is it around the agent?**
A: It is the radius of the prediction set at the final timestep. It means, with 90% probability, the agent's true future position will fall within a 10m radius of the model's point prediction.

---

## SLIDE 15 — Robustness Gap

### 1. What this slide means
Shows what happens when models are deployed on harder data than they were calibrated on (distribution shift).

### 2. 30–40 second script
"We also tested how these safety guarantees hold up under real-world distribution shift, specifically calibrating on the validation set but deploying on the harder test set. Every model loses coverage—meaning the 90% guarantee breaks. However, Trajectron++ degrades by only 5.2 points, while the baseline degrades by 15.2 points. This shows that stronger predictive models are not just more accurate, they are fundamentally more robust and trustworthy when traffic conditions unexpectedly change."

### 3. Technical deep dive
- **Regime A (val->test shift):** We use the validation split as the calibration set to find the conformal radius. We then check empirical coverage on the test split. Because the IDD-3D test split happens to contain harder, highly erratic scenarios, the computed radius is too small, and coverage drops below the 90% target.
- **Implementation:** `conformal_wrapper.py`, specifically the `Regime A` output.

---

## SLIDE 16 & 17 — Alpha Sensitivity

### 1. What this slide means
Defends the conformal results by proving you didn't just cherry-pick a 90% target to make your model look good. The patterns hold everywhere.

### 2. 30–40 second script
"A common question is whether our findings are just an artifact of choosing a 90% coverage target. We ran an alpha sensitivity analysis at 80%, 90%, and 95% targets. As shown in the left panel, the conformal math works perfectly in ideal conditions at every level. In the right panel, under distribution shift, the ranking of models never changes. Trajectron++ is always the most robust, and CV is always the worst. Table 3 shows the cost of safety: demanding 95% safety costs Trajectron++ an extra 5 meters of radius, but it costs the baseline 7.5 meters."

### 3. Technical deep dive
- **Alpha (α):** The allowed miscoverage rate. 90% target coverage = α of 0.10.
- **Code:** `scripts/conformal_alpha_sensitivity.py`.
- **Numbers:** Table 2/3 numbers are verifiable in `outputs/benchmark_v2/conformal_sensitivity.json`.

### 6. Likely panel questions
**Q1: Why does the band radius increase when you go from 90% to 95%?**
A: To capture 95% of future outcomes instead of 90%, the safety bubble must be drawn larger to encompass the more extreme, rare, or erratic edge-case behaviors of the agents.

---

## SLIDE 18 — Class Conditional Conformal

### 1. What this slide means
This is a highly novel contribution. It shows that using one size of safety bubble for all vehicles is dangerous and unfair.

### 2. 30–40 second script
"This plot represents one of our most important findings: class-conditional conformal prediction. A single, global safety circle is unfair. It over-covers slow pedestrians (giving them 96% coverage) and dangerously under-covers fast cars (providing only 78% safety). By computing separate conformal radii for each vehicle class—the orange dots—we pull every class back to the 90% target. Given the 16x disparity in error between animals and cars in Indian traffic, class-conditional safety margins are absolutely necessary."

### 3. Technical deep dive
- **Global vs Class-conditional:** Global groups all nonconformity scores together. Class-conditional groups them by `agent_type` (Car, Pedestrian, PoliceCar) and computes a separate quantile for each.
- **Code:** `radii_per_class()` inside `scripts/conformal_wrapper.py`.
- **PoliceCar:** The hardest class because it is fast, rare, and drives erratically. Global radius gave it 75% coverage; class-conditional bumped it to 89%.

### 7. Counter-question chain
**Panel:** What happens if you encounter a class that wasn't in your calibration set?
**Me:** We implemented a fallback mechanism in our code (`min_cal` parameter in `conformal_wrapper.py`). If an agent class has fewer than 30 observations in the calibration set, the code automatically falls back to using the global marginal radius to ensure statistical stability.

---

## SLIDE 20 — Future Work: Advanced Models

### 1. What this slide means
Acknowledges that while T++ is good, newer Transformer and Diffusion architectures exist.

### 2. 30–40 second script
"Looking ahead to Phase 3, we plan to benchmark more modern architectures. We have already written the integration code for AgentFormer, a Transformer-based CVAE model, and are awaiting GPU allocation for the 3-seed runs. Beyond that, we plan to explore Latent Equivariant Diffusion, or LED, which uses iterative denoising to generate highly realistic, physically plausible trajectory distributions."

### 3. Technical deep dive
- **AgentFormer:** `scripts/model6_agentformer_train_eval.py` and `scripts/dd_agentformer.py` exist in the repo. You actually wrote the adapter to feed your canonical dataset into it.
- **LED:** Not implemented yet. It's a pure future-work proposal.

---

## SLIDE 21 & 22 — Future Work: Beyond Benchmarking

### 1. What this slide means
This maps out how you transition from just drawing lines on a screen to actual autonomous driving.

### 2. 30–40 second script
"Ultimately, trajectory prediction is only useful if it prevents crashes. Our planned 'Chaos Gap' study will quantify exactly how much Western-trained models degrade when evaluated zero-shot on our Indian dataset. We also plan to integrate our conformal safety bubbles directly into a Frenét or MPC planner as strict geometric constraints. Finally, we aim to import our IDD-3D trajectories into the CARLA simulator to measure closed-loop safety metrics like collision rates and RSS violations."

### 8. Vulnerabilities / things they may challenge
Do NOT claim you have run CARLA or built the Frenét planner yet.
**Me:** "These are our concrete architectural plans for Phase 3. We have the conformal radii ready, and the next step is writing the cost function that consumes them."

---

## SLIDE 23 — Individual Contribution

### 1. What this slide means
Proves who did what.

### 2. 30–40 second script
"To summarize our individual contributions: I focused on the ego-motion compensation algorithm, integrating the Trajectron++ architecture, and building our CI/CD and conformal analysis pipelines. Rishav spearheaded the core conformal wrapper and fairness audits. Shagal managed the data pipeline and AgentFormer scaffolding, and Shourya developed our baseline models and testing frameworks. Each of us contributed roughly 600 to 800 lines of core logic over the past few weeks."

### 9. What I MUST remember (Idhant's contributions)
- You wrote `0_compensate_ego_motion.py`
- You wrote `6_export_trajectronpp_world.py` and `7_train_eval_trajectronpp.py`
- You wrote `.github/workflows/ci.yml` and `test_provenance.py`
- You wrote `conformal_alpha_sensitivity.py`

If asked: "Explain your ego-motion code", be ready to talk about RANSAC SE(2) transforms.

---

## SLIDE 24 & 25 — Timeline and Demonstration

### 1. What this slide means
Checks the boxes for the academic review requirements.

### 2. 30–40 second script
"We have successfully completed all Phase 2 modules on schedule, including data fixes, benchmarking, and uncertainty analysis. Our deliverables are fully tested. We have automated sanity checks enforcing physical speed limits, mathematical tests validating conformal coverage, and strict provenance tests ensuring every number in our presentation matches a committed artifact in our repository. There is zero possibility of fabricated results in this project."

### 3. Technical deep dive
- **test_speed_sanity.py:** Fails if pedestrian median speed > 2 m/s.
- **test_conformal.py:** Asserts empirical coverage >= 89%.
- **test_provenance.py:** Opens `README.md` and checks for banned text (old fabricated numbers).

---

# FINAL VIVA PREPARATION SECTION

### 1. PROJECT IN ONE MINUTE
"DreamDrive solves the problem of autonomous navigation in chaotic Indian traffic. Western self-driving cars rely on lane rules and single-future predictions, which fail when rickshaws and pedestrians weave unpredictably. We took the IDD-3D dataset, fixed a massive ego-motion bug using RANSAC, and built a canonical multi-agent dataset. We benchmarked 8 models and proved that Trajectron++ predicts Indian traffic best. More importantly, we wrapped these models in a conformal prediction pipeline to output guaranteed 'safety bubbles' around agents. We proved that because of the massive diversity in Indian traffic, autonomous vehicles must use class-conditional safety margins—drawing different sized bubbles for cars versus pedestrians—to drive safely."

### 4. TOP VIVA QUESTIONS & ANSWERS

**Q: Why use IDD-3D instead of nuScenes or Waymo?**
A: nuScenes and Waymo contain highly structured Western traffic with strict lane discipline. Autonomous systems trained on them fail in unstructured environments. IDD-3D captures the percolation and heterogeneous agents unique to Indian roads.

**Q: What is conformal prediction and why didn't you just use standard deviations?**
A: Standard deviations assume errors follow a normal (Gaussian) distribution, which is false for complex trajectories. Conformal prediction is distribution-free. It uses empirical error quantiles to guarantee that the true future falls in the predicted set 90% of the time, regardless of the underlying distribution.

**Q: You mention 'wasted knowledge' and the 1.21m gap. How do you fix that?**
A: The gap shows the generative backbone is excellent, but the mode-scoring likelihood is weak. Our negative result with the 'content-aware scorer' showed that just re-scoring the output doesn't work. The real fix likely requires injecting explicit scene context or intent classifiers into the latent variable selection process.

### 5. TRAP QUESTIONS
**Trap:** "Your Trajectron++ oracle achieves 1.37m ADE. Why don't you deploy that?"
**Defuse:** "The oracle is an analytical metric, not a deployable model. It cheats by looking at the ground truth to select the best of 20 predictions. We report it only to quantify the model's generative capacity. The true deployable performance is 2.58m."

**Trap:** "How do you know your conformal 90% guarantee will hold in the real world?"
**Defuse:** "We specifically proved that it *won't* hold perfectly if the real world is harder than the calibration data. Our Regime A (val-to-test shift) experiment showed coverage drops from 90% to ~85% under distribution shift. However, we proved that using stronger models like Trajectron++ minimizes this degradation."

### 10. STRONGEST CONTRIBUTIONS
1. RANSAC Ego-motion fix saving the IDD-3D dataset from contamination.
2. The Canonical Dataset definition (10,576 agents) preventing metric cheating.
3. Class-Conditional Conformal Prediction proving global safety radii are dangerous.
4. The 8-model rigorous benchmark with automated CI/CD provenance.

### 11. BIGGEST LIMITATIONS
1. We still don't have a closed-loop CARLA simulation running to prove ADE correlates with fewer crashes.
2. The models lack map/HD-map context (lane boundaries/drivable area).
3. The dataset is still small (105 scenes) compared to industry standards.

Good luck! You have built a highly rigorous, well-tested repository. Stick to what the code actually does, and you will do great.
