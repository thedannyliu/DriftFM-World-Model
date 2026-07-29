# Literature and novelty audit

Last verified: 2026-07-29. Conference papers and 2026 preprints are separated
deliberately. A recent arXiv paper is not described as a top-venue publication
unless an official proceedings page confirms the venue.

## Established top-conference results

| Work | Venue | Established result | Boundary for this project |
| --- | --- | --- | --- |
| [The Value Equivalence Principle for Model-Based RL](https://proceedings.neurips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html) | NeurIPS 2020 | Models can be trained to preserve Bellman updates useful for planning rather than reconstruct every transition. | “Decision relevance over pixel fidelity” is an evaluation principle, not our novelty. |
| [Static and Dynamic Values of Computation in MCTS](https://proceedings.mlr.press/v124/sezener20a.html) | UAI 2020 | Quantifies a computation by its expected effect on the quality of the action eventually chosen. | Choosing the next simulation by decision value is established. A generic value-of-computation router is not new. |
| [Multi-Fidelity Best-Arm Identification](https://proceedings.neurips.cc/paper_files/paper/2022/hash/71c31ebf577ffdad5f4a74156daad518-Abstract-Conference.html) | NeurIPS 2022 | Allocates cost among cheap biased and expensive accurate simulators; its guarantee assumes the maximum fidelity bias is known. | Cheap-to-expensive candidate elimination is prior art. We must test the fidelity-bias/order assumption rather than silently inherit it. |
| [Optimal Multi-Fidelity Best-Arm Identification](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dc9e095f668044e7a0909a4ea3926beb-Abstract-Conference.html) | NeurIPS 2024 | Gives an instance-dependent lower bound and asymptotically optimal allocation; identifies arm-specific optimal fidelity. | “Different candidates deserve different fidelity” is also not sufficient novelty. The open boundary is learned evaluators without trusted bias bounds. |
| [Consistency Models](https://proceedings.mlr.press/v202/song23a.html) | ICML 2023 | Supports one-step generation and multistep compute-quality tradeoffs in one model. | One/few-step generation with a variable compute knob is established. |
| [Parallel Sampling of Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0d1986a61e30e5fa408c81216a616e20-Abstract-Conference.html) | NeurIPS 2023 | ParaDiGMS parallelizes diffusion sampling and demonstrates DiffusionPolicy acceleration without measurable reward loss. | Compute, sequential steps, and wall-clock latency are distinct; comparisons must use measured end-to-end latency. |
| [Training-Free Adaptive Diffusion](https://proceedings.neurips.cc/paper_files/paper/2024/hash/00d1f03b87a401b1c7957e0cc785d0bc-Abstract-Conference.html) | NeurIPS 2024 | Adaptively skips diffusion predictions while preserving the full-step output, with reported 2--5× speedup. | Input-adaptive step skipping is occupied. Our target must be task decision regret rather than reproduction of a full-step sample. |
| [Is Best-of-N the Best of Them?](https://proceedings.mlr.press/v267/huang25c.html) | ICML 2025 | Connects Best-of-N performance to policy coverage, shows reward hacking under imperfect selection, and proposes scaling-monotonic pessimism. | Coverage-versus-selector error is a direct conceptual ancestor. The robotics contribution must measure these terms for action proposals and world-model evaluators. |
| [Test-Time Scaling of Diffusion Models via Noise Trajectory Search](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7e62167d35d1e830fd6afe5c899ed124-Abstract-Conference.html) | NeurIPS 2025 | Shows quickly diminishing returns from merely increasing denoising steps and searches noise trajectories for arbitrary rewards. | “More steps saturate” and reward-driven diffusion search are occupied; world-model decision structure must be the contribution. |
| [Diffusion Policy](https://roboticsproceedings.org/rss19/p026.html) | RSS 2023 | Introduces conditional action diffusion for visuomotor control across manipulation benchmarks. | Diffusion or flow action policies are baselines, not a new direction by themselves. |
| [RoboMonkey](https://proceedings.mlr.press/v305/kwok25a.html) | CoRL 2025 | Scales VLA action sampling and uses a learned verifier; reports action-error scaling and downstream gains. | Test-time action proposal sampling and verification are established. The unresolved choice is proposal breadth versus world-model evaluation compute. |
| [Verifier-free Test-Time Sampling for VLA Models](https://iclr.cc/virtual/2026/poster/10009251) | ICLR 2026 | MG-Select scores multiple VLA actions using KL divergence from a condition-masked reference distribution, without an external verifier. | Internal uncertainty-based candidate selection is a required baseline; “verifier-free confidence” is not a novel claim. |
| [TapSampling](https://openreview.net/forum?id=zeIa4sZCD1) | ICML 2026 | Generates action candidates through an Action-VAE and selects them with task-progress outcome prediction across policies and robot settings. | Learned task-progress verification and policy-agnostic action resampling are occupied. Our supervision must isolate world-model compute allocation and regret. |

## Closest 2026 work

The following items were current preprints at the audit date.

| Work | Relevant result | Boundary for this project |
| --- | --- | --- |
| [DriftWorld](https://arxiv.org/abs/2607.15065) and [official code](https://github.com/Susie-Lu/driftworld) | One-forward-pass action-conditioned world model and GPC-RANK planning; the official repository now includes Push-T and Robomimic code/checkpoints. | DriftWorld and GPC-RANK are the base system. The released Robomimic configs (`can_1view`, `lift_1view`, `lift_2view`) make Lift/Can the nearest cross-task path; Bridge-V2, RT-1, and Language Table code are still listed as forthcoming. |
| [Drift Flow Matching](https://arxiv.org/abs/2605.17244) | Learns arbitrary-time maps and reports extra-NFE gains in generation and robotic policies. | DFM, DFM policies, and generic test-time scaling are prior art. The question here is learned world-model depth as a decision evaluator. |
| [Flow Map Matching](https://arxiv.org/abs/2406.07507), [Shortcut Models](https://arxiv.org/abs/2410.12557), and [Consistency Flow Matching](https://arxiv.org/abs/2407.02398) | Learn direct/variable-step maps or enforce flow consistency. | A time-pair embedding, semigroup loss, or variable-step sampler alone is incremental. Exact consistency also does not imply that deeper approximate predictions improve action selection. |
| [SANTS](https://arxiv.org/abs/2605.27947) | Learns a state-adaptive stopping point along WAM video denoising using downstream action quality and reports large latency reductions. | Generic state-adaptive depth is occupied. Any new scheduler must be candidate-set-conditioned and decide between breadth and evaluator depth. |
| [Test-Time Scaling for WAMs via Zero-Shot Geometric Evaluation](https://arxiv.org/abs/2607.17454) | GeoBoN ranks independent WAM rollouts with geometry and gates extra samples using action-future consistency. | Best-of-N rollout selection, consistency gating, and adaptive extra sampling are occupied. Geometry is a baseline or input signal, not our central novelty. |
| [ACID](https://arxiv.org/abs/2607.02403) | Adds inverse-dynamics cycle action consistency to world-model planning and reports improvements across four world models and six tasks. | Action consistency is a strong decision-aligned verifier baseline. We should not claim first decision-centric world-model scoring. |
| [Is the Future Compatible?](https://arxiv.org/abs/2605.07514) | Diagnoses action-state consistency in WAM rollouts and uses consensus for selection. | Physical compatibility is established as a selector; it does not by itself solve breadth-depth allocation. |
| [Inference-Time Scaling via Progressive Seed Pruning](https://arxiv.org/abs/2607.21591) | Starts from many diffusion seeds and progressively prunes them using intermediate evidence under a fixed budget. | Generic “start broad, then refine survivors” is occupied. Our current static racing failure makes a calibrated no-order treatment essential. |
| [How Should World Models Be Evaluated for Embodied Decision-Making?](https://arxiv.org/abs/2606.15032) | Argues for a decision-centric evaluation ladder beyond visual prediction scores. | A benchmark that only says LPIPS is insufficient will not be novel. We need direct candidate regret and a method that changes decisions. |
| [Implicit Drifting Policy](https://arxiv.org/abs/2606.01098) and [Drifting Field Policy](https://arxiv.org/abs/2605.07727) | Drifting-style one-step action policies. | “Drift policy” is already crowded and is not the paper's name or core claim. |

## Synthesis

Three attractive but invalid novelty claims are now ruled out:

1. **“Decision fidelity over visual fidelity.”** Value Equivalence and recent
   decision-centric world-model work already establish this motivation.
2. **“Learn an adaptive NFE scheduler.”** AdaptiveDiffusion and SANTS already own
   input/state-adaptive depth.
3. **“Screen broadly, then refine a few candidates.”** Multi-fidelity BAI and
   progressive seed pruning already own the generic algorithmic pattern.

The remaining defensible object is:

> Learned inference depth is an untrusted, policy- and candidate-set-conditioned
> evaluator. Planning compute must be allocated between proposal coverage and
> selection accuracy using environment-calibrated marginal decision value, without
> assuming a fidelity order.

This boundary is narrower than the earlier “unordered generative fidelity” slogan.
The current Push-T results show that an ordered benefit cannot be assumed, but do
not statistically prove that every depth ordering is reversed. The word
**untrusted** is therefore safer until cross-task candidate-utility evidence is
available.

## What an oral-level paper must add

1. An exact coverage-regret plus selection-regret decomposition and a clear
   no-order counterexample or impossibility boundary.
2. A counterfactual decision dataset with true candidate continuation utilities,
   multiple policies, model seeds, and tasks.
3. A simple calibrated expand/refine/stop allocator that beats static breadth,
   static depth, SANTS-style scheduling, progressive pruning, and multi-fidelity
   elimination at matched latency.
4. Replication on Push-T, Robomimic Lift/Can, and a second world-model family.

Without all four, the strongest honest output is a careful negative benchmark or a
DFM-specific systems paper, not an oral-level general claim. The complete proposal
and kill criteria are in
[`oral-paper-blueprint.md`](oral-paper-blueprint.md).

## Search protocol

This audit used official NeurIPS/PMLR/RSS proceedings for venue claims, arXiv pages
for 2026 preprints, and the official DriftWorld GitHub repository for released-task
availability. Search themes were:

- value-aware and policy-aware world models;
- value of computation and metareasoning;
- multi-fidelity best-arm identification;
- diffusion/flow step adaptation and parallel sampling;
- Best-of-N coverage, pessimism, and reward-model error;
- robot action sampling, verification, WAM scheduling, and rollout consistency.

Refresh the audit immediately before submission; the 2026 WAM literature is moving
too quickly for this file to be treated as permanently complete.
