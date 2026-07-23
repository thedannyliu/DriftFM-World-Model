# Push-T research protocol

## Question and primary claim

Can an endpoint-compatible Drift Flow Matching world model allocate a fixed inference
budget between proposal breadth and transport depth better than DriftWorld's one-pass
generator?

The primary comparison is paired Push-T GPC-RANK under equal measured H100 wall-clock
budget. Success requires a 95% paired bootstrap confidence interval strictly above zero
for final IoU improvement on at least one of the released epoch-100 and epoch-300
policies. Effect size and latency are always reported even though no minimum effect size
is imposed.

## Controlled setup

- Task/data: released Push-T split and preprocessing.
- Input/output: four current/history RGB frames, four future actions, four future RGB
  frames at 96x96 pixels.
- Backbone: released factorized U-Net; only a zero-initialized time-pair adapter is
  added.
- Drift supervision: released per-future-frame normalized multi-temperature loss with
  temperatures 0.02, 0.05, and 0.2, initially using 16 particles.
- Baselines: released DriftWorld checkpoint, continued DriftWorld training with matched
  updates, and DriftFlowWorld post-training from that same checkpoint.
- Tracking: private W&B project `driftflowworld-pusht`; resumption reuses the stored run
  ID and checkpointed RNG/scaler states.
- Checkpoint selection: in each 500-episode domain, episodes 0–489 post-train and
  490–499 select `best` by fixed stochastic-objective loss. The released parent may
  already have seen all 500, so this is an adaptation-overfit monitor, not an unseen
  test set. Single-episode long-trajectory domains remain train-only. `latest` is the
  full resumable state; rollout metrics do not select checkpoints.

The released Push-T code is the executable baseline. Paper components not present in
that code (for example DINO supervision or no-action negatives) are not added to only
one comparison arm.

## Method under test

For source state `x_t`, source time `t`, and positive interval `delta`, predict

`T(x_t, t, delta, c) = x_t + delta * (G(x_t, c, t, delta) - x_t)`.

The time adapter is zero-initialized, so loading a released DriftWorld checkpoint and
using `(t, delta)=(0,1)` exactly preserves its endpoint map before post-training. Train
with 25% endpoint replay and 75% arbitrary time pairs sampled from a logit-normal
distribution. At inference, NFE=1 uses the original endpoint; NFE=2/4 composes equal
transport intervals while reusing the initial noise for paired comparisons.

## Staged gates

1. **Compatibility smoke:** checkpoint loads with only time-adapter keys missing;
   NFE=1 output is numerically equal before training.
2. **Baseline reproduction:** released 64-frame/full-video metrics fall within 10%
   relative MSE/LPIPS, 0.005 absolute SSIM, and 0.5 dB PSNR of reported values.
3. **10k pilot:** on the first 25 development rollouts, compare the matched continued
   DriftWorld arm with DriftFlowWorld NFE=1/2/4 before committing the remaining budget.
   This is a direction check, not the locked transport claim.
4. **Transport signal:** NFE=1 degrades by no more than 5%; NFE=4 improves LPIPS or
   block-pose error by at least 5% relative to NFE=1 on a locked validation set.
5. **Planning signal:** coarse-to-fine beats equal-wall-clock uniform one-step breadth
   with paired 95% bootstrap CI above zero on at least one locked policy.
6. **Confirmation:** repeat the selected configuration from scratch for seeds 1, 2,
   and 3.

If gate 3 fails after 100k post-training updates, run one bounded ablation wave:
uniform versus logit-normal time sampling, endpoint replay 0.25 versus 0.5, 16 versus
32 particles, and released drift versus grouped Sinkhorn drift, using 25k-update pilots.
Do not expand to Robomimic until the Push-T planning gate passes.

Block-pose error is a frozen-predictor proxy, not simulator ground truth: the released
GPC-RANK xy and angle predictors estimate the T-block pose from both the generated and
ground-truth final frame. Report xy L2, circular angle error, and the released
vertex-distance reward between those two estimates. The identical predictor on both
sides controls estimator bias while keeping the metric aligned with planning.

## Breadth-depth protocol

Calibrate end-to-end H100 latency for 50, 100, and 200 one-step proposals. Compare:

- uniform breadth: all proposals use NFE=1;
- uniform depth: fewer proposals all use NFE=4;
- coarse-to-fine: screen `M` proposals at NFE=1, then regenerate top `K` proposals from
  the same initial noise with NFE=4.

Tune only `K/M` in {0.1, 0.2, 0.4} on 25 development seeds. Lock the selected ratio and
evaluate 100 seeds for both released policies. Report mean IoU, paired differences,
95% bootstrap intervals, proposal/NFE counts, end-to-end latency, GPU, and peak memory.

## Cluster policy

Use PACE-Phoenix `embers` only. H100/H200/A100 jobs use account `gts-agarg35`; L40S
smokes use `gts-agarg35-ideas_l40s`. Never submit `inferno`. At most eight GPU jobs may
run concurrently. Start with CPU imports, one L40S forward smoke, and a two-GPU DDP
smoke before formal H100 training or timing.
