"""
Evaluate visual quality metrics for DriftWorld on Push-T.
"""
import time
import logging
import json
import os
import numpy as np
import torch
from omegaconf import OmegaConf

from data.pushT_dataloader import get_pushT_loader_shuffleFalse, get_pushT_full_loader
from .util_eval_setup import set_seed, setup_model
from .eval_metrics import get_mse, get_ssim, get_psnr, get_lpips
from gpc_rank.reward_predictor import RewardPredictor, estimate_reward_torch

log = logging.getLogger(__name__)


@torch.no_grad()
def _rollout_autoregressive(denoiser, all_obs, all_act, n_history, nfe):
    """
    Autoregressive rollout: condition on the GT seed window and chain the model's own
    predictions. Returns (B, T, C, H, W), where frames 0..n_history-1 are GT and the rest predicted.
    """
    T = all_obs.shape[1]
    cur_state = all_obs[:, :n_history]  # (B, M, C, H, W) GT initial frames s_0..s_{M-1}
    actions = all_act[:, :T - 1]        # F = T-1 actions a_0..a_{T-2} -> output length T
    return denoiser.sample_autoregressive(cur_state, actions, nfe=nfe)


@torch.no_grad()
def _warm_up_rollout(denoiser, all_obs, all_act, n_history, nfe):
    """Run one unmeasured rollout without changing the paired noise stream."""
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        _rollout_autoregressive(denoiser, all_obs, all_act, n_history, nfe)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _new_state():
    """Accumulators: per-video metric lists and timing totals."""
    return {
        "mse": [], "ssim": [], "psnr": [], "lpips": [],
        "final_block_xy_l2": [],
        "final_block_angle_abs_rad": [],
        "final_block_vertex_error": [],
        "total_gen_time": 0.0,
        "total_gen_frames": 0,
    }


def _summarize_state(s, video_len, nfe):
    tpf = (s["total_gen_time"] / s["total_gen_frames"]
           if s["total_gen_frames"] > 0 else float('nan'))
    summary = {
        "num_videos": len(s["mse"]),
        "video_len": video_len,
        "nfe": nfe,
        "mse": float(np.mean(s["mse"])),
        "ssim": float(np.mean(s["ssim"])),
        "psnr": float(np.mean(s["psnr"])),
        "lpips": float(np.mean(s["lpips"])),
        "total_generation_seconds": s["total_gen_time"],
        "generated_frames": s["total_gen_frames"],
        "seconds_per_frame": tpf,
        "per_video": {name: list(s[name]) for name in ("mse", "ssim", "psnr", "lpips")},
    }
    for name in (
        "final_block_xy_l2",
        "final_block_angle_abs_rad",
        "final_block_vertex_error",
    ):
        if s.get(name):
            summary[name] = float(np.mean(s[name]))
            summary["per_video"][name] = list(s[name])
    return summary


def _load_pose_predictors(cfg, device):
    eval_cfg = cfg.get("eval", {})
    xy_path = eval_cfg.get("reward_predictor_xy_checkpoint")
    angle_path = eval_cfg.get("reward_predictor_angle_checkpoint")
    if not xy_path and not angle_path:
        return None
    if not xy_path or not angle_path:
        raise ValueError("Both reward predictor checkpoints are required for block-pose metrics")

    xy_predictor = RewardPredictor().to(device)
    angle_predictor = RewardPredictor().to(device)
    xy_predictor.load_state_dict(torch.load(xy_path, map_location=device, weights_only=True))
    angle_predictor.load_state_dict(
        torch.load(angle_path, map_location=device, weights_only=True)
    )
    xy_predictor.eval()
    angle_predictor.eval()
    return xy_predictor, angle_predictor


def _predict_block_pose(image, predictors):
    xy_predictor, angle_predictor = predictors
    xy = xy_predictor(image.unsqueeze(0))[0]
    cossin = angle_predictor(image.unsqueeze(0))[0]
    cossin = cossin / torch.linalg.vector_norm(cossin).clamp_min(1e-8)
    angle = torch.atan2(cossin[1], cossin[0])
    return torch.stack((xy[0], xy[1], angle))


def _block_pose_errors(gen_pose, gt_pose):
    angle_delta = torch.atan2(
        torch.sin(gen_pose[2] - gt_pose[2]),
        torch.cos(gen_pose[2] - gt_pose[2]),
    ).abs()
    return {
        "final_block_xy_l2": torch.linalg.vector_norm(gen_pose[:2] - gt_pose[:2]).item(),
        "final_block_angle_abs_rad": angle_delta.item(),
        "final_block_vertex_error": estimate_reward_torch(gen_pose, gt_pose).item(),
    }


def evaluate_on_many_videos(cfg, num_videos=1000, video_len=64, step=None):
    """
    Evaluate visual quality metrics for DriftWorld on generated videos of length video_len.

    Args:
        cfg: Hydra config
        num_videos: number of videos to evaluate
        video_len: rollout length (overrides cfg.data.pred_horizon).
            If None, generate full-length videos at each episode's natural length.
        step: checkpoint step to load (None = latest)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_mode = (video_len is None)
    log.info(f"Full-video multiframe evaluation on Push-T (num_videos={num_videos}, "
             f"video_len={'FULL' if full_mode else video_len})")
    set_seed(cfg.train.seed)

    if not full_mode:
        OmegaConf.update(cfg, "data.pred_horizon", video_len, force_add=True)
        assert cfg.data.pred_horizon == video_len

    denoiser, device, _ = setup_model(cfg, step)
    pose_predictors = _load_pose_predictors(cfg, device)
    n_history = denoiser.num_history_frames
    nfe = cfg.get("eval", {}).get("nfe", 1)
    log.info(f"num_history_steps (history frames excluded from frame metrics): {n_history}")

    # Full-length episode vs fixed-length windows
    dataloader = get_pushT_full_loader(cfg) if full_mode else get_pushT_loader_shuffleFalse(cfg)

    # Metric accumulator
    s = _new_state()

    processed = 0
    for i, batch in enumerate(dataloader):
        log.info(f"(batch {i}/{len(dataloader)}) start")
        if processed >= num_videos:
            break

        # Pixels [0, 1] -> [-1, 1]
        all_obs = batch['image'].to(device)
        if cfg.data.normalize_img:
            all_obs = (all_obs - 0.5) / 0.5
        all_act = batch['action'].to(device)
        B = all_obs.shape[0]
        T = all_obs.shape[1]  # rollout length

        gt = all_obs

        if i == 0:
            log.info("Running one unmeasured rollout warm-up with preserved RNG state")
            _warm_up_rollout(denoiser, all_obs, all_act, n_history, nfe)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        gen = _rollout_autoregressive(denoiser, all_obs, all_act, n_history, nfe)  # (B, T, C, H, W) in [-1, 1]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_gen_time = time.perf_counter() - _t0

        batch_gen_frames = B * max(0, T - n_history)
        s["total_gen_time"] += batch_gen_time
        s["total_gen_frames"] += batch_gen_frames
        tpf = (s["total_gen_time"] / s["total_gen_frames"]) if s["total_gen_frames"] > 0 else float('nan')
        log.info(f"[timing] batch {i}: gen_time={batch_gen_time:.4f}s frames={batch_gen_frames} | "
                 f"running totals: gen_time={s['total_gen_time']:.4f}s frames={s['total_gen_frames']} "
                 f"time/frame={tpf:.4f}s")

        for j in range(B):
            if processed >= num_videos:
                break
            gt_j = gt[j]                      # (T, C, H, W)
            gt_g = gt_j[n_history:]           # generated-frame region (GT side)
            n_gen = gt_g.shape[0]
            if n_gen <= 0:
                log.warning(f"video {processed} (batch {i} sample {j}) has length {T} <= "
                            f"history {n_history}; skipping metrics")
                processed += 1
                continue

            gen_j = gen[j]                    # (T, C, H, W)
            gen_g = gen_j[n_history:]

            mse_m = float(get_mse(gen_g, gt_g).mean())
            ssim_m = float(get_ssim(gen_g, gt_g).mean())
            psnr_m = float(get_psnr(gen_g, gt_g).mean())
            lpips_m = float(get_lpips(gen_g, gt_g).mean())

            s["mse"].append(mse_m); s["ssim"].append(ssim_m); s["psnr"].append(psnr_m)
            s["lpips"].append(lpips_m)

            if pose_predictors is not None:
                gen_last = gen_g[-1]
                gt_last = gt_g[-1]
                if cfg.data.normalize_img:
                    gen_last = (gen_last * 0.5) + 0.5
                    gt_last = (gt_last * 0.5) + 0.5
                pose_errors = _block_pose_errors(
                    _predict_block_pose(gen_last, pose_predictors),
                    _predict_block_pose(gt_last, pose_predictors),
                )
                for name, value in pose_errors.items():
                    s[name].append(value)

            log.info(f"[metrics] video {processed} (batch {i} sample {j}, "
                     f"len={T}, n_gen={n_gen}): MSE={mse_m:.5f} SSIM={ssim_m:.5f} "
                     f"PSNR={psnr_m:.5f} LPIPS={lpips_m:.5f}")

            log.info(f"[running avg] over {len(s['mse'])} videos: "
                     f"MSE={np.mean(s['mse']):.5f} SSIM={np.mean(s['ssim']):.5f} "
                     f"PSNR={np.mean(s['psnr']):.5f} LPIPS={np.mean(s['lpips']):.5f}")

            processed += 1

    tpf = (s["total_gen_time"] / s["total_gen_frames"]) if s["total_gen_frames"] > 0 else float('nan')
    log.info(f"[timing] FINAL: total gen_time={s['total_gen_time']:.3f}s over "
             f"{s['total_gen_frames']} generated frames | time/frame={tpf:.5f}s")

    if len(s["mse"]) == 0:
        log.info("[summary] no videos were evaluated")
        return

    summary = _summarize_state(s, video_len, nfe)
    metrics_dir = cfg.get("eval", {}).get("metrics_dir", f"{cfg.output_dir}/metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    length_name = "full" if full_mode else str(video_len)
    metrics_path = os.path.join(metrics_dir, f"rollout_len-{length_name}_nfe-{nfe}.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Wrote summary to {metrics_path}")
    log.info(
        f"[summary] per-video averages over {len(s['mse'])} videos: "
        f"MSE={np.mean(s['mse']):.5f} SSIM={np.mean(s['ssim']):.5f} PSNR={np.mean(s['psnr']):.5f} "
        f"LPIPS={np.mean(s['lpips']):.5f}"
    )
    return summary
