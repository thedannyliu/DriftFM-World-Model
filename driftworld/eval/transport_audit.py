"""Diagnostics that localize Drift Flow transport-composition failures."""

from collections import defaultdict
import json
import logging
import os

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


GRIDS = {
    "nfe1": (0.0, 1.0),
    "nfe2": (0.0, 0.5, 1.0),
    "nfe4": (0.0, 0.25, 0.5, 0.75, 1.0),
}


def _pair_name(source, target):
    return f"{source:g}-{target:g}"


def _progress_stats(source, predicted, target):
    """Return displacement projection and orthogonal error relative to a paired target."""
    actual = (predicted - source).flatten(1)
    required = (target - source).flatten(1)
    denominator = required.square().sum(dim=1).clamp_min(1e-12)
    progress = (actual * required).sum(dim=1) / denominator
    residual = actual - progress[:, None] * required
    orthogonal = (
        residual.square().sum(dim=1) / denominator
    ).clamp_min(0).sqrt()
    return progress.mean().item(), orthogonal.mean().item()


def _append_prediction_metrics(values, prefix, source, predicted, target):
    progress, orthogonal = _progress_stats(source, predicted, target)
    values[f"{prefix}/paired_mse"].append(
        F.mse_loss(predicted.float(), target.float()).item()
    )
    values[f"{prefix}/particle_mean_mse"].append(
        F.mse_loss(
            predicted.float().mean(dim=0),
            target.float().mean(dim=0),
        ).item()
    )
    values[f"{prefix}/particle_std_mae"].append(
        F.l1_loss(
            predicted.float().std(dim=0, unbiased=False),
            target.float().std(dim=0, unbiased=False),
        ).item()
    )
    values[f"{prefix}/progress_ratio"].append(progress)
    values[f"{prefix}/orthogonal_error_ratio"].append(orthogonal)
    values[f"{prefix}/outside_range_fraction"].append(
        (predicted.abs() > 1.0).float().mean().item()
    )


def _velocity(model, state, history, actions, source, target):
    delta = target - source
    time_pair = state.new_tensor((source, delta)).expand(state.shape[0], 2)
    endpoint = model(state, history, actions, time_pair=time_pair)
    return endpoint - state


@torch.no_grad()
def evaluate_transport_audit(cfg):
    """Evaluate oracle local maps, free composition, and time-pair sensitivity."""
    from data.pushT_dataloader import get_pushT_validation_loader
    from .util_eval_setup import set_seed, setup_model

    set_seed(cfg.train.seed)
    denoiser, device, checkpoint_step = setup_model(cfg, step=None)
    if denoiser.objective != "drift_flow":
        raise ValueError("Transport audit requires a drift_flow checkpoint")

    audit_cfg = cfg.audit
    num_batches = int(audit_cfg.num_batches)
    particles = int(audit_cfg.particles)
    if num_batches < 1 or particles < 1:
        raise ValueError("audit.num_batches and audit.particles must be positive")

    dataloader = get_pushT_validation_loader(cfg)
    values = defaultdict(list)
    generator = torch.Generator(device=device).manual_seed(int(audit_cfg.seed))
    model = denoiser.ema_model

    for batch_index, batch in enumerate(dataloader):
        if batch_index >= num_batches:
            break
        log.info(f"[transport-audit] batch={batch_index + 1}/{num_batches}")
        obs = batch["image"].to(device)
        if cfg.data.normalize_img:
            obs = (obs - 0.5) / 0.5
        actions_all = batch["action"].to(device)
        if obs.shape[0] != 1:
            raise ValueError("Transport audit requires validation.batch_size=1")

        history_count = denoiser.num_history_frames
        future_count = denoiser.num_future_frames
        current_index = history_count - 1
        target = obs[:, current_index + 1:current_index + 1 + future_count]
        target = target.permute(0, 2, 1, 3, 4)
        history = obs[:, :history_count].permute(0, 2, 1, 3, 4)
        actions = actions_all[:, current_index:current_index + future_count]

        target = target.expand(particles, -1, -1, -1, -1)
        history = history.expand(particles, -1, -1, -1, -1)
        actions = actions.expand(particles, -1, -1)
        noise = torch.randn(
            target.shape, device=device, dtype=target.dtype, generator=generator
        )

        unique_pairs = {
            (source, target_time)
            for grid in GRIDS.values()
            for source, target_time in zip(grid[:-1], grid[1:])
        }
        for source_time, target_time in sorted(unique_pairs):
            source = (1.0 - source_time) * noise + source_time * target
            paired_target = (1.0 - target_time) * noise + target_time * target
            velocity = _velocity(
                model, source, history, actions, source_time, target_time
            )
            predicted = source + (target_time - source_time) * velocity
            pair = _pair_name(source_time, target_time)
            _append_prediction_metrics(
                values, f"teacher/{pair}", source, predicted, paired_target
            )

            endpoint_velocity = _velocity(model, source, history, actions, 0.0, 1.0)
            velocity_flat = velocity.flatten(1).float()
            endpoint_flat = endpoint_velocity.flatten(1).float()
            values[f"sensitivity/{pair}/cosine_to_endpoint"].append(
                F.cosine_similarity(velocity_flat, endpoint_flat, dim=1).mean().item()
            )
            values[f"sensitivity/{pair}/relative_norm_to_endpoint"].append(
                (
                    velocity_flat.norm(dim=1)
                    / endpoint_flat.norm(dim=1).clamp_min(1e-12)
                ).mean().item()
            )
            values[f"sensitivity/{pair}/mean_abs_difference"].append(
                (velocity - endpoint_velocity).abs().mean().item()
            )

        for grid_name, grid in GRIDS.items():
            state = noise.clone()
            for step_index, (source_time, target_time) in enumerate(
                zip(grid[:-1], grid[1:]), start=1
            ):
                paired_target = (
                    (1.0 - target_time) * noise + target_time * target
                )
                source = state
                velocity = _velocity(
                    model, state, history, actions, source_time, target_time
                )
                state = state + (target_time - source_time) * velocity
                _append_prediction_metrics(
                    values,
                    f"free/{grid_name}/step{step_index}",
                    source,
                    state,
                    paired_target,
                )
            _append_prediction_metrics(
                values, f"free/{grid_name}/final", noise, state, target
            )

    if not values:
        raise RuntimeError("Transport audit did not receive any validation batches")

    summary = {
        "status": "complete",
        "checkpoint": os.path.abspath(str(cfg.eval.checkpoint)),
        "checkpoint_step": checkpoint_step,
        "seed": int(cfg.train.seed),
        "num_batches": min(num_batches, batch_index + 1),
        "particles": particles,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "metrics": {
            key: float(sum(samples) / len(samples))
            for key, samples in sorted(values.items())
        },
    }
    output_path = os.path.abspath(str(audit_cfg.output))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
    return summary
