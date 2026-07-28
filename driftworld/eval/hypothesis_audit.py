"""Per-example diagnostics for advantage-aligned transport hypotheses."""

from collections import defaultdict
import json
import logging
import os

import numpy as np
import torch

log = logging.getLogger(__name__)


GRIDS = {
    1: (0.0, 1.0),
    2: (0.0, 0.5, 1.0),
    4: (0.0, 0.25, 0.5, 0.75, 1.0),
    8: (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0),
}
TRANSITIONS = ((1, 2), (2, 4), (4, 8))


def _mse_per_sample(left, right):
    return (left.float() - right.float()).square().flatten(1).mean(dim=1)


def _mean(values):
    return float(np.mean(values)) if values else float("nan")


def _pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 3 or left.std() < 1e-12 or right.std() < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _step(denoiser, state, history, actions, source, target):
    delta = target - source
    time_pair = state.new_tensor((source, delta)).expand(state.shape[0], 2)
    endpoint = denoiser.ema_model(
        state, history, actions, time_pair=time_pair
    )
    if source == 0.0 and target == 1.0:
        return endpoint
    source_tensor = state.new_full((state.shape[0], 1, 1, 1, 1), source)
    delta_tensor = state.new_full((state.shape[0], 1, 1, 1, 1), delta)
    scale = denoiser._transport_scale(source_tensor, delta_tensor)
    return state + scale * (endpoint - state)


def _load_pose_predictors(cfg, device):
    from gpc_rank.reward_predictor import RewardPredictor

    xy_path = str(cfg.eval.reward_predictor_xy_checkpoint)
    angle_path = str(cfg.eval.reward_predictor_angle_checkpoint)
    xy_predictor = RewardPredictor().to(device)
    angle_predictor = RewardPredictor().to(device)
    xy_predictor.load_state_dict(
        torch.load(xy_path, map_location=device, weights_only=True)
    )
    angle_predictor.load_state_dict(
        torch.load(angle_path, map_location=device, weights_only=True)
    )
    xy_predictor.eval()
    angle_predictor.eval()
    return xy_predictor, angle_predictor


def _predict_pose(images, predictors):
    xy_predictor, angle_predictor = predictors
    xy = xy_predictor(images)
    cossin = angle_predictor(images)
    cossin = cossin / torch.linalg.vector_norm(
        cossin, dim=1, keepdim=True
    ).clamp_min(1e-8)
    angle = torch.atan2(cossin[:, 1], cossin[:, 0])
    return torch.cat((xy, angle[:, None]), dim=1)


def _vertex_error_per_sample(predicted, target):
    from gpc_rank.reward_predictor import estimate_reward_torch

    return torch.stack([
        estimate_reward_torch(predicted[index], target[index])
        for index in range(predicted.shape[0])
    ])


def _images_01(images, normalized):
    return images * 0.5 + 0.5 if normalized else images


def _record_mean(record, name, values):
    record[name] = float(values.mean().item())


def _summarize_records(records):
    metric_names = sorted({
        name for record in records for name in record
        if name != "example"
    })
    metrics = {
        name: _mean([record[name] for record in records if name in record])
        for name in metric_names
    }
    correlations = {}
    for shallow, deep in TRANSITIONS:
        defect_key = f"route/{shallow}_to_{deep}/defect_mse"
        degradation_key = f"degradation/{shallow}_to_{deep}/mse"
        pose_degradation_key = f"degradation/{shallow}_to_{deep}/pose_vertex"
        correlations[
            f"defect_vs_degradation/{shallow}_to_{deep}/mse"
        ] = _pearson(
            [record[defect_key] for record in records],
            [record[degradation_key] for record in records],
        )
        correlations[
            f"defect_vs_degradation/{shallow}_to_{deep}/pose_vertex"
        ] = _pearson(
            [record[defect_key] for record in records],
            [record[pose_degradation_key] for record in records],
        )

    strata = {}
    motion = np.asarray(
        [record["difficulty/motion_vertex"] for record in records],
        dtype=np.float64,
    )
    ranked_groups = dict(zip(
        ("low", "mid", "high"),
        np.array_split(np.argsort(motion, kind="stable"), 3),
    ))
    for label, indices in ranked_groups.items():
        selected = [records[int(index)] for index in indices]
        strata[label] = {
            "count": len(selected),
            "motion_vertex": _mean([
                record["difficulty/motion_vertex"] for record in selected
            ]),
        }
        for shallow, deep in TRANSITIONS:
            for risk in ("mse", "pose_vertex"):
                key = f"degradation/{shallow}_to_{deep}/{risk}"
                strata[label][key] = _mean([
                    record[key] for record in selected
                ])
    return metrics, correlations, strata


@torch.no_grad()
def evaluate_hypothesis_audit(cfg):
    """Measure route defect, source shift, and action-relevant degradation."""
    from data.pushT_dataloader import get_pushT_validation_loader
    from .util_eval_setup import set_seed, setup_model

    set_seed(cfg.train.seed)
    denoiser, device, checkpoint_step = setup_model(cfg, step=None)
    if denoiser.objective != "drift_flow":
        raise ValueError("Hypothesis audit requires a drift_flow checkpoint")
    if denoiser.transport_parameterization != "endpoint_normalized":
        raise ValueError(
            "Hypothesis audit requires endpoint_normalized transport"
        )

    audit_cfg = cfg.hypothesis_audit
    num_batches = int(audit_cfg.num_batches)
    particles = int(audit_cfg.particles)
    progress_every = int(audit_cfg.get("progress_every", 8))
    if num_batches < 3 or particles < 1 or progress_every < 1:
        raise ValueError(
            "num_batches must be at least three; particles and progress_every "
            "must be positive"
        )

    predictors = _load_pose_predictors(cfg, device)
    dataloader = get_pushT_validation_loader(cfg)
    generator = torch.Generator(device=device).manual_seed(int(audit_cfg.seed))
    records = []

    for batch_index, batch in enumerate(dataloader):
        if batch_index >= num_batches:
            break
        if batch_index % progress_every == 0 or batch_index + 1 == num_batches:
            log.info(
                f"[hypothesis-audit] batch={batch_index + 1}/{num_batches}"
            )

        obs = batch["image"].to(device)
        normalized = bool(cfg.data.normalize_img)
        if normalized:
            obs = (obs - 0.5) / 0.5
        actions_all = batch["action"].to(device)
        if obs.shape[0] != 1:
            raise ValueError("Hypothesis audit requires validation.batch_size=1")

        history_count = denoiser.num_history_frames
        future_count = denoiser.num_future_frames
        current_index = history_count - 1
        target = obs[
            :, current_index + 1:current_index + 1 + future_count
        ].permute(0, 2, 1, 3, 4)
        history = obs[:, :history_count].permute(0, 2, 1, 3, 4)
        actions = actions_all[:, current_index:current_index + future_count]

        target = target.expand(particles, -1, -1, -1, -1)
        history = history.expand(particles, -1, -1, -1, -1)
        actions = actions.expand(particles, -1, -1)
        noise = torch.randn(
            target.shape,
            device=device,
            dtype=target.dtype,
            generator=generator,
        )

        record = {"example": batch_index}
        current = history[:, :, -1]
        target_last = target[:, :, -1]
        _record_mean(
            record,
            "difficulty/motion_pixel",
            _mse_per_sample(
                target,
                current[:, :, None].expand_as(target),
            ),
        )
        if actions.shape[1] > 1:
            action_path = torch.linalg.vector_norm(
                actions[:, 1:].float() - actions[:, :-1].float(), dim=2
            ).sum(dim=1)
        else:
            action_path = torch.linalg.vector_norm(actions.float(), dim=2).mean(dim=1)
        _record_mean(record, "difficulty/action_path", action_path)

        current_pose = _predict_pose(
            _images_01(current, normalized), predictors
        )
        target_pose = _predict_pose(
            _images_01(target_last, normalized), predictors
        )
        _record_mean(
            record,
            "difficulty/motion_vertex",
            _vertex_error_per_sample(current_pose, target_pose),
        )

        outputs = {}
        risks = {}
        pose_risks = {}
        for nfe, grid in GRIDS.items():
            state = noise.clone()
            relative_penalties = []
            source_shifts = []
            for step_index, (source, target_time) in enumerate(
                zip(grid[:-1], grid[1:]), start=1
            ):
                clean_source = (1.0 - source) * noise + source * target
                clean_target = (
                    (1.0 - target_time) * noise + target_time * target
                )
                clean_prediction = _step(
                    denoiser,
                    clean_source,
                    history,
                    actions,
                    source,
                    target_time,
                )
                clean_error = _mse_per_sample(
                    clean_prediction, clean_target
                )
                source_shift = _mse_per_sample(state, clean_source)
                free_prediction = _step(
                    denoiser,
                    state,
                    history,
                    actions,
                    source,
                    target_time,
                )
                free_error = _mse_per_sample(free_prediction, clean_target)
                penalty = free_error - clean_error
                relative_penalty = penalty / clean_error.clamp_min(1e-12)
                prefix = f"local/nfe{nfe}/step{step_index}"
                _record_mean(record, f"{prefix}/clean_mse", clean_error)
                _record_mean(record, f"{prefix}/free_mse", free_error)
                _record_mean(record, f"{prefix}/source_shift_mse", source_shift)
                _record_mean(record, f"{prefix}/relative_penalty", relative_penalty)
                relative_penalties.append(relative_penalty)
                source_shifts.append(source_shift)
                state = free_prediction

            if len(relative_penalties) > 1:
                later_penalties = torch.stack(relative_penalties[1:], dim=0)
                later_shifts = torch.stack(source_shifts[1:], dim=0)
            else:
                later_penalties = torch.stack(relative_penalties, dim=0)
                later_shifts = torch.stack(source_shifts, dim=0)
            _record_mean(
                record,
                f"off_manifold/nfe{nfe}/later_relative_penalty",
                later_penalties.mean(dim=0),
            )
            _record_mean(
                record,
                f"off_manifold/nfe{nfe}/later_source_shift_mse",
                later_shifts.mean(dim=0),
            )

            outputs[nfe] = state
            risks[nfe] = _mse_per_sample(state, target)
            generated_pose = _predict_pose(
                _images_01(state[:, :, -1], normalized), predictors
            )
            pose_risks[nfe] = _vertex_error_per_sample(
                generated_pose, target_pose
            )
            _record_mean(record, f"risk/nfe{nfe}/mse", risks[nfe])
            _record_mean(
                record, f"risk/nfe{nfe}/pose_vertex", pose_risks[nfe]
            )
            _record_mean(
                record,
                f"route/nfe{nfe}_vs_nfe1/defect_mse",
                _mse_per_sample(state, outputs[1]),
            )

        for shallow, deep in TRANSITIONS:
            defect = _mse_per_sample(outputs[shallow], outputs[deep])
            mse_degradation = risks[deep] - risks[shallow]
            pose_degradation = pose_risks[deep] - pose_risks[shallow]
            _record_mean(
                record,
                f"route/{shallow}_to_{deep}/defect_mse",
                defect,
            )
            _record_mean(
                record,
                f"degradation/{shallow}_to_{deep}/mse",
                mse_degradation,
            )
            _record_mean(
                record,
                f"degradation/{shallow}_to_{deep}/pose_vertex",
                pose_degradation,
            )
        records.append(record)

    if len(records) < 3:
        raise RuntimeError(
            f"Hypothesis audit received only {len(records)} validation batches"
        )

    metrics, correlations, strata = _summarize_records(records)
    summary = {
        "status": "complete",
        "checkpoint": os.path.abspath(str(cfg.eval.checkpoint)),
        "checkpoint_step": checkpoint_step,
        "family": str(audit_cfg.family),
        "seed": int(cfg.train.seed),
        "num_batches": len(records),
        "particles": particles,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "metrics": metrics,
        "correlations": correlations,
        "motion_strata": strata,
        "records": records,
    }
    output_path = os.path.abspath(str(audit_cfg.output))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
    return summary
