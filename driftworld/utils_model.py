import logging
from drifting_denoiser_multi import Denoiser

log = logging.getLogger(__name__)

def create_model(cfg, device):
    objective = cfg.model.get("objective", "driftworld")
    drift_flow = cfg.model.get("drift_flow", {})
    log.info(f"Creating world model with objective={objective}")
    return Denoiser(
        unet_name=cfg.model.unet_name,
        temp_list=cfg.model.temp_list,
        n_neg=cfg.model.n_neg,
        num_future_frames=cfg.model.num_future_frames,
        num_history_frames=cfg.model.num_history_frames,
        decay=cfg.train.decay,
        objective=objective,
        endpoint_replay_probability=drift_flow.get("endpoint_replay_probability", 0.25),
        grid_replay_probability=drift_flow.get("grid_replay_probability", 0.0),
        positive_particles=drift_flow.get("positive_particles", 1),
        transport_parameterization=drift_flow.get(
            "transport_parameterization", "residual"
        ),
        composed_source_replay_probability=drift_flow.get(
            "composed_source_replay_probability", 0.0
        ),
        time_sampling=drift_flow.get("time_sampling", "logit_normal"),
        time_mu=drift_flow.get("time_mu", -0.4),
        time_sigma=drift_flow.get("time_sigma", 1.0),
    ).to(device)
