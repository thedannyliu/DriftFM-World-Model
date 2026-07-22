#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT / "driftworld"))

from drifting_denoiser_multi import Denoiser


def make_model(objective):
    return Denoiser(
        unet_name="UNet_PushT",
        temp_list=(0.02, 0.05, 0.2),
        n_neg=2,
        num_future_frames=4,
        num_history_frames=4,
        decay=0.999,
        objective=objective,
        endpoint_replay_probability=0.0 if objective == "drift_flow" else 0.25,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a Slurm GPU allocation")

    torch.manual_seed(1)
    device = torch.device("cuda")
    baseline = make_model("driftworld").to(device).eval()
    drift_flow = make_model("drift_flow").to(device).eval()
    incompatible = drift_flow.load_state_dict(baseline.state_dict(), strict=False)
    if incompatible.unexpected_keys or not all(
        "time_embed." in key for key in incompatible.missing_keys
    ):
        raise RuntimeError(str(incompatible))

    history = torch.randn(1, 3, 4, 96, 96, device=device)
    actions = torch.randn(1, 4, 2, device=device)
    noise = torch.randn(1, 3, 4, 96, 96, device=device)
    with torch.no_grad():
        endpoint = baseline.sample(history, actions, noise=noise)
        nfe1 = drift_flow.sample(history, actions, nfe=1, noise=noise)
        nfe4 = drift_flow.sample(history, actions, nfe=4, noise=noise)
    max_endpoint_difference = (endpoint - nfe1).abs().max().item()
    if max_endpoint_difference != 0:
        raise RuntimeError(f"Endpoint mismatch: {max_endpoint_difference}")

    drift_flow.train()
    batch = {
        "image": torch.randn(1, 8, 3, 96, 96),
        "action": torch.randn(1, 8, 2),
    }
    loss, metrics = drift_flow(batch, device)
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss")

    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "endpoint_max_abs_difference": max_endpoint_difference,
        "nfe1_shape": list(nfe1.shape),
        "nfe4_shape": list(nfe4.shape),
        "loss": loss.item(),
        "endpoint_fraction": metrics["time/endpoint_fraction"],
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
