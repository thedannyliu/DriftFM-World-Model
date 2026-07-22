import sys
from pathlib import Path

import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).parents[1] / "driftworld"))

from drifting_denoiser_multi import Denoiser
from unet_multi.unet_configs import UNet_model_dict


class TinyWorldModel(nn.Module):
    def __init__(self, time_conditioning):
        super().__init__()
        self.proj = nn.Conv3d(3, 3, kernel_size=1)
        self.time_conditioning = time_conditioning
        if time_conditioning:
            self.time_embed = nn.Linear(2, 3)
            nn.init.zeros_(self.time_embed.weight)
            nn.init.zeros_(self.time_embed.bias)

    def forward(self, x, history, actions, time_pair=None):
        output = self.proj(x)
        if self.time_conditioning:
            if time_pair is None:
                time_pair = x.new_tensor((0.0, 1.0)).expand(x.shape[0], 2)
            output = output + self.time_embed(time_pair)[:, :, None, None, None]
        return output


def make_denoiser(objective):
    return Denoiser(
        unet_name="TinyWorldModel",
        temp_list=(0.02, 0.05, 0.2),
        n_neg=4,
        num_future_frames=2,
        num_history_frames=2,
        decay=0.999,
        objective=objective,
    )


def copy_endpoint_weights(source, target):
    incompatible = target.load_state_dict(source.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all("time_embed." in key for key in incompatible.missing_keys)


def setup_module():
    UNet_model_dict["TinyWorldModel"] = lambda num_history, time_conditioning: TinyWorldModel(
        time_conditioning
    )


def teardown_module():
    del UNet_model_dict["TinyWorldModel"]


def test_zero_initialized_nfe_one_preserves_endpoint():
    torch.manual_seed(1)
    baseline = make_denoiser("driftworld")
    drift_flow = make_denoiser("drift_flow")
    copy_endpoint_weights(baseline, drift_flow)

    history = torch.randn(2, 3, 2, 4, 4)
    actions = torch.randn(2, 2, 2)
    noise = torch.randn(2, 3, 2, 4, 4)

    expected = baseline.sample(history, actions, noise=noise)
    actual = drift_flow.sample(history, actions, nfe=1, noise=noise)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_variable_nfe_shapes_and_determinism():
    torch.manual_seed(2)
    model = make_denoiser("drift_flow")
    history = torch.randn(2, 3, 2, 4, 4)
    actions = torch.randn(2, 2, 2)
    noise = torch.randn(2, 3, 2, 4, 4)

    first = model.sample(history, actions, nfe=4, noise=noise)
    second = model.sample(history, actions, nfe=4, noise=noise)
    assert first.shape == (2, 2, 3, 4, 4)
    torch.testing.assert_close(first, second)


def test_arbitrary_time_training_has_finite_gradient():
    torch.manual_seed(3)
    model = make_denoiser("drift_flow")
    batch = {
        "image": torch.randn(2, 4, 3, 4, 4),
        "action": torch.randn(2, 4, 2),
    }

    loss, metrics = model(batch, torch.device("cpu"))
    loss.backward()

    assert torch.isfinite(loss)
    assert 0.0 <= metrics["time/endpoint_fraction"] <= 1.0
    assert model.inner_model.time_embed.weight.grad is not None
    assert torch.isfinite(model.inner_model.time_embed.weight.grad).all()
