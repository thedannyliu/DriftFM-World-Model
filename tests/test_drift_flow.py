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


def make_denoiser(objective, **kwargs):
    return Denoiser(
        unet_name="TinyWorldModel",
        temp_list=(0.02, 0.05, 0.2),
        n_neg=4,
        num_future_frames=2,
        num_history_frames=2,
        decay=0.999,
        objective=objective,
        **kwargs,
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


def test_endpoint_normalization_makes_oracle_endpoint_nfe_invariant():
    model = make_denoiser(
        "drift_flow", transport_parameterization="endpoint_normalized"
    )
    history = torch.randn(2, 3, 2, 4, 4)
    actions = torch.randn(2, 2, 2)
    noise = torch.randn(2, 3, 2, 4, 4)

    def zero_endpoint(x, history, actions, time_pair=None):
        return torch.zeros_like(x)

    model.ema_model.forward = zero_endpoint
    one_step = model.sample(history, actions, nfe=1, noise=noise)
    two_step = model.sample(history, actions, nfe=2, noise=noise)
    four_step = model.sample(history, actions, nfe=4, noise=noise)

    torch.testing.assert_close(one_step, two_step, rtol=0, atol=0)
    torch.testing.assert_close(one_step, four_step, rtol=0, atol=0)


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


def test_grid_replay_samples_only_inference_intervals():
    torch.manual_seed(4)
    model = make_denoiser(
        "drift_flow",
        endpoint_replay_probability=0.0,
        grid_replay_probability=1.0,
    )
    source, target, delta, endpoint, grid = model._sample_time_pairs(
        256, torch.device("cpu")
    )
    observed = set(zip(source.tolist(), delta.tolist()))
    expected = {
        (0.0, 0.5),
        (0.5, 0.5),
        (0.0, 0.25),
        (0.25, 0.25),
        (0.5, 0.25),
        (0.75, 0.25),
    }
    assert observed == expected
    assert not endpoint.any()
    assert grid.all()
    torch.testing.assert_close(target, source + delta)


def test_intermediate_uses_all_positive_particles_but_endpoint_uses_one():
    batch = {
        "image": torch.randn(2, 4, 3, 4, 4),
        "action": torch.randn(2, 4, 2),
    }
    for endpoint_probability, expected_particles in ((0.0, 4), (1.0, 1)):
        model = make_denoiser(
            "drift_flow",
            endpoint_replay_probability=endpoint_probability,
            positive_particles=4,
        )
        observed = []
        original = model.drifting_loss

        def record_particles(gen, pos):
            observed.append(pos.shape[1])
            return original(gen, pos)

        model.drifting_loss = record_particles
        loss, metrics = model(batch, torch.device("cpu"))
        assert torch.isfinite(loss)
        assert set(observed) == {expected_particles}
        assert metrics["time/positive_particles"] == 4


def test_mixed_batch_keeps_endpoint_and_intermediate_positive_counts_separate():
    model = make_denoiser("drift_flow", positive_particles=4)
    gen = torch.randn(2, 4, 3, 2, 2, requires_grad=True)
    pos = torch.randn(2, 4, 3, 2, 2)
    observed = []

    def record_groups(group_gen, group_pos):
        observed.append((group_gen.shape[0], group_pos.shape[1]))
        return group_gen.mean(), {"scale": float(group_pos.shape[1])}

    model.drifting_loss = record_groups
    loss, _ = model._mixed_positive_drift_loss(
        gen, pos, torch.tensor([True, False])
    )
    loss.backward()
    assert observed == [(1, 1), (1, 4)]
    assert gen.grad is not None


def test_composed_source_replay_uses_ema_and_keeps_training_gradient_finite():
    torch.manual_seed(5)
    model = make_denoiser(
        "drift_flow",
        endpoint_replay_probability=0.0,
        composed_source_replay_probability=1.0,
    )
    batch = {
        "image": torch.randn(2, 4, 3, 4, 4),
        "action": torch.randn(2, 4, 2),
    }
    calls = []
    original = model.ema_model.forward

    def record_ema_call(*args, **kwargs):
        calls.append(kwargs["time_pair"].detach().clone())
        return original(*args, **kwargs)

    model.ema_model.forward = record_ema_call
    loss, metrics = model(batch, torch.device("cpu"))
    loss.backward()

    assert len(calls) == 2
    assert metrics["time/composed_source_fraction"] == 1.0
    assert torch.isfinite(loss)
    assert model.inner_model.time_embed.weight.grad is not None
    assert torch.isfinite(model.inner_model.time_embed.weight.grad).all()
