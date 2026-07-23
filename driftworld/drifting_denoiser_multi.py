"""
DriftWorld: Drifting denoiser, which contains the forward pass, drifting loss, and sampling code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from collections import defaultdict
import copy
import math
from typing import TYPE_CHECKING, Any

from unet_multi.unet_configs import UNet_model_dict
from drifting.drift_loss_indep import drift_loss

if TYPE_CHECKING:
    from data import Batch
else:
    Batch = Any

log = logging.getLogger(__name__)

class Denoiser(nn.Module):
    """
    Denoiser based on drifting loss.
    It conditions on num_history_frames current+history frames and generates num_future_frames future frames.
    """
    def __init__(self, unet_name: str, # name of U-Net
                 temp_list: tuple, # temperatures for drifting loss
                 n_neg: int, # number of negative samples for drifting loss
                 num_future_frames: int, # number of future frames to predict
                 num_history_frames: int, # number current+history frames to condition on
                 decay: float, # EMA decay
                 objective: str = "driftworld",
                 endpoint_replay_probability: float = 0.25,
                 grid_replay_probability: float = 0.0,
                 positive_particles: int = 1,
                 time_sampling: str = "logit_normal",
                 time_mu: float = -0.4,
                 time_sigma: float = 1.0,
                 ) -> None:
        super().__init__()
        if objective not in {"driftworld", "drift_flow"}:
            raise ValueError(f"Unknown objective: {objective}")
        if endpoint_replay_probability + grid_replay_probability > 1.0:
            raise ValueError("Endpoint and grid replay probabilities must sum to at most one")
        if positive_particles < 1:
            raise ValueError("positive_particles must be at least one")
        self.objective = objective
        self.inner_model = UNet_model_dict[unet_name](
            num_history=num_history_frames,
            time_conditioning=objective == "drift_flow",
        ) # U-Net
        self.temp_list = temp_list
        self.n_neg = n_neg
        self.num_future_frames = num_future_frames
        self.num_history_frames = num_history_frames
        self.decay = decay
        self.endpoint_replay_probability = endpoint_replay_probability
        self.grid_replay_probability = grid_replay_probability
        self.positive_particles = positive_particles
        self.time_sampling = time_sampling
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.ema_model = copy.deepcopy(self.inner_model)
        self.ema_model.requires_grad_(False)

    def drifting_loss(self, gen: torch.Tensor, pos: torch.Tensor):
        """
        Drifting loss: MSE(gen, stopgrad(gen + V)).
        Args:
            gen: Generated samples [B, N, C, H, W]
            pos: Data samples [B, 1, C, H, W]
        Returns:
            scalar drifting loss and dictionary of metrics
        """
        B, N, C, H, W = gen.shape
        gen_flat = gen.reshape(B, N, -1) # [B, N, D], where D=C*H*W
        pos_flat = pos.reshape(B, pos.shape[1], -1) # [B, P, D]

        loss, info = drift_loss(gen=gen_flat, fixed_pos=pos_flat, R_list=self.temp_list)
        return loss.mean(), info

    def _sample_time_pairs(self, batch_size, device):
        if self.time_sampling == "uniform":
            endpoints = torch.rand((batch_size, 2), device=device)
        elif self.time_sampling == "logit_normal":
            endpoints = torch.sigmoid(
                torch.randn((batch_size, 2), device=device) * self.time_sigma + self.time_mu
            )
        else:
            raise ValueError(f"Unknown time sampling: {self.time_sampling}")

        source = endpoints.min(dim=1).values
        target = endpoints.max(dim=1).values
        category = torch.rand(batch_size, device=device)
        endpoint_replay = category < self.endpoint_replay_probability
        grid_replay = (
            (category >= self.endpoint_replay_probability)
            & (category < self.endpoint_replay_probability + self.grid_replay_probability)
        )

        grid_sources = source.new_tensor((0.0, 0.5, 0.0, 0.25, 0.5, 0.75))
        grid_deltas = source.new_tensor((0.5, 0.5, 0.25, 0.25, 0.25, 0.25))
        grid_indices = torch.randint(grid_sources.numel(), (batch_size,), device=device)
        source = torch.where(grid_replay, grid_sources[grid_indices], source)
        target = torch.where(
            grid_replay,
            grid_sources[grid_indices] + grid_deltas[grid_indices],
            target,
        )
        source = torch.where(endpoint_replay, torch.zeros_like(source), source)
        target = torch.where(endpoint_replay, torch.ones_like(target), target)
        delta = (target - source).clamp_min(1e-4)
        return source, target, delta, endpoint_replay, grid_replay

    def _mixed_positive_drift_loss(self, gen, pos, endpoint_replay):
        """Use one endpoint positive but all intermediate-marginal positives."""
        if pos.shape[1] == 1:
            return self.drifting_loss(gen, pos)

        loss = gen.new_zeros(())
        info = defaultdict(float)
        for mask, num_positives in (
            (endpoint_replay, 1),
            (~endpoint_replay, pos.shape[1]),
        ):
            count = int(mask.sum().item())
            if count == 0:
                continue
            group_loss, group_info = self.drifting_loss(
                gen[mask], pos[mask, :num_positives]
            )
            weight = count / gen.shape[0]
            loss = loss + weight * group_loss
            for key, value in group_info.items():
                info[key] += weight * value
        return loss, info

    def forward(self, batch: Batch, device):
        """
        Forward pass to train DriftWorld
        Args:
            batch: a batch of data containing
                - 'image': sequence of visual observations o_0, ..., o_{time_steps-1}
                           shape (b, t, c, h, w)
                - 'action': sequence of actions taken at each timestep a_0, ..., a_{time_steps-1}
                           shape (b, t, 2)
            device: device
        Returns:
            scalar loss and dictionary of metrics
        """
        obs = batch['image']
        act = batch['action'].to(device)

        b, t, c, h, w = obs.size()
        n = self.num_future_frames # num of future frames output by one forward pass
        k = self.num_history_frames # num of current+history frames used as context
        cur_idx = k - 1 # index of the current frame o_t within the window
        assert t == n + k

        target_x = obs[:, cur_idx+1:].permute(0, 2, 1, 3, 4).to(device) # (b, n, c, h, w) -> (b, c, n, h, w)
        history = obs[:, 0 : cur_idx+1].permute(0, 2, 1, 3, 4).to(device) # (b, k, c, h, w) -> (b, c, k, h, w)
        actions = act[:, cur_idx : cur_idx+n] # (b, n, 2)

        history = history.repeat_interleave(self.n_neg, dim=0)
        actions = actions.repeat_interleave(self.n_neg, dim=0)

        # j = self.n_neg
        # gen: (j*b, c, n, h, w) generated samples, i.e. "negative" samples for drifting
        noise = torch.randn((self.n_neg * b, c, n, h, w), device=device)
        if self.objective == "drift_flow":
            source_time, target_time, delta, replay, grid_replay = self._sample_time_pairs(
                b, device
            )
            target_rep = target_x.repeat_interleave(self.n_neg, dim=0)
            source_rep = source_time.repeat_interleave(self.n_neg).view(-1, 1, 1, 1, 1)
            delta_rep = delta.repeat_interleave(self.n_neg).view(-1, 1, 1, 1, 1)
            source_x = (1.0 - source_rep) * noise + source_rep * target_rep
            time_pair = torch.stack((source_time, delta), dim=1).repeat_interleave(
                self.n_neg, dim=0
            )
            endpoint = self.inner_model(source_x, history, actions, time_pair=time_pair)
            gen = source_x + delta_rep * (endpoint - source_x)

            positive_noise = torch.randn(
                (b, self.positive_particles, c, n, h, w), device=device
            )
            target_view = target_time.view(b, 1, 1, 1, 1, 1)
            positive_x = (
                (1.0 - target_view) * positive_noise
                + target_view * target_x.unsqueeze(1)
            )
        else:
            gen = self.inner_model(noise, history, actions)
            positive_x = target_x.unsqueeze(1)
            # raw output from the U-Net. the inputs are
            # noise: (j*b, c, n, h, w) noise for future states s_(T+1), ..., s_(T+n)
            # history: (j*b, c, k, h, w) current+history states s_(T-k+1), ..., s_T
            # actions: (j*b, n, 2) actions a_T, ..., a_(T+n-1)

        # compute drifting loss for every timestep separately
        loss = 0
        metrics = defaultdict(float)
        for i in range(n):
            target_x_slice = positive_x[:, :, :, i] # (b, P, c, h, w)
            gen_slice = gen[:, :, i].reshape((b, self.n_neg, c, h, w)) # (j*b, c, h, w) -> (b, j, c, h, w)
            if self.objective == "drift_flow":
                loss_i, info_i = self._mixed_positive_drift_loss(
                    gen_slice, target_x_slice, replay
                )
            else:
                loss_i, info_i = self.drifting_loss(gen_slice, target_x_slice)
            loss += loss_i
            for key, value in info_i.items():
                metrics[key] += value

        loss /= n
        averages = {key: total / n for key, total in metrics.items()}
        averages['loss_backprop'] = loss.item()
        if self.objective == "drift_flow":
            averages['time/source_mean'] = source_time.mean().item()
            averages['time/target_mean'] = target_time.mean().item()
            averages['time/delta_mean'] = delta.mean().item()
            averages['time/endpoint_fraction'] = replay.float().mean().item()
            averages['time/grid_fraction'] = grid_replay.float().mean().item()
            averages['time/positive_particles'] = self.positive_particles
        return loss, averages

    @torch.no_grad()
    def sample(self, history, actions, nfe=1, noise=None, time_grid=None):
        """
        Sample from DriftWorld (EMA weights, random-noise init).
        Args:
            history: (B, C, K, H, W) tensor of the current+history states o_(t-K+1), ..., o_t
            actions: (B, F, 2) tensor of actions a_t, ..., a_(t+F-1)
        Returns:
            (B, F, C, H, W) tensor of predicted future states o_(t+1), ..., o_(t+F)
        """
        B, C, K, H, W = history.size()
        F = actions.shape[1]
        if nfe < 1:
            raise ValueError("nfe must be at least 1")
        state = noise
        if state is None:
            state = torch.randn((B, C, F, H, W), device=history.device)
        if state.shape != (B, C, F, H, W):
            raise ValueError(f"noise has shape {state.shape}, expected {(B, C, F, H, W)}")

        if self.objective == "driftworld":
            if nfe != 1:
                raise ValueError("The DriftWorld endpoint model only supports nfe=1")
            state = self.ema_model(state, history, actions)
        else:
            if time_grid is None:
                time_grid = torch.linspace(0.0, 1.0, nfe + 1, device=history.device)
            else:
                time_grid = torch.as_tensor(time_grid, device=history.device, dtype=state.dtype)
                if time_grid.numel() != nfe + 1:
                    raise ValueError("time_grid must contain nfe + 1 values")
            for source, target in zip(time_grid[:-1], time_grid[1:]):
                delta = target - source
                if delta <= 0:
                    raise ValueError("time_grid must be strictly increasing")
                time_pair = torch.stack((source, delta)).expand(B, 2)
                endpoint = self.ema_model(state, history, actions, time_pair=time_pair)
                if source == 0 and target == 1:
                    state = endpoint
                else:
                    state = state + delta * (endpoint - state)
        return state.permute(0, 2, 1, 3, 4)

    @torch.no_grad()
    def create_noise_schedule(self, cur_state, actions, generator=None):
        B, K, C, H, W = cur_state.size()
        cur_idx = K - 1
        future = actions.shape[1] - cur_idx
        schedule = []
        for start in range(0, future, self.num_future_frames):
            frames = min(self.num_future_frames, future - start)
            schedule.append(
                torch.randn(
                    (B, C, frames, H, W),
                    device=cur_state.device,
                    generator=generator,
                )
            )
        return schedule

    @torch.no_grad()
    def sample_autoregressive(self, cur_state, actions, nfe=1, noise_schedule=None):
        """
        Sample autoregressively from DriftWorld (EMA weights, random-noise init).
        Args:
            cur_state: (B, K, C, H, W) history frames o_(t-(K-1)), ..., o_t
            actions: (B, T, 2) actions a_(t-(K-1)), ..., a_(t+T-K).
                     Only the future actions a_t, ..., a_(t+F-1) are used
                     (F = T - (K-1)); the K-1 history actions are ignored.
        Returns:
            (B, T+1, C, H, W) = (B, K+F, C, H, W) tensor:
                - the K init history frames o_(t-(K-1)), ..., o_t
                - followed by the F predicted future frames o_(t+1), ..., o_(t+F)
        """
        B, K, C, H, W = cur_state.size()
        assert K == self.num_history_frames
        cur_idx = K - 1 # index of the current frame o_t within cur_state
        n = self.num_future_frames
        F = actions.shape[1] - cur_idx # number of future frames to predict
        num_iter = math.ceil(F / n)
        if noise_schedule is not None and len(noise_schedule) != num_iter:
            raise ValueError(f"noise_schedule has {len(noise_schedule)} chunks, expected {num_iter}")
        log.info(f"Number future frames: F = {F}")
        log.info(f"Number of iterations: num_iter = {num_iter}")

        # autoregressive rollout
        #   i=0: history frames 0:cur_idx+1 => predict cur_idx+1, ..., cur_idx+n
        #   i=1: history frames n:n+cur_idx+1 => predict cur_idx+n+1, ...
        out = torch.zeros((B, cur_idx + 1 + F, C, H, W), device=cur_state.device)
        out[:, :cur_idx+1] = cur_state
        for i in range(num_iter):
            log.info(f"(iter {i}/{num_iter})")
            history_i = out[:, i*n : i*n + cur_idx + 1].permute(0, 2, 1, 3, 4) # (B, C, K, H, W)
            act_i = actions[:, cur_idx + i*n : cur_idx + (i+1)*n] # (B, n, 2)

            noise_i = None if noise_schedule is None else noise_schedule[i]
            gen = self.sample(history_i, act_i, nfe=nfe, noise=noise_i) # (B, n, C, H, W)

            out[:, cur_idx + 1 + i*n : cur_idx + 1 + i*n + gen.shape[1]] = gen
        return out

    @torch.no_grad()
    def update_ema(self):
        """
        Updates the EMA parameters.
        ema_new = decay * ema_old + (1 - decay) * current
        """
        for p_ema, p_net in zip(self.ema_model.parameters(), self.inner_model.parameters()):
            p_ema.lerp_(p_net, 1 - self.decay)
