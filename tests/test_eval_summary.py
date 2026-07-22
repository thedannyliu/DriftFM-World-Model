import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).parents[1] / "driftworld"))

from eval.eval_on_many_videos import _block_pose_errors, _summarize_state, _warm_up_rollout


class _NoiseRollout:
    def sample_autoregressive(self, cur_state, actions, nfe):
        return torch.randn_like(cur_state)


def test_summary_keeps_paired_per_video_metrics():
    state = {
        "mse": [0.1, 0.2],
        "ssim": [0.8, 0.9],
        "psnr": [20.0, 21.0],
        "lpips": [0.3, 0.2],
        "total_gen_time": 2.0,
        "total_gen_frames": 8,
    }

    summary = _summarize_state(state, video_len=64, nfe=4)

    assert summary["num_videos"] == 2
    assert summary["mse"] == pytest.approx(0.15)
    assert summary["seconds_per_frame"] == pytest.approx(0.25)
    assert summary["per_video"] == {
        "mse": [0.1, 0.2],
        "ssim": [0.8, 0.9],
        "psnr": [20.0, 21.0],
        "lpips": [0.3, 0.2],
    }


def test_warmup_preserves_noise_stream():
    obs = torch.zeros(1, 2, 3, 2, 2)
    action = torch.zeros(1, 1, 2)
    torch.manual_seed(7)
    expected = torch.randn_like(obs)

    torch.manual_seed(7)
    _warm_up_rollout(_NoiseRollout(), obs, action, n_history=2, nfe=1)

    torch.testing.assert_close(torch.randn_like(obs), expected, rtol=0, atol=0)


def test_block_pose_error_uses_circular_angle_distance():
    gen_pose = torch.tensor([3.0, 4.0, torch.pi - 0.1])
    gt_pose = torch.tensor([0.0, 0.0, -torch.pi + 0.1])

    errors = _block_pose_errors(gen_pose, gt_pose)

    assert errors["final_block_xy_l2"] == pytest.approx(5.0)
    assert errors["final_block_angle_abs_rad"] == pytest.approx(0.2)
    assert errors["final_block_vertex_error"] > 0
