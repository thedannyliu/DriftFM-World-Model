import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / "driftworld"))

from eval.eval_on_many_videos import _summarize_state


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
