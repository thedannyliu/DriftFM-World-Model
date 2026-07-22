import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from summarize_paired_iou import paired_summary


def test_positive_paired_gain_passes_gate():
    baseline = np.linspace(0.4, 0.7, 100)
    method = baseline + 0.02
    summary = paired_summary(baseline, method, bootstrap_samples=1000, seed=1)
    assert summary["primary_gate_passed"]
    assert np.isclose(summary["paired_mean_delta"], 0.02)


def test_mismatched_pairs_are_rejected():
    try:
        paired_summary([0.1], [0.1, 0.2])
    except ValueError as error:
        assert "differ" in str(error)
    else:
        raise AssertionError("Expected a pairing error")
