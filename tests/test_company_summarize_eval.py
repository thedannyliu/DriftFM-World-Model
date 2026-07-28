import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).parents[1]))

from company.summarize_eval import paired_analysis


def test_action_routing_interleaves_ordered_examples(tmp_path):
    actions = [0.0, 10.0, 2.0, 1.0, 4.0, 3.0]
    paths = {}
    for nfe, mse in (
        (1, [9.0, 10.0, 9.0, 20.0, 9.0, 30.0]),
        (2, [9.0, 1.0, 9.0, 40.0, 9.0, 3.0]),
    ):
        path = tmp_path / f"nfe-{nfe}.json"
        path.write_text(json.dumps({
            "per_video": {
                "action_step_mean": actions,
                "mse": mse,
            }
        }))
        paths[nfe] = path

    routing = paired_analysis(paths)["action_routing_nfe1_or_2"]

    assert routing["split"] == "even_index_dev_odd_index_test"
    assert routing["dev_examples"] == 3
    assert routing["test_examples"] == 3
    assert routing["action_path_threshold"] == pytest.approx(2.0)
    assert routing["nfe2_fraction"] == pytest.approx(2 / 3)
    assert routing["metrics"]["mse"]["adaptive"] == pytest.approx(8.0)
