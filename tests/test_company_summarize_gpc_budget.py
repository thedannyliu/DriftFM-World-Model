import importlib.util
import json
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "company"
    / "summarize_gpc_budget.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_gpc_budget", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_shard(root, start, end, scores, decisions):
    shard = root / f"num_trial_4_seeds_{start}_{end}"
    shard.mkdir(parents=True)
    np.save(
        shard
        / (
            "final_corrected_sampling_based_testing_no_simulation_planning_"
            f"receding_result_from_index_f{start}.npy"
        ),
        np.asarray([scores]),
    )
    np.save(
        shard / f"planning_seconds_from_index_f{start}.npy",
        np.asarray([0.1, 0.2]),
    )
    (
        shard / f"first_decision_candidates_from_index_f{start}.json"
    ).write_text(json.dumps(decisions))


def test_load_shards_and_ground_truth_diagnostics(tmp_path):
    decisions = [{
        "policy_actions_sha256": "paired-candidates",
        "final_candidate_scores": [0.1, 0.4, 0.7, 1.0],
        "coarse_candidate_scores": [0.1, 0.8, 0.2, 1.0],
        "ground_truth_candidate_rewards": [1.0, 0.6, 0.4, 0.0],
    }]
    write_shard(tmp_path, 0, 2, [0.8, 0.9], decisions)
    write_shard(tmp_path, 2, 4, [0.7, 1.0], decisions)

    scores, timings, loaded_decisions = MODULE.load_shards(tmp_path)
    summary = MODULE.decision_summary(loaded_decisions)

    assert scores.tolist() == [0.8, 0.9, 0.7, 1.0]
    assert timings.tolist() == [0.1, 0.2, 0.1, 0.2]
    assert summary["num_ground_truth_candidate_sets"] == 2
    assert summary["mean_rank_correlation_with_ground_truth"] == 1.0
    assert summary["mean_ground_truth_selection_regret"] == 0.0
    assert summary["oracle_selection_fraction"] == 1.0
    assert summary["coarse_to_final_winner_flip_fraction"] == 0.0
    assert summary["policy_action_hashes"] == [
        "paired-candidates",
        "paired-candidates",
    ]


def test_constant_ground_truth_does_not_emit_nan_correlation():
    summary = MODULE.decision_summary([{
        "policy_actions_sha256": "constant-ground-truth",
        "final_candidate_scores": [0.1, 0.2],
        "coarse_candidate_scores": None,
        "ground_truth_candidate_rewards": [0.0, 0.0],
    }])

    assert summary["num_ground_truth_candidate_sets"] == 1
    assert summary["mean_rank_correlation_with_ground_truth"] is None
    assert summary["mean_ground_truth_selection_regret"] == 0.0
