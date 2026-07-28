#!/usr/bin/env python3
"""Summarize sharded fixed-budget GPC evaluations and log compact W&B metrics."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def bootstrap_interval(values, samples=10000, seed=0):
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def load_shards(output_dir):
    score_paths = sorted(output_dir.glob(
        "num_trial_*_seeds_*_*/"
        "final_corrected_sampling_based_testing_no_simulation_planning_"
        "receding_result_from_index_f*.npy"
    ))
    if not score_paths:
        raise FileNotFoundError(f"No completed GPC shards under {output_dir}")

    scores = []
    timings = []
    decisions = []
    for score_path in score_paths:
        scores.extend(np.load(score_path).reshape(-1).tolist())
        timing_path = next(score_path.parent.glob("planning_seconds_from_index_f*.npy"))
        timings.extend(np.load(timing_path).reshape(-1).tolist())
        decision_path = next(
            score_path.parent.glob("first_decision_candidates_from_index_f*.json")
        )
        decisions.extend(json.loads(decision_path.read_text()))
    return np.asarray(scores), np.asarray(timings), decisions


def decision_summary(decisions):
    margins = []
    winner_flips = []
    candidate_counts = []
    rank_correlations = []
    selected_rewards = []
    oracle_rewards = []
    selection_regrets = []
    oracle_selections = []
    ground_truth_sets = 0
    policy_action_hashes = []
    for decision in decisions:
        policy_action_hashes.append(decision.get("policy_actions_sha256"))
        final = np.asarray(decision["final_candidate_scores"], dtype=np.float64)
        candidate_counts.append(final.size)
        if final.size >= 2:
            ordered = np.sort(final)
            margins.append(float(ordered[1] - ordered[0]))
        coarse = decision.get("coarse_candidate_scores")
        if coarse is not None:
            coarse = np.asarray(coarse, dtype=np.float64)
            if coarse.size == final.size:
                winner_flips.append(float(np.argmin(coarse) != np.argmin(final)))
        ground_truth = decision.get("ground_truth_candidate_rewards")
        if ground_truth is not None:
            ground_truth = np.asarray(ground_truth, dtype=np.float64)
            if ground_truth.size == final.size:
                ground_truth_sets += 1
                predicted_ranks = np.empty(final.size, dtype=np.float64)
                predicted_ranks[np.argsort(-final, kind="stable")] = np.arange(
                    final.size
                )
                ground_truth_ranks = np.empty(final.size, dtype=np.float64)
                ground_truth_ranks[
                    np.argsort(ground_truth, kind="stable")
                ] = np.arange(final.size)
                if np.std(ground_truth) > 0:
                    rank_correlations.append(float(np.corrcoef(
                        predicted_ranks,
                        ground_truth_ranks,
                    )[0, 1]))
                selected = int(np.argmin(final))
                oracle = int(np.argmax(ground_truth))
                selected_rewards.append(float(ground_truth[selected]))
                oracle_rewards.append(float(ground_truth[oracle]))
                selection_regrets.append(float(
                    ground_truth[oracle] - ground_truth[selected]
                ))
                oracle_selections.append(float(selected == oracle))
    return {
        "num_first_decisions": len(decisions),
        "num_candidates": (
            int(candidate_counts[0])
            if candidate_counts and len(set(candidate_counts)) == 1
            else None
        ),
        "mean_top2_score_margin": (
            float(np.mean(margins)) if margins else None
        ),
        "coarse_to_final_winner_flip_fraction": (
            float(np.mean(winner_flips)) if winner_flips else None
        ),
        "num_ground_truth_candidate_sets": ground_truth_sets,
        "mean_rank_correlation_with_ground_truth": (
            float(np.mean(rank_correlations)) if rank_correlations else None
        ),
        "mean_selected_ground_truth_reward": (
            float(np.mean(selected_rewards)) if selected_rewards else None
        ),
        "mean_oracle_ground_truth_reward": (
            float(np.mean(oracle_rewards)) if oracle_rewards else None
        ),
        "mean_ground_truth_selection_regret": (
            float(np.mean(selection_regrets)) if selection_regrets else None
        ),
        "oracle_selection_fraction": (
            float(np.mean(oracle_selections)) if oracle_selections else None
        ),
        "policy_action_hashes": policy_action_hashes,
    }


def numeric_items(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else key
            yield from numeric_items(child, child_prefix)
    elif isinstance(value, (int, float)) and value is not None:
        yield prefix, value


def write_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--num-proposals", type=int, required=True)
    parser.add_argument("--nfe", type=int, required=True)
    parser.add_argument("--refine-nfe", type=int, required=True)
    parser.add_argument("--refine-ratio", type=float, required=True)
    parser.add_argument("--expected-seeds", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    args = parser.parse_args()

    scores, timings, decisions = load_shards(args.output_dir)
    if scores.size != args.expected_seeds:
        raise ValueError(
            f"Expected {args.expected_seeds} trial scores, found {scores.size}"
        )

    result = {
        "status": "complete",
        "name": args.name,
        "family": args.family,
        "policy": args.policy,
        "planning": {
            "strategy": args.strategy,
            "num_proposals": args.num_proposals,
            "nfe": args.nfe,
            "refine_nfe": args.refine_nfe,
            "refine_ratio": args.refine_ratio,
            "nominal_model_evaluations": (
                args.num_proposals * args.nfe
                if args.strategy != "coarse_to_fine"
                else args.num_proposals + round(
                    args.num_proposals * args.refine_ratio
                ) * args.refine_nfe
            ),
        },
        "task": {
            "num_trials": int(scores.size),
            "mean_iou": float(scores.mean()),
            "median_iou": float(np.median(scores)),
            "mean_iou_ci95": bootstrap_interval(scores),
            "success_fraction": float(np.mean(scores >= 0.95)),
            "per_trial_iou": scores.tolist(),
        },
        "latency": {
            "num_plans": int(timings.size),
            "mean_planning_seconds": (
                float(timings.mean()) if timings.size else None
            ),
            "p95_planning_seconds": (
                float(np.quantile(timings, 0.95)) if timings.size else None
            ),
        },
        "candidate_diagnostics": decision_summary(decisions),
    }

    if args.wandb_project:
        import wandb

        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_name or args.name,
            job_type="fixed-budget-planning",
        )
        wandb.log(dict(numeric_items(result)))
        result["wandb_run_id"] = run.id
        run.finish()

    write_atomic(args.output, result)
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
