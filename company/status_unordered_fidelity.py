#!/usr/bin/env python3
"""Print completion and compact metrics for the unordered-fidelity queues."""

import argparse
import csv
import json
import os
from pathlib import Path


def load_plan(path, scope):
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    if scope == "all":
        return rows
    return [row for row in rows if row[0] == scope]


def load_marker(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def compact(value):
    return "NA" if value is None else f"{value:.6g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scope",
        choices=("all", "node-a", "node-b", "node-c", "node-d"),
        nargs="?",
        default="all",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(os.environ.get(
            "DRIFTFLOWWORLD_ASSET_ROOT",
            "/group-volume/danny-dataset/driftworld",
        )),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    rows = load_plan(
        repo_root / "company" / "unordered_fidelity_plan.tsv",
        args.scope,
    )
    experiment_root = args.asset_root / "checkpoints" / "experiments"
    completed = 0
    fixed_depth_hashes = {}
    for row in rows:
        (
            role,
            name,
            tag,
            seed,
            checkpoint,
            family,
            policy,
            strategy,
            proposals,
            nfe,
            refine_nfe,
            refine_ratio,
            question,
        ) = row
        marker_path = experiment_root / f"gpc-unordered-{name}.json"
        marker = load_marker(marker_path)
        if not marker or marker.get("status") != "complete":
            print(
                f"WAIT queue={role} name={name} family={family} "
                f"policy={policy} marker={marker_path}"
            )
            continue

        completed += 1
        task = marker["task"]
        latency = marker["latency"]
        planning = marker["planning"]
        diagnostics = marker["candidate_diagnostics"]
        if role == "node-a":
            fixed_depth_hashes.setdefault(family, []).append(
                diagnostics.get("policy_action_hashes")
            )
        interval = task["mean_iou_ci95"]
        print(
            f"DONE queue={role} name={name} family={family} policy={policy} "
            f"strategy={strategy} budget={planning['nominal_model_evaluations']} "
            f"trials={task['num_trials']} mean_iou={compact(task['mean_iou'])} "
            f"ci95={compact(interval[0])},{compact(interval[1])} "
            f"success={compact(task['success_fraction'])} "
            f"latency_mean={compact(latency['mean_planning_seconds'])} "
            f"winner_flip={compact(diagnostics['coarse_to_final_winner_flip_fraction'])} "
            f"rank_gt={compact(diagnostics['mean_rank_correlation_with_ground_truth'])} "
            f"selection_regret={compact(diagnostics['mean_ground_truth_selection_regret'])} "
            f"wandb={marker.get('wandb_run_id', 'NA')} question={question}"
        )

    for family, hashes in sorted(fixed_depth_hashes.items()):
        if len(hashes) == 4:
            status = "ready" if all(value == hashes[0] for value in hashes) else "mismatch"
            print(
                f"PAIRING family={family} fixed_depth_candidate_hashes={status}"
            )

    print(
        f"unordered_fidelity_status scope={args.scope} "
        f"completed={completed}/{len(rows)}"
    )


if __name__ == "__main__":
    main()
