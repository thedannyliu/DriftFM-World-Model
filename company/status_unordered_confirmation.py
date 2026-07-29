#!/usr/bin/env python3
"""Print completion and compact metrics for held-out fixed-candidate confirmation."""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


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
    with (repo_root / "company" / "unordered_confirmation_plan.tsv").open(
        newline=""
    ) as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    if args.scope != "all":
        rows = [row for row in rows if row[0] == args.scope]

    marker_root = args.asset_root / "checkpoints" / "experiments"
    completed = 0
    hashes = defaultdict(list)
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
        marker_path = marker_root / f"gpc-unordered-{name}.json"
        try:
            marker = json.loads(marker_path.read_text())
        except (OSError, json.JSONDecodeError):
            marker = None
        if not marker or marker.get("status") != "complete":
            print(
                f"WAIT queue={role} name={name} family={family} "
                f"policy={policy} nfe={nfe} marker={marker_path}"
            )
            continue

        completed += 1
        task = marker["task"]
        latency = marker["latency"]
        diagnostics = marker["candidate_diagnostics"]
        hashes[(family, policy)].append(
            diagnostics.get("policy_action_hashes")
        )
        interval = task["mean_iou_ci95"]
        print(
            f"DONE queue={role} name={name} family={family} "
            f"policy={policy} nfe={nfe} trials={task['num_trials']} "
            f"mean_iou={compact(task['mean_iou'])} "
            f"ci95={compact(interval[0])},{compact(interval[1])} "
            f"success={compact(task['success_fraction'])} "
            f"latency={compact(latency['mean_planning_seconds'])} "
            f"rank_gt={compact(diagnostics['mean_rank_correlation_with_ground_truth'])} "
            f"regret={compact(diagnostics['mean_ground_truth_selection_regret'])} "
            f"oracle={compact(diagnostics['oracle_selection_fraction'])} "
            f"audit_records={len(marker.get('candidate_audit_records', []))} "
            f"wandb={marker.get('wandb_run_id', 'NA')}"
        )

    for (family, policy), values in sorted(hashes.items()):
        if len(values) == 4:
            status = (
                "ready"
                if all(value == values[0] for value in values)
                else "mismatch"
            )
            print(
                f"PAIRING family={family} policy={policy} "
                f"candidate_hashes={status}"
            )

    print(
        f"unordered_confirmation_status scope={args.scope} "
        f"completed={completed}/{len(rows)}"
    )


if __name__ == "__main__":
    main()
