#!/usr/bin/env python3
"""Print compact status and frozen gates for the two-node Push-T study."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


ALIASES = {
    "k1-grid25": "k1",
    "k32": "k32",
    "joint-k16": "joint",
    "deep-base-k16": "deep",
}


def compact(value):
    return "NA" if value is None else f"{value:.6g}"


def load_marker(marker_root, name):
    path = marker_root / f"gpc-unordered-{name}.json"
    try:
        marker = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return path, None
    if marker.get("status") != "complete":
        return path, None
    return path, marker


def paired_interval(values, samples=20000, seed=0):
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return np.quantile(means, (0.025, 0.975))


def n0_analysis(marker_root, rows):
    blocks = []
    for row in rows:
        _, name, _, _, _, family, policy, *_ = row
        _, new = load_marker(marker_root, name)
        comparator_name = (
            f"a-confirm80-{ALIASES[family]}-{policy}-nfe2"
        )
        _, old = load_marker(marker_root, comparator_name)
        if new is None or old is None:
            continue

        new_scores = np.asarray(new["task"]["per_trial_iou"], dtype=np.float64)
        old_scores = np.asarray(old["task"]["per_trial_iou"], dtype=np.float64)
        if new_scores.shape != old_scores.shape:
            print(
                f"N0_INVALID family={family} policy={policy} "
                f"reason=score_shape new={new_scores.size} old={old_scores.size}"
            )
            continue
        delta = new_scores - old_scores
        interval = paired_interval(delta)
        new_prefixes = [
            value.get("32") if value else None
            for value in new["candidate_diagnostics"].get(
                "policy_action_prefix_hashes", []
            )
        ]
        old_hashes = old["candidate_diagnostics"].get(
            "policy_action_hashes", []
        )
        pairing = (
            "ready"
            if len(new_prefixes) == len(old_hashes) == new_scores.size
            and new_prefixes == old_hashes
            else "mismatch"
        )
        switch_gain = float(
            np.maximum(new_scores, old_scores).mean()
            - max(new_scores.mean(), old_scores.mean())
        )
        print(
            f"N0_PAIR family={family} policy={policy} pairing={pairing} "
            f"delta_iou={compact(delta.mean())} "
            f"ci95={compact(interval[0])},{compact(interval[1])} "
            f"switch_gain={compact(switch_gain)}"
        )
        blocks.append((new_scores, old_scores, pairing))

    if len(blocks) != 8:
        print(f"N0_GATE status=waiting blocks={len(blocks)}/8")
        return

    generator = np.random.default_rng(0)
    bootstrap = []
    for _ in range(20000):
        block_indices = generator.integers(0, len(blocks), len(blocks))
        gains = []
        for block_index in block_indices:
            new_scores, old_scores, _ = blocks[block_index]
            trial_indices = generator.integers(
                0, len(new_scores), len(new_scores)
            )
            new_sample = new_scores[trial_indices]
            old_sample = old_scores[trial_indices]
            gains.append(
                np.maximum(new_sample, old_sample).mean()
                - max(new_sample.mean(), old_sample.mean())
            )
        bootstrap.append(np.mean(gains))
    observed = np.mean([
        np.maximum(new, old).mean() - max(new.mean(), old.mean())
        for new, old, _ in blocks
    ])
    interval = np.quantile(bootstrap, (0.025, 0.975))
    pairing_ready = all(pairing == "ready" for _, _, pairing in blocks)
    decision = (
        "continue"
        if pairing_ready and interval[1] > 0.01
        else "stop"
        if pairing_ready
        else "invalid"
    )
    print(
        f"N0_GATE status={decision} pairing={str(pairing_ready).lower()} "
        f"mean_switch_gain={compact(observed)} "
        f"ci95={compact(interval[0])},{compact(interval[1])} "
        f"threshold=0.01"
    )


def record_map(marker):
    return {
        (record["test_index"], record["step_index"]): record
        for record in marker.get("candidate_audit_records", [])
    }


def n1_analysis(marker_root, rows):
    by_policy = {}
    for row in rows:
        _, name, _, _, _, _, policy, _, _, nfe, *_ = row
        _, marker = load_marker(marker_root, name)
        if marker is not None:
            by_policy.setdefault(policy, {})[int(nfe)] = marker

    for policy in ("ep100", "ep300"):
        pair = by_policy.get(policy, {})
        if set(pair) != {1, 4}:
            print(
                f"N1_SMOKE policy={policy} status=waiting "
                f"depths={sorted(pair)}"
            )
            continue
        shallow = record_map(pair[1])
        deep = record_map(pair[4])
        common = sorted(set(shallow) & set(deep))
        hashes_ready = (
            len(common) == len(shallow) == len(deep) == 64
            and all(
                shallow[key].get("environment_state_sha256")
                == deep[key].get("environment_state_sha256")
                and shallow[key].get("policy_actions_sha256")
                == deep[key].get("policy_actions_sha256")
                for key in common
            )
        )
        finite = True
        repeat_error = 0.0
        nondegenerate = 0
        expand_wins = 0
        refine_wins = 0
        for key in common:
            shallow_record = shallow[key]
            deep_record = deep[key]
            utility = np.asarray(
                shallow_record["ground_truth_candidate_rewards"],
                dtype=np.float64,
            )
            utility_repeat = np.asarray(
                shallow_record["ground_truth_candidate_rewards_repeat"],
                dtype=np.float64,
            )
            deep_utility = np.asarray(
                deep_record["ground_truth_candidate_rewards"],
                dtype=np.float64,
            )
            deep_repeat = np.asarray(
                deep_record["ground_truth_candidate_rewards_repeat"],
                dtype=np.float64,
            )
            arrays = (utility, utility_repeat, deep_utility, deep_repeat)
            finite = finite and all(
                value.size == 16 and np.isfinite(value).all()
                for value in arrays
            )
            if not all(value.size == 16 for value in arrays):
                continue
            repeat_error = max(
                repeat_error,
                float(np.max(np.abs(utility - utility_repeat))),
                float(np.max(np.abs(deep_utility - deep_repeat))),
                float(np.max(np.abs(utility - deep_utility))),
            )
            nondegenerate += int(np.ptp(utility) > 1e-12)
            expand_wins += int(
                np.max(utility[8:]) > np.max(utility[:8]) + 1e-12
            )
            shallow_scores = np.asarray(
                shallow_record["final_candidate_scores"],
                dtype=np.float64,
            )
            deep_scores = np.asarray(
                deep_record["final_candidate_scores"],
                dtype=np.float64,
            )
            shallow_selected = int(np.argmin(shallow_scores))
            deep_selected = int(np.argmin(deep_scores))
            refine_wins += int(
                utility[deep_selected]
                > utility[shallow_selected] + 1e-12
            )

        denominator = len(common)
        nondegenerate_fraction = (
            nondegenerate / denominator if denominator else 0.0
        )
        expand_fraction = expand_wins / denominator if denominator else 0.0
        refine_fraction = refine_wins / denominator if denominator else 0.0
        passed = (
            hashes_ready
            and finite
            and repeat_error <= 1e-12
            and nondegenerate_fraction >= 0.90
            and expand_fraction >= 0.10
            and refine_fraction >= 0.10
        )
        print(
            f"N1_SMOKE policy={policy} "
            f"status={'pass' if passed else 'fail'} "
            f"paired={len(common)}/64 hashes={str(hashes_ready).lower()} "
            f"finite={str(finite).lower()} repeat_max={compact(repeat_error)} "
            f"nondegenerate={compact(nondegenerate_fraction)} "
            f"expand={compact(expand_fraction)} "
            f"refine={compact(refine_fraction)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scope",
        choices=("all", "node-a", "node-b"),
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
    with (repo_root / "company" / "decision_fidelity_pusht_plan.tsv").open(
        newline=""
    ) as stream:
        all_rows = list(csv.reader(stream, delimiter="\t"))
    rows = (
        all_rows
        if args.scope == "all"
        else [row for row in all_rows if row[0] == args.scope]
    )
    marker_root = args.asset_root / "checkpoints" / "experiments"

    completed = 0
    for row in rows:
        role, name, _, _, _, family, policy, _, proposals, nfe, *_ = row
        path, marker = load_marker(marker_root, name)
        if marker is None:
            print(
                f"WAIT queue={role} name={name} family={family} "
                f"policy={policy} marker={path}"
            )
            continue
        completed += 1
        print(
            f"DONE queue={role} name={name} family={family} policy={policy} "
            f"proposals={proposals} nfe={nfe} "
            f"trials={marker['task']['num_trials']} "
            f"mean_iou={compact(marker['task']['mean_iou'])} "
            f"success={compact(marker['task']['success_fraction'])} "
            f"latency={compact(marker['latency']['mean_planning_seconds'])} "
            f"wandb={marker.get('wandb_run_id', 'NA')}"
        )

    if args.scope in ("all", "node-a"):
        n0_analysis(
            marker_root,
            [row for row in all_rows if row[0] == "node-a"],
        )
    if args.scope in ("all", "node-b"):
        n1_analysis(
            marker_root,
            [row for row in all_rows if row[0] == "node-b"],
        )
    print(
        f"decision_fidelity_pusht_status scope={args.scope} "
        f"completed={completed}/{len(rows)}"
    )


if __name__ == "__main__":
    main()
