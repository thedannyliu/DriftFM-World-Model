#!/usr/bin/env python3
"""Print a compact shared-filesystem report for the four weekend queues."""

import argparse
import json
import os
from pathlib import Path

from status_long_research import (
    checkpoint_progress,
    eval_marker,
    load_json,
    number,
    training_run_id,
    triplet,
    wandb_id,
)


NODES = {
    "node-a": {
        "question": "grid-source-interaction-surface",
        "prefix": "wknd-a-",
        "selection": True,
    },
    "node-b": {
        "question": "dyadic-grid-depth",
        "prefix": "wknd-b-",
        "selection": True,
    },
    "node-c": {
        "question": "composed-source-depth",
        "prefix": "wknd-c-",
        "selection": True,
    },
    "node-d": {
        "question": "three-seed-causal-confirmation",
        "prefix": "wknd-d-",
        "selection": False,
    },
}
MILESTONES = (10000, 30000, 60000, 100000, 150000, 200000, 250000, 300000, 400000)


def tags_for_prefix(experiment_root, prefix):
    tags = set()
    for path in experiment_root.glob(f"{prefix}*_seed*"):
        tag, separator, seed = path.name.rpartition("_seed")
        if separator and seed.isdigit():
            tags.add(tag)
    return sorted(tags)


def milestone_marker(experiment_root, tag, seed, step):
    return eval_marker(experiment_root, tag, seed, "latest", f"step{step}")


def latest_milestone(experiment_root, tag, seed):
    completed = [
        step for step in MILESTONES
        if load_json(milestone_marker(experiment_root, tag, seed, step))
    ]
    return completed[-1] if completed else None


def seed_line(experiment_root, tag, seed):
    output_dir = experiment_root / f"{tag}_seed{seed}"
    progress = checkpoint_progress(output_dir)
    step = latest_milestone(experiment_root, tag, seed)
    result = (
        load_json(milestone_marker(experiment_root, tag, seed, step))
        if step else None
    )
    rollout_best = load_json(output_dir / "rollout-best.json")
    updates = (
        progress.get("updates")
        if progress and "updates" in progress
        else 0
    )
    error = progress.get("error") if progress and "error" in progress else None
    return (
        f"  RUN tag={tag} seed={seed} updates={updates} "
        f"eval_step={step or 'NA'} "
        f"lpips_1/2/4/8={triplet(result, 'lpips', (1, 2, 4, 8))} "
        f"vertex_1/2/4/8="
        f"{triplet(result, 'final_block_vertex_error', (1, 2, 4, 8))} "
        f"rollout_best={rollout_best.get('step', 'NA') if rollout_best else 'NA'}"
        f"@{number(rollout_best.get('score')) if rollout_best else 'NA'} "
        f"train_wandb={training_run_id(output_dir) or 'NA'}"
        + (f" checkpoint_error={error}" if error else "")
    )


def locked_line(experiment_root, tag, seed, target):
    label = f"weekend-locked100-step{target}"
    latest = load_json(eval_marker(experiment_root, tag, seed, "latest", label))
    best = load_json(eval_marker(experiment_root, tag, seed, "best", label))
    if not latest and not best:
        return None
    return (
        f"  LOCKED tag={tag} seed={seed} target={target} "
        f"latest_lpips={triplet(latest, 'lpips', (1, 2, 4, 8))} "
        f"best_lpips={triplet(best, 'lpips', (1, 2, 4, 8))} "
        f"eval_wandb={wandb_id(latest)}/{wandb_id(best)}"
    )


def report_node(role, experiment_root):
    spec = NODES[role]
    tags = tags_for_prefix(experiment_root, spec["prefix"])
    print(f"NODE role={role} question={spec['question']} active_tags={len(tags)}")
    if not tags:
        print(f"NODE_STATUS role={role} state=pending")
        return "pending"

    selected = set()
    if spec["selection"]:
        selection_path = experiment_root / f"weekend-selection-{role}-step30000.json"
        selection = load_json(selection_path)
        if selection:
            ranked = selection["candidates"][:2]
            selected = {item["name"] for item in ranked}
            print(
                "  SELECTION winner="
                f"{ranked[0]['name']}@{number(ranked[0]['score'])} "
                f"runner_up={ranked[1]['name']}@{number(ranked[1]['score'])}"
            )
        else:
            done = sum(
                latest_milestone(experiment_root, tag, 1) == 30000
                for tag in tags
            )
            print(f"  SELECTION waiting screens_done={done}/{len(tags)}")

    for tag in tags:
        seed_dirs = sorted(experiment_root.glob(f"{tag}_seed*"))
        for output_dir in seed_dirs:
            seed = int(output_dir.name.rpartition("_seed")[2])
            if selected and tag not in selected and seed > 1:
                continue
            print(seed_line(experiment_root, tag, seed))
            for target in (200000, 300000, 400000):
                locked = locked_line(experiment_root, tag, seed, target)
                if locked:
                    print(locked)

    state = "selected_training" if selected else "screening_or_training"
    print(f"NODE_STATUS role={role} state={state}")
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scope", choices=(*NODES, "all"), nargs="?", default="all")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(os.environ.get(
            "DRIFTFLOWWORLD_ASSET_ROOT",
            "/group-volume/danny-dataset/driftworld",
        )),
    )
    args = parser.parse_args()
    experiment_root = args.asset_root / "checkpoints" / "experiments"
    roles = tuple(NODES) if args.scope == "all" else (args.scope,)
    print(
        f"weekend_research_report scope={args.scope} "
        f"shared_root={experiment_root} "
        f"milestones={','.join(map(str, MILESTONES))}"
    )
    states = [report_node(role, experiment_root) for role in roles]
    print(f"OVERALL node_states={','.join(states)}")


if __name__ == "__main__":
    main()
