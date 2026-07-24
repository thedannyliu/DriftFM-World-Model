#!/usr/bin/env python3
"""Report shared progress and rollout results for the four long research queues."""

import argparse
import json
import os
from pathlib import Path

from select_corrected_variant import read_candidate


NODES = {
    "node-a": {
        "question": "time-pair-curriculum-k1",
        "candidates": (
            "driftflow-endpointnorm-k1",
            "driftflow-endpointnorm-k1-noreplay",
            "driftflow-endpointnorm-k1-replay50",
            "driftflow-endpointnorm-k1-grid25",
            "driftflow-endpointnorm-k1-uniform",
        ),
    },
    "node-b": {
        "question": "positive-particle-scaling",
        "candidates": (
            "driftflow-endpointnorm-k16",
            "driftflow-endpointnorm-k2",
            "driftflow-endpointnorm-k4",
            "driftflow-endpointnorm-k8",
            "driftflow-endpointnorm-k32",
        ),
    },
    "node-c": {
        "question": "composed-source-replay",
        "candidates": (
            "driftflow-endpointnorm-k1-sr10",
            "driftflow-endpointnorm-k1-sr25",
            "driftflow-endpointnorm-k16-sr10",
            "driftflow-endpointnorm-k16-sr25",
            "driftflow-endpointnorm-k16-sr50",
            "driftflow-endpointnorm-k16-grid25-sr25",
        ),
    },
    "node-d": {
        "question": "endpoint-warmup-and-optimization",
        "candidates": (
            "driftflow-endpointnorm-k16-warmup1k",
            "driftflow-endpointnorm-k16-warmup3k",
            "driftflow-endpointnorm-k16-lrhalf",
            "driftflow-endpointnorm-k16-lrdouble",
            "driftflow-endpointnorm-k16-grid50",
        ),
    },
}
MILESTONES = (1000, 3000, 10000, 30000, 60000, 100000)
LOCKED_LABEL = "locked100-step100000"


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def number(value):
    return "NA" if value is None else f"{value:.6g}"


def wandb_id(data):
    if not data:
        return "NA"
    return data.get("wandb_run_id") or "NA"


def triplet(data, metric, nfes=(1, 2, 4)):
    if not data:
        return "NA"
    values = [
        data.get(f"variant_full_nfe{nfe}", {}).get(metric)
        for nfe in nfes
    ]
    if any(value is None for value in values):
        return "NA"
    return "/".join(number(value) for value in values)


def eval_marker(experiment_root, tag, seed, kind, label):
    return (
        experiment_root
        / f"eval-{tag}-seed{seed}-{kind}-endpoint_normalized-{label}.json"
    )


def milestone_marker(experiment_root, tag, seed, kind, step):
    return eval_marker(experiment_root, tag, seed, kind, f"step{step}")


def training_run_id(output_dir):
    data = load_json(output_dir / "wandb_run_id.json")
    return data.get("run_id") if data else None


def checkpoint_progress(output_dir):
    checkpoint_path = output_dir / "ckpt-latest.pth"
    if not checkpoint_path.exists():
        return None
    try:
        import torch

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
        except (TypeError, RuntimeError):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        return {
            "updates": int(checkpoint["step"]) + 1,
            "best_loss": checkpoint.get("best_val_loss"),
            "best_step": checkpoint.get("best_val_step"),
        }
    except Exception as error:
        return {"error": f"{type(error).__name__}:{error}"}


def screen_line(experiment_root, tag):
    latest_path = milestone_marker(experiment_root, tag, 1, "latest", 10000)
    best_path = milestone_marker(experiment_root, tag, 1, "best", 10000)
    latest = load_json(latest_path)
    best = load_json(best_path)
    output_dir = experiment_root / f"{tag}_seed1"
    if latest:
        try:
            score = read_candidate(tag, latest_path)["score"]
        except (KeyError, TypeError, ValueError):
            score = None
        return (
            "DONE",
            f"score={number(score)} "
            f"latest_lpips={triplet(latest, 'lpips')} "
            f"latest_vertex={triplet(latest, 'final_block_vertex_error')} "
            f"best_lpips={triplet(best, 'lpips')} "
            f"eval_wandb={wandb_id(latest)}/{wandb_id(best)}",
        )
    progress = checkpoint_progress(output_dir)
    if progress and "updates" in progress:
        return (
            "RUN",
            f"updates={progress['updates']} "
            f"best_val={number(progress['best_loss'])}"
            f"@{progress['best_step'] if progress['best_step'] is not None else 'NA'} "
            f"train_wandb={training_run_id(output_dir) or 'NA'}",
        )
    if progress:
        return "ERROR", f"checkpoint={progress['error']}"
    return "WAIT", "updates=0"


def completed_milestones(experiment_root, tag, seed):
    return [
        step
        for step in MILESTONES
        if load_json(milestone_marker(
            experiment_root, tag, seed, "latest", step
        ))
    ]


def selected_seed_line(experiment_root, tag, seed):
    output_dir = experiment_root / f"{tag}_seed{seed}"
    progress = checkpoint_progress(output_dir)
    checkpoints = completed_milestones(experiment_root, tag, seed)
    latest_step = checkpoints[-1] if checkpoints else None
    checkpoint_updates = (
        progress.get("updates") if progress and "updates" in progress else None
    )
    updates = checkpoint_updates or latest_step
    latest = (
        load_json(milestone_marker(
            experiment_root, tag, seed, "latest", latest_step
        ))
        if latest_step
        else None
    )
    best_step = 100000 if 100000 in checkpoints else 10000 if 10000 in checkpoints else None
    best = (
        load_json(milestone_marker(
            experiment_root, tag, seed, "best", best_step
        ))
        if best_step
        else None
    )
    error = progress.get("error") if progress and "error" in progress else None
    return (
        f"  SEED seed={seed} updates={updates or 0} "
        f"milestones={','.join(map(str, checkpoints)) or 'none'} "
        f"latest_step={latest_step or 'NA'} "
        f"latest_lpips={triplet(latest, 'lpips')} "
        f"latest_vertex={triplet(latest, 'final_block_vertex_error')} "
        f"best_step={best_step or 'NA'} best_lpips={triplet(best, 'lpips')} "
        f"train_wandb={training_run_id(output_dir) or 'NA'}"
        + (f" checkpoint_error={error}" if error else "")
    )


def locked_line(experiment_root, tag, seed):
    latest = load_json(eval_marker(
        experiment_root, tag, seed, "latest", LOCKED_LABEL
    ))
    best = load_json(eval_marker(
        experiment_root, tag, seed, "best", LOCKED_LABEL
    ))
    if not latest and not best:
        return None
    return (
        f"  LOCKED seed={seed} "
        f"latest_lpips_1/2/4/8={triplet(latest, 'lpips', (1, 2, 4, 8))} "
        f"latest_vertex_1/2/4/8="
        f"{triplet(latest, 'final_block_vertex_error', (1, 2, 4, 8))} "
        f"best_lpips_1/2/4/8={triplet(best, 'lpips', (1, 2, 4, 8))} "
        f"eval_wandb={wandb_id(latest)}/{wandb_id(best)}"
    )


def report_node(role, experiment_root):
    spec = NODES[role]
    screen_states = []
    print(f"NODE role={role} question={spec['question']}")
    for tag in spec["candidates"]:
        state, details = screen_line(experiment_root, tag)
        screen_states.append(state)
        print(f"  SCREEN {state} tag={tag} {details}")

    selection_path = experiment_root / f"selection-{role}-step10000.json"
    selection = load_json(selection_path)
    if not selection:
        active = any(state in {"RUN", "DONE"} for state in screen_states)
        state = "screening" if active else "pending"
        print(
            f"  SELECTION WAIT screens_done={screen_states.count('DONE')}/"
            f"{len(screen_states)} marker={selection_path}"
        )
        print(f"NODE_STATUS role={role} state={state}")
        return state

    selected = selection["selected"]
    selected_score = next(
        (
            candidate.get("score")
            for candidate in selection.get("candidates", ())
            if candidate.get("name") == selected
        ),
        None,
    )
    print(
        f"  SELECTION DONE selected={selected} score={number(selected_score)} "
        f"screens_done={screen_states.count('DONE')}/{len(screen_states)}"
    )
    for seed in (1, 2, 3):
        print(selected_seed_line(experiment_root, selected, seed))
        locked = locked_line(experiment_root, selected, seed)
        if locked:
            print(locked)

    complete = all(
        load_json(eval_marker(
            experiment_root, selected, seed, kind, LOCKED_LABEL
        ))
        for seed in (1, 2, 3)
        for kind in ("latest", "best")
    )
    state = "complete" if complete else "selected_training"
    print(f"NODE_STATUS role={role} state={state} selected={selected}")
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scope",
        choices=(*NODES, "all"),
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

    experiment_root = args.asset_root / "checkpoints" / "experiments"
    roles = tuple(NODES) if args.scope == "all" else (args.scope,)
    print(
        f"long_research_report scope={args.scope} "
        f"shared_root={experiment_root} milestones="
        f"{','.join(map(str, MILESTONES))}"
    )
    states = [report_node(role, experiment_root) for role in roles]
    complete = states.count("complete")
    overall = "complete" if complete == len(states) else "running_or_incomplete"
    print(
        f"OVERALL state={overall} nodes_complete={complete}/{len(states)} "
        f"node_states={','.join(states)}"
    )


if __name__ == "__main__":
    main()
