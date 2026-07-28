#!/usr/bin/env python3
"""Print one terminal report covering every company experiment generation."""

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPORTERS = (
    ("OVERNIGHT_BASELINE_AND_ORIGINAL_DFM", "status_overnight.py"),
    ("CORRECTED_LONG_RESEARCH", "status_long_research.py"),
    ("AGGRESSIVE_WEEKEND_RESEARCH", "status_weekend_research.py"),
    ("ADVANTAGE_ALIGNED_HYPOTHESIS_AUDIT", "status_hypothesis_audit.py"),
    ("LOCKED_ADVANTAGE_FRONTIER", "status_advantage_frontier.py"),
    ("UNORDERED_GENERATIVE_FIDELITY", "status_unordered_fidelity.py"),
)


def repository_commit(repo_root):
    result = subprocess.run(
        ("git", "rev-parse", "--short", "HEAD"),
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "NA"


def reporter_command(python, script, asset_root, runtime_root):
    if script.name == "status_hypothesis_audit.py":
        return [python, str(script)]
    command = [
        python,
        str(script),
        "all",
        "--asset-root",
        str(asset_root),
    ]
    if script.name == "status_overnight.py":
        command.extend(("--runtime-root", str(runtime_root)))
    return command


def run_reporter(label, script, args, repo_root):
    print(f"\n===== {label} =====", flush=True)
    environment = os.environ.copy()
    environment["DRIFTFLOWWORLD_ASSET_ROOT"] = str(args.asset_root)
    environment["DRIFTFLOWWORLD_RUNTIME_ROOT"] = str(args.runtime_root)
    result = subprocess.run(
        reporter_command(
            args.python,
            script,
            args.asset_root,
            args.runtime_root,
        ),
        cwd=repo_root,
        env=environment,
    )
    print(f"reporter={script.name} exit_code={result.returncode}", flush=True)
    return result.returncode


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def inventory(args):
    experiment_root = args.asset_root / "checkpoints" / "experiments"
    checkpoints = sorted(
        path
        for name in ("ckpt-latest.pth", "ckpt-best.pth")
        for path in experiment_root.glob(f"*_seed*/{name}")
    )
    latest = [path for path in checkpoints if path.name == "ckpt-latest.pth"]
    best = [path for path in checkpoints if path.name == "ckpt-best.pth"]
    checkpoint_bytes = sum(path.stat().st_size for path in checkpoints)
    eval_markers = list(experiment_root.glob("eval-*.json"))
    complete_eval = sum(
        (load_json(path) or {}).get("status") == "complete"
        for path in eval_markers
    )
    training_markers = list(experiment_root.glob("*_seed*/complete-step*.json"))
    rollout_best = list(experiment_root.glob("*_seed*/rollout-best.json"))
    temporary = list(experiment_root.glob("*_seed*/*.tmp"))

    print("\n===== SHARED_ARTIFACT_INVENTORY =====")
    print(f"experiment_root={experiment_root}")
    print(
        f"checkpoints latest={len(latest)} best={len(best)} "
        f"total={len(checkpoints)} size_gib={checkpoint_bytes / 2**30:.2f}"
    )
    print(
        f"markers training_complete={len(training_markers)} "
        f"evaluation_complete={complete_eval}/{len(eval_markers)} "
        f"rollout_best={len(rollout_best)} temporary_files={len(temporary)}"
    )
    for label, path in (
        (
            "dataset",
            args.asset_root
            / "data"
            / "world_model_data"
            / "dataset_domain"
            / "all_data",
        ),
        (
            "official_checkpoint",
            args.asset_root
            / "checkpoints"
            / "official"
            / "pusht_checkpoints"
            / "pushT_driftworld"
            / "ckpt_save"
            / "ckpt-step1180500.pth",
        ),
    ):
        print(f"asset {label}={'READY' if path.exists() else 'MISSING'} path={path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(os.environ.get(
            "DRIFTFLOWWORLD_ASSET_ROOT",
            "/group-volume/danny-dataset/driftworld",
        )),
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(os.environ.get(
            "DRIFTFLOWWORLD_RUNTIME_ROOT",
            "/user-volume/driftworld",
        )),
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    print(
        "all_experiments_report "
        f"time={datetime.now().astimezone().isoformat(timespec='seconds')} "
        f"host={socket.gethostname()} commit={repository_commit(repo_root)}"
    )
    print(f"asset_root={args.asset_root} runtime_root={args.runtime_root}")

    failures = []
    for label, filename in REPORTERS:
        script = repo_root / "company" / filename
        returncode = run_reporter(label, script, args, repo_root)
        if returncode:
            failures.append(filename)

    inventory(args)
    print(
        "\n===== REPORT_STATUS =====\n"
        f"status={'complete' if not failures else 'reporter_failures'} "
        f"failed_reporters={','.join(failures) if failures else 'none'}"
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
