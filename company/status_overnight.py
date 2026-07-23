#!/usr/bin/env python3
"""Report concise completion, progress, and result status for company overnight runs."""

import argparse
import json
import os
import re
from pathlib import Path


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def format_number(value, digits=6):
    if value is None:
        return "NA"
    return f"{value:.{digits}g}"


def percent_change(current, reference):
    if current is None or reference in (None, 0):
        return "NA"
    return f"{100.0 * (current - reference) / reference:+.1f}%"


def process_alive(pid):
    try:
        os.kill(pid, 0)
    except (OSError, TypeError):
        return False
    cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        return "run_overnight.sh" in cmdline.read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return False


def tail_text(path, max_bytes=2 * 1024 * 1024):
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode(errors="replace")
    except OSError:
        return ""


def last_matching(lines, pattern):
    regex = re.compile(pattern, re.IGNORECASE)
    for line in reversed(lines):
        if regex.search(line):
            return line.strip()[-500:]
    return None


def queue_status(role, runtime_root):
    log_dir = runtime_root / "logs" / "overnight"
    pid_path = log_dir / f"{role}.pid"
    pid = None
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
        except (OSError, ValueError):
            pass
    logs = sorted(log_dir.glob(f"{role}-*.log"), key=lambda path: path.stat().st_mtime)
    latest_log = logs[-1] if logs else None
    text = tail_text(latest_log) if latest_log else ""
    lines = text.splitlines()
    complete = last_matching(lines, rf"\[overnight\] status=complete node={re.escape(role)}")
    error = last_matching(
        lines,
        r"Training failed|Asset preparation failed|Traceback \(most recent|"
        r"ChildFailedError|ModuleNotFoundError|RuntimeError:|ERROR:",
    )
    state = "running" if process_alive(pid) else "complete" if complete else "stopped"
    print(
        f"queue role={role} state={state} pid={pid or 'NA'} "
        f"log={latest_log or 'NA'}"
    )
    for label, pattern in (
        ("last_event", r"\[overnight\] (start|complete|waiting_for|status=)"),
        ("last_progress", r"step: [0-9]+ \| loss_backprop:"),
        ("last_validation", r"validation/loss:"),
        ("last_error", r"Training failed|Traceback \(most recent|ChildFailedError|"
                       r"ModuleNotFoundError|RuntimeError:|ERROR:"),
    ):
        value = last_matching(lines, pattern)
        if value:
            print(f"  {label}: {value}")
    return state, error


def training_tasks(scope, experiment_root, primary_steps, replication_steps,
                   ablation_steps, milestones, seeds):
    tasks = []
    if scope in ("node-a", "all"):
        output = experiment_root / "pushT_driftworld_continue_seed1"
        for step in milestones:
            tasks.append((f"control-seed1-step{step}", output, step))
        for seed in seeds:
            if seed != 1:
                tasks.append((
                    f"control-seed{seed}-step{replication_steps}",
                    experiment_root / f"pushT_driftworld_continue_seed{seed}",
                    replication_steps,
                ))
        tasks.append((
            f"driftflow-uniform-seed1-step{ablation_steps}",
            experiment_root / "driftflow-uniform_seed1",
            ablation_steps,
        ))
    if scope in ("node-b", "all"):
        output = experiment_root / "pushT_driftflow_posttrain_seed1"
        for step in milestones:
            tasks.append((f"driftflow-seed1-step{step}", output, step))
        for seed in seeds:
            if seed != 1:
                tasks.append((
                    f"driftflow-seed{seed}-step{replication_steps}",
                    experiment_root / f"pushT_driftflow_posttrain_seed{seed}",
                    replication_steps,
                ))
        tasks.append((
            f"driftflow-replay50-seed1-step{ablation_steps}",
            experiment_root / "driftflow-replay50_seed1",
            ablation_steps,
        ))
    return tasks


def report_training(tasks):
    missing = []
    output_dirs = set()
    print("training:")
    for label, output_dir, target in tasks:
        output_dirs.add(output_dir)
        marker = output_dir / f"complete-step{target}.json"
        data = load_json(marker)
        if not data or data.get("status") != "complete":
            missing.append((label, marker))
            print(f"  MISSING {label} marker={marker}")
            continue
        print(
            f"  DONE {label} checkpoint_step={data.get('step', 'NA')} "
            f"best={format_number(data.get('best_validation_loss'))}"
            f"@{data.get('best_validation_step', 'NA')} "
            f"last_loss={format_number(data.get('last_logged_loss'))} "
            f"wandb={data.get('wandb_run_id') or 'NA'}"
        )

    checkpoint_paths = {
        output_dir / name
        for output_dir in output_dirs
        for name in ("ckpt-latest.pth", "ckpt-best.pth")
        if (output_dir / name).exists()
    }
    total_bytes = sum(path.stat().st_size for path in checkpoint_paths)
    print(
        f"retained_checkpoints count={len(checkpoint_paths)} "
        f"size_gib={total_bytes / 2**30:.2f}"
    )
    return missing


def metric(data, key, name="lpips"):
    section = data.get(key, {}) if data else {}
    return section.get(name)


def report_evaluations(experiment_root, milestones, primary_steps):
    expected = [(step, "latest") for step in milestones]
    expected.append((primary_steps, "best"))
    missing = []
    print("rollout_evaluation (lower LPIPS is better):")
    for step, kind in expected:
        label = f"eval-seed1-step{step}-{kind}"
        marker = experiment_root / f"{label}.json"
        data = load_json(marker)
        if not data or data.get("status") != "complete":
            missing.append((label, marker))
            print(f"  MISSING {label} marker={marker}")
            continue
        control = metric(data, "control_full")
        drift1 = metric(data, "driftflow_full_nfe1")
        drift2 = metric(data, "driftflow_full_nfe2")
        drift4 = metric(data, "driftflow_full_nfe4")
        vertex1 = metric(data, "driftflow_full_nfe1", "final_block_vertex_error")
        vertex4 = metric(data, "driftflow_full_nfe4", "final_block_vertex_error")
        print(
            f"  DONE step={step} checkpoint={kind} full_lpips "
            f"control={format_number(control)} drift_nfe1={format_number(drift1)} "
            f"nfe2={format_number(drift2)} nfe4={format_number(drift4)} "
            f"nfe1_vs_control={percent_change(drift1, control)} "
            f"nfe4_vs_nfe1={percent_change(drift4, drift1)} "
            f"vertex_nfe1={format_number(vertex1)} "
            f"vertex_nfe4={format_number(vertex4)}"
        )
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scope", choices=("node-a", "node-b", "all"), nargs="?", default="all")
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
    args = parser.parse_args()

    primary_steps = int(os.environ.get("OVERNIGHT_PRIMARY_STEPS", "30000"))
    replication_steps = int(os.environ.get("OVERNIGHT_REPLICATION_STEPS", "10000"))
    ablation_steps = int(os.environ.get("OVERNIGHT_ABLATION_STEPS", "10000"))
    seeds = [int(value) for value in os.environ.get("OVERNIGHT_SEEDS", "1 2 3").split()]
    milestones = [
        int(value)
        for value in os.environ.get("OVERNIGHT_MILESTONES", "10000 20000").split()
    ]
    if primary_steps not in milestones:
        milestones.append(primary_steps)
    milestones = [step for step in milestones if step <= primary_steps]

    experiment_root = args.asset_root / "checkpoints" / "experiments"
    print(
        f"overnight_report scope={args.scope} primary={primary_steps} "
        f"replication={replication_steps} ablation={ablation_steps} "
        f"milestones={','.join(map(str, milestones))}"
    )

    roles = ("node-a", "node-b") if args.scope == "all" else (args.scope,)
    queue_states = []
    errors = []
    for role in roles:
        state, error = queue_status(role, args.runtime_root)
        queue_states.append(state)
        if error:
            errors.append((role, error))

    tasks = training_tasks(
        args.scope,
        experiment_root,
        primary_steps,
        replication_steps,
        ablation_steps,
        milestones,
        seeds,
    )
    missing = report_training(tasks)
    missing += report_evaluations(experiment_root, milestones, primary_steps)

    if not missing:
        overall = "complete"
    elif "running" in queue_states:
        overall = "running"
    elif errors:
        overall = "failed_or_interrupted"
    else:
        overall = "incomplete_or_interrupted"
    print(f"overall={overall} missing={len(missing)}")


if __name__ == "__main__":
    main()
