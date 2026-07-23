#!/usr/bin/env python3
"""Print compact transport-audit results suitable for terminal handoff."""

import argparse
import json
from pathlib import Path


def _ratio(value, reference):
    return value / reference if reference else None


def summarize(name, path):
    result = json.loads(path.read_text())
    metrics = result["metrics"]
    nfe1 = metrics["free/nfe1/final/paired_mse"]
    nfe2 = metrics["free/nfe2/final/paired_mse"]
    nfe4 = metrics["free/nfe4/final/paired_mse"]
    return {
        "name": name,
        "step": result["checkpoint_step"],
        "gpu": result.get("gpu"),
        "free_mse": {"nfe1": nfe1, "nfe2": nfe2, "nfe4": nfe4},
        "free_vs_nfe1": {
            "nfe2": _ratio(nfe2, nfe1),
            "nfe4": _ratio(nfe4, nfe1),
        },
        "teacher_progress": {
            "0-.5": metrics["teacher/0-0.5/progress_ratio"],
            ".5-1": metrics["teacher/0.5-1/progress_ratio"],
            ".75-1": metrics["teacher/0.75-1/progress_ratio"],
        },
        "teacher_mse": {
            "0-.5": metrics["teacher/0-0.5/paired_mse"],
            ".5-1": metrics["teacher/0.5-1/paired_mse"],
            ".75-1": metrics["teacher/0.75-1/paired_mse"],
        },
        "teacher_particle_mean_mse": {
            "0-.5": metrics["teacher/0-0.5/particle_mean_mse"],
            ".5-1": metrics["teacher/0.5-1/particle_mean_mse"],
            ".75-1": metrics["teacher/0.75-1/particle_mean_mse"],
        },
        "time_sensitivity": {
            ".5-1_cosine_to_endpoint": metrics[
                "sensitivity/0.5-1/cosine_to_endpoint"
            ],
            ".5-1_mean_abs_difference": metrics[
                "sensitivity/0.5-1/mean_abs_difference"
            ],
        },
        "wandb": result.get("wandb_run_id"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="NAME=PATH",
    )
    args = parser.parse_args()
    summaries = []
    for item in args.result:
        name, separator, path = item.partition("=")
        if not separator:
            parser.error("--result must use NAME=PATH")
        summaries.append(summarize(name, Path(path)))
    print(json.dumps({"status": "complete", "runs": summaries}, separators=(",", ":")))


if __name__ == "__main__":
    main()
