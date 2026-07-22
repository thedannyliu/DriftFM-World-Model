#!/usr/bin/env python3
"""Summarize baseline or pilot rollout JSON files without verbose logs."""

import argparse
import json
from pathlib import Path


def read_metrics(path):
    metrics = json.loads(path.read_text())
    keys = ("mse", "lpips", "seconds_per_frame", "final_block_vertex_error")
    return {key: metrics[key] for key in keys if key in metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--control-dir", type=Path)
    parser.add_argument("--driftflow-dir", type=Path)
    args = parser.parse_args()

    if args.baseline_dir:
        result = {
            "status": "complete",
            "baseline_64": read_metrics(args.baseline_dir / "rollout_len-64_nfe-1.json"),
            "baseline_full": read_metrics(args.baseline_dir / "rollout_len-full_nfe-1.json"),
        }
    else:
        result = {"status": "complete"}
        for length in ("64", "full"):
            result[f"control_{length}"] = read_metrics(
                args.control_dir / f"rollout_len-{length}_nfe-1.json"
            )
            for nfe in (1, 2, 4):
                result[f"driftflow_{length}_nfe{nfe}"] = read_metrics(
                    args.driftflow_dir / f"rollout_len-{length}_nfe-{nfe}.json"
                )
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
